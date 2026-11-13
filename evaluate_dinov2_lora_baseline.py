import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-base"

FRAME_ROOT = Path("frames/ffpp")
CHECKPOINT = Path(
    "checkpoints/dinov2_lora_baseline/best.pt"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class FFPPFrameDataset(Dataset):

    def __init__(self, root, split, transform):

        self.samples = []
        split_root = Path(root) / split

        video_dirs = sorted(
            p for p in split_root.iterdir()
            if p.is_dir()
        )

        for video_dir in video_dirs:

            meta_path = video_dir / "meta.json"

            if not meta_path.exists():
                continue

            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta.get("status") != "ok":
                continue

            label = int(meta["label"])
            video_id = meta["video_id"]

            for frame_path in sorted(
                video_dir.glob("*.jpg")
            ):

                self.samples.append(
                    (
                        frame_path,
                        label,
                        video_id,
                    )
                )

        self.transform = transform

        print(
            f"{split}: "
            f"{len(self.samples)} frames from "
            f"{len(set(x[2] for x in self.samples))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        frame_path, label, video_id = self.samples[idx]

        image = Image.open(
            frame_path
        ).convert("RGB")

        image = self.transform(image)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32,
            ),
            video_id,
        )


class LoRALinear(nn.Module):

    def __init__(
        self,
        original_layer,
        rank,
        alpha,
        dropout=0.0,
    ):
        super().__init__()

        self.original = original_layer

        for param in self.original.parameters():
            param.requires_grad = False

        self.scaling = alpha / rank

        self.lora_dropout = nn.Dropout(
            dropout
        )

        self.lora_A = nn.Linear(
            original_layer.in_features,
            rank,
            bias=False,
        )

        self.lora_B = nn.Linear(
            rank,
            original_layer.out_features,
            bias=False,
        )

    def forward(self, x):

        original_output = self.original(x)

        lora_output = self.lora_B(
            self.lora_A(
                self.lora_dropout(x)
            )
        )

        return (
            original_output
            + self.scaling * lora_output
        )


def inject_lora(
    model,
    rank=16,
    alpha=32,
    dropout=0.0,
):

    attention = (
        model.encoder.layer[-1]
        .attention.attention
    )

    attention.query = LoRALinear(
        attention.query,
        rank,
        alpha,
        dropout,
    )

    attention.value = LoRALinear(
        attention.value,
        rank,
        alpha,
        dropout,
    )


class LoRABaseline(nn.Module):

    def __init__(self, dino):

        super().__init__()

        self.dino = dino

        self.classifier = nn.Linear(
            768,
            1,
        )

    def forward(self, pixel_values):

        outputs = self.dino(
            pixel_values=pixel_values
        )

        cls_embedding = (
            outputs.last_hidden_state[:, 0]
        )

        logits = self.classifier(
            cls_embedding
        )

        return logits.squeeze(1)


def compute_eer(labels, scores):

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
    )

    fnr = 1.0 - tpr

    index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = (
        fpr[index] + fnr[index]
    ) / 2.0

    return eer, thresholds[index]


@torch.no_grad()
def main():

    print("=" * 60)
    print("CP6 — DINOv2 + Plain LoRA Evaluation")
    print("=" * 60)

    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {CHECKPOINT}")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        ),
    ])

    test_dataset = FFPPFrameDataset(
        FRAME_ROOT,
        "test",
        transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        prefetch_factor=(
            2 if NUM_WORKERS > 0 else None
        ),
    )

    print()
    print("Loading DINOv2...")

    dino = AutoModel.from_pretrained(
        MODEL_NAME
    )

    for param in dino.parameters():
        param.requires_grad = False

    inject_lora(
        dino,
        rank=16,
        alpha=32,
        dropout=0.0,
    )

    model = LoRABaseline(
        dino
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Best epoch  : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best val AUC: "
        f"{checkpoint['val_auc']:.5f}"
    )

    print()
    print("Running test evaluation...")

    all_labels = []
    all_scores = []
    all_video_ids = []

    for step, (
        images,
        labels,
        video_ids,
    ) in enumerate(
        test_loader,
        start=1,
    ):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda",
        ):

            logits = model(images)

        scores = torch.sigmoid(
            logits
        ).cpu().numpy()

        all_labels.extend(
            labels.numpy()
        )

        all_scores.extend(
            scores
        )

        all_video_ids.extend(
            video_ids
        )

        if step % 500 == 0:
            print(
                f"Batch "
                f"{step}/{len(test_loader)}",
                flush=True,
            )

    all_labels = np.asarray(
        all_labels
    )

    all_scores = np.asarray(
        all_scores
    )

    frame_auc = roc_auc_score(
        all_labels,
        all_scores,
    )

    # Aggregate frame scores by video.
    video_scores = {}
    video_labels = {}

    for video_id, label, score in zip(
        all_video_ids,
        all_labels,
        all_scores,
    ):

        video_scores.setdefault(
            video_id,
            [],
        ).append(score)

        video_labels[
            video_id
        ] = label

    video_ids = sorted(
        video_scores.keys()
    )

    video_scores_array = np.asarray([
        np.mean(video_scores[v])
        for v in video_ids
    ])

    video_labels_array = np.asarray([
        video_labels[v]
        for v in video_ids
    ])

    video_auc = roc_auc_score(
        video_labels_array,
        video_scores_array,
    )

    eer, eer_threshold = compute_eer(
        video_labels_array,
        video_scores_array,
    )

    print()
    print("=" * 60)
    print("CP6 TEST RESULTS")
    print("=" * 60)

    print(
        f"Test frame AUC  : "
        f"{frame_auc:.5f}"
    )

    print(
        f"Test video AUC  : "
        f"{video_auc:.5f}"
    )

    print(
        f"Test video EER  : "
        f"{eer:.5f}"
    )

    print(
        f"EER threshold   : "
        f"{eer_threshold:.5f}"
    )

    print(
        f"Videos evaluated: "
        f"{len(video_ids)}"
    )

    print(
        f"Real/Fake       : "
        f"{np.sum(video_labels_array == 0)}/"
        f"{np.sum(video_labels_array == 1)}"
    )

    print(
        f"Frames/video    : "
        f"{min(len(v) for v in video_scores.values())}-"
        f"{max(len(v) for v in video_scores.values())}"
    )


if __name__ == "__main__":
    main()
