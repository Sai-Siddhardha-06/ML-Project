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


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "facebook/dinov2-base"

FRAME_ROOT = Path("frames/ffpp")

CHECKPOINT = Path(
    "checkpoints/dinov2_lora_baseline/best.pt"
)

OUTPUT_ROOT = Path("evaluation")

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 1

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class FFPPFrameDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        transform,
    ):

        self.samples = []
        self.transform = transform

        split_root = Path(root) / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Missing split directory: {split_root}"
            )

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

            frame_paths = sorted(
                video_dir.glob("*.jpg")
            )

            for frame_path in frame_paths:

                self.samples.append(
                    (
                        frame_path,
                        label,
                        video_id,
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid frames found in {split_root}"
            )

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


# ============================================================
# LORA
# ============================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        original_layer,
        rank,
        alpha,
        dropout=0.0,
    ):

        super().__init__()

        if not isinstance(
            original_layer,
            nn.Linear,
        ):
            raise TypeError(
                "LoRALinear requires nn.Linear"
            )

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
    rank,
    alpha,
    dropout,
):

    attention = (
        model.encoder.layer[-1]
        .attention.attention
    )

    attention.query = LoRALinear(
        attention.query,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )

    attention.value = LoRALinear(
        attention.value,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


# ============================================================
# MODEL
# ============================================================

class LoRABaseline(nn.Module):

    def __init__(
        self,
        dino,
        hidden_size=768,
    ):

        super().__init__()

        self.dino = dino

        self.classifier = nn.Linear(
            hidden_size,
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


# ============================================================
# EER
# ============================================================

def compute_eer(
    labels,
    scores,
):

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

    return (
        float(eer),
        float(thresholds[index]),
    )


# ============================================================
# GENERIC EVALUATOR
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
):

    model.eval()

    all_labels = []
    all_scores = []
    all_video_ids = []

    for step, (
        images,
        labels,
        video_ids,
    ) in enumerate(
        loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
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

        if (
            step % 500 == 0
            or step == len(loader)
        ):

            print(
                f"Evaluation batch "
                f"{step}/{len(loader)}",
                flush=True,
            )

    frame_labels = np.asarray(
        all_labels
    )

    frame_scores = np.asarray(
        all_scores
    )

    frame_auc = roc_auc_score(
        frame_labels,
        frame_scores,
    )

    # --------------------------------------------------------
    # Video-level aggregation
    # --------------------------------------------------------

    video_scores_dict = {}
    video_labels_dict = {}

    for video_id, label, score in zip(
        all_video_ids,
        frame_labels,
        frame_scores,
    ):

        video_scores_dict.setdefault(
            video_id,
            [],
        ).append(
            float(score)
        )

        video_labels_dict[
            video_id
        ] = int(label)

    video_ids = sorted(
        video_scores_dict.keys()
    )

    video_scores = np.asarray([
        np.mean(
            video_scores_dict[video_id]
        )
        for video_id in video_ids
    ])

    video_labels = np.asarray([
        video_labels_dict[video_id]
        for video_id in video_ids
    ])

    video_auc = roc_auc_score(
        video_labels,
        video_scores,
    )

    video_eer, eer_threshold = compute_eer(
        video_labels,
        video_scores,
    )

    return {
        "frame_auc": float(frame_auc),
        "video_auc": float(video_auc),
        "video_eer": float(video_eer),
        "eer_threshold": float(eer_threshold),
        "frame_labels": frame_labels,
        "frame_scores": frame_scores,
        "video_labels": video_labels,
        "video_scores": video_scores,
        "video_ids": video_ids,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        OUTPUT_ROOT
        / "cp6_test_predictions.npz"
    )

    results_path = (
        OUTPUT_ROOT
        / "cp6_test_results.json"
    )

    np.savez_compressed(
        predictions_path,
        frame_labels=results["frame_labels"],
        frame_scores=results["frame_scores"],
        video_labels=results["video_labels"],
        video_scores=results["video_scores"],
        video_ids=np.asarray(
            results["video_ids"],
            dtype=str,
        ),
    )

    metrics = {
        "checkpoint": str(CHECKPOINT),
        "frame_auc": results["frame_auc"],
        "video_auc": results["video_auc"],
        "video_eer": results["video_eer"],
        "eer_threshold": results["eer_threshold"],
        "videos_evaluated": len(
            results["video_ids"]
        ),
        "real_videos": int(
            np.sum(
                results["video_labels"] == 0
            )
        ),
        "fake_videos": int(
            np.sum(
                results["video_labels"] == 1
            )
        ),
        "frames_per_video": 32,
        "aggregation": "mean",
    }

    with open(
        results_path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print()
    print(
        f"Predictions saved: "
        f"{predictions_path}"
    )

    print(
        f"Results saved    : "
        f"{results_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CP7 — Generic Evaluation Infrastructure")
    print("=" * 60)

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Checkpoint : {CHECKPOINT}"
    )

    print()

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load DINOv2
    # --------------------------------------------------------

    print("Loading DINOv2...")

    dino = AutoModel.from_pretrained(
        MODEL_NAME
    )

    for param in dino.parameters():
        param.requires_grad = False

    inject_lora(
        dino,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
    )

    model = LoRABaseline(
        dino=dino,
        hidden_size=768,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load CP6 checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch  : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best val AUC: "
        f"{checkpoint['val_auc']:.5f}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("Running evaluation...")

    results = evaluate_model(
        model=model,
        loader=test_loader,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CP7 EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Frame AUC      : "
        f"{results['frame_auc']:.5f}"
    )

    print(
        f"Video AUC      : "
        f"{results['video_auc']:.5f}"
    )

    print(
        f"Video EER      : "
        f"{results['video_eer']:.5f}"
    )

    print(
        f"EER threshold  : "
        f"{results['eer_threshold']:.5f}"
    )

    print(
        f"Videos evaluated: "
        f"{len(results['video_ids'])}"
    )

    print(
        f"Real/Fake      : "
        f"{np.sum(results['video_labels'] == 0)}/"
        f"{np.sum(results['video_labels'] == 1)}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(results)

    print()
    print("=" * 60)
    print("CP7 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
