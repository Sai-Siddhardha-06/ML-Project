import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# Configuration

ROOT = Path("frames/ffpp")

MODEL_NAME = "facebook/dinov2-base"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
GRADIENT_ACCUMULATION_STEPS = 2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_PATH = Path("dinov2_linear_probe_best.pt")


# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# Dataset

class FFPPFrameDataset(Dataset):
    """
    Frame-level dataset.

    Each sample corresponds to one preprocessed face frame.
    The video_id is retained so that predictions can later be
    aggregated to video-level.
    """

    def __init__(self, root, split, transform):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.samples = []

        split_dir = self.root / split

        for video_dir in sorted(split_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            meta_path = video_dir / "meta.json"

            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            label = int(meta["label"])
            video_id = meta["video_id"]

            frames = sorted(video_dir.glob("*.jpg"))

            for frame_path in frames:
                self.samples.append(
                    {
                        "frame_path": frame_path,
                        "label": label,
                        "video_id": video_id,
                    }
                )

        print(
            f"{split}: "
            f"{len(self.samples)} frames "
            f"from {len(set(x['video_id'] for x in self.samples))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample["frame_path"]).convert("RGB")
        image = self.transform(image)

        return (
            image,
            sample["label"],
            sample["video_id"],
        )


# Transforms

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# DINOv2 processor provides the correct normalization.
image_mean = processor.image_mean
image_std = processor.image_std

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ]
)


# Model

class DINOv2LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(MODEL_NAME)

        # Freeze DINOv2 completely.
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values)

        # CLS token
        cls_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_embedding)

        return logits


# Evaluation

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    fnr = 1.0 - tpr

    index = np.nanargmin(np.abs(fnr - fpr))

    eer = (fpr[index] + fnr[index]) / 2.0

    return eer


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    video_scores = {}
    video_labels = {}

    for images, labels, video_ids in tqdm(
        loader,
        desc="Evaluating",
        leave=False,
    ):
        images = images.to(DEVICE, non_blocking=True)

        logits = model(images)

        probabilities = torch.softmax(logits, dim=1)[:, 1]

        probabilities = probabilities.cpu().numpy()
        labels = labels.numpy()

        for video_id, score, label in zip(
            video_ids,
            probabilities,
            labels,
        ):
            video_scores.setdefault(video_id, []).append(float(score))
            video_labels[video_id] = int(label)

    y_true = []
    y_score = []

    for video_id in video_scores:
        y_true.append(video_labels[video_id])
        y_score.append(np.mean(video_scores[video_id]))

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    auc = roc_auc_score(y_true, y_score)
    eer = calculate_eer(y_true, y_score)

    return auc, eer


# Main

def main():

    print("=" * 60)
    print("DINOv2-Base + Linear Probe Baseline")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Datasets

    train_dataset = FFPPFrameDataset(
        ROOT,
        "train",
        train_transform,
    )

    val_dataset = FFPPFrameDataset(
        ROOT,
        "val",
        eval_transform,
    )

    test_dataset = FFPPFrameDataset(
        ROOT,
        "test",
        eval_transform,
    )

    # DataLoaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # Model

    model = DINOv2LinearProbe().to(DEVICE)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(f"Total parameters:     {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.4f} M")

    # Loss / optimizer

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    # Training

    best_val_auc = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
        )

        for images, labels, _ in progress:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}",
            )

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation

        val_auc, val_eer = evaluate(
            model,
            val_loader,
        )

        print()
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_AUC={val_auc:.4f}, "
            f"val_EER={val_eer:.4f}"
        )

        # Save best model

        if val_auc > best_val_auc:

            best_val_auc = val_auc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "val_eer": val_eer,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"Saved best model → {BEST_MODEL_PATH}"
            )

    # Test

    print()
    print("=" * 60)
    print("Loading best checkpoint")
    print("=" * 60)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_auc, test_eer = evaluate(
        model,
        test_loader,
    )

    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test AUC : {test_auc:.4f}")
    print(f"Test EER : {test_eer:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
