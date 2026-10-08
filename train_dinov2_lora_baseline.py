import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel


# CONFIG

MODEL_NAME = "facebook/dinov2-base"

FRAME_ROOT = Path("frames/ffpp")
OUTPUT_ROOT = Path("checkpoints/dinov2_lora_baseline")

IMAGE_SIZE = 224

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# Training configuration
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 1

SEED = 42

# Mixed precision
USE_AMP = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# REPRODUCIBILITY

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# DATASET

class FFPPFrameDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        transform=None,
        max_videos=None,
    ):

        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_root = self.root / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Missing split directory: {split_root}"
            )

        self.samples = []

        video_dirs = sorted(
            [
                p for p in split_root.iterdir()
                if p.is_dir()
            ]
        )

        if max_videos is not None:
            video_dirs = video_dirs[:max_videos]

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
                f"No valid frames found for split: {split}"
            )

        print(
            f"{split}: "
            f"{len(self.samples)} frames "
            f"from {len(set(x[2] for x in self.samples))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        frame_path, label, video_id = self.samples[idx]

        image = Image.open(frame_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.float32
        )


# LORA

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
            nn.Linear
        ):
            raise TypeError(
                "LoRALinear requires nn.Linear"
            )

        self.original = original_layer

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_dropout = nn.Dropout(dropout)

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

        # Standard LoRA initialization:
        # A random, B zero -> initial output identical
        # to original frozen layer.
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=np.sqrt(5),
        )

        nn.init.zeros_(
            self.lora_B.weight
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


# INSERT LORA INTO FINAL DINOv2 BLOCK

def inject_lora_into_last_block(
    model,
    rank,
    alpha,
    dropout,
):

    # Hugging Face DINOv2 structure:
    #
    # model.encoder.layer[-1].attention.attention.query
    # model.encoder.layer[-1].attention.attention.value
    #
    # We verify the structure instead of silently assuming it.

    last_block = model.encoder.layer[-1]

    attention = last_block.attention.attention

    if not hasattr(attention, "query"):
        raise AttributeError(
            "Could not find DINOv2 query projection."
        )

    if not hasattr(attention, "value"):
        raise AttributeError(
            "Could not find DINOv2 value projection."
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

    return last_block


# CLASSIFIER

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

        # CLS token
        cls_embedding = (
            outputs.last_hidden_state[:, 0]
        )

        logits = self.classifier(
            cls_embedding
        )

        return logits.squeeze(1)


# PARAMETER REPORT

def print_parameter_report(model):

    total = 0
    trainable = 0

    for name, param in model.named_parameters():

        n = param.numel()

        total += n

        if param.requires_grad:
            trainable += n

    print()
    print("PARAMETER REPORT")
    print("-" * 60)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(
        f"Trainable percentage : "
        f"{100.0 * trainable / total:.4f}%"
    )

    print()
    print("TRAINABLE PARAMETERS")
    print("-" * 60)

    for name, param in model.named_parameters():

        if param.requires_grad:
            print(
                f"{name:<70} "
                f"{param.numel():,}"
            )

    print()


# EVALUATION

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_scores = []

    for pixel_values, labels in loader:

        pixel_values = pixel_values.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(pixel_values)

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_labels.append(
            labels.cpu()
        )

        all_scores.append(
            torch.sigmoid(logits).cpu()
        )

    avg_loss = (
        total_loss / total_samples
    )

    all_labels = torch.cat(
        all_labels
    ).numpy()

    all_scores = torch.cat(
        all_scores
    ).numpy()

    auc = roc_auc_score(
        all_labels,
        all_scores,
    )

    return avg_loss, auc


# MAIN

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny training test.",
    )

    parser.add_argument(
        "--smoke-videos",
        type=int,
        default=4,
        help="Number of videos per split in smoke test.",
    )

    args = parser.parse_args()

    set_seed(SEED)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("DINOv2-Base + Plain LoRA Baseline")
    print("=" * 60)

    print(f"Device        : {DEVICE}")
    print(f"Model         : {MODEL_NAME}")
    print(f"LoRA rank     : {LORA_RANK}")
    print(f"LoRA alpha    : {LORA_ALPHA}")
    print(f"LoRA dropout  : {LORA_DROPOUT}")
    print(f"LoRA target   : final block Q/V")
    print()

    if args.smoke_test:

        epochs = 1
        max_videos = args.smoke_videos

        print("MODE          : SMOKE TEST")
        print(f"Videos/split  : {max_videos}")
        print(f"Epochs        : {epochs}")

    else:

        epochs = EPOCHS
        max_videos = None

        print("MODE          : FULL TRAINING")
        print(f"Epochs        : {epochs}")

    print()

    # Processor

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    image_mean = processor.image_mean
    image_std = processor.image_std

    # Training augmentation
    train_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.RandomHorizontalFlip(
            p=0.5
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_mean,
            std=image_std,
        ),
    ])

    # Validation: no random augmentation
    val_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_mean,
            std=image_std,
        ),
    ])

    # Datasets

    train_dataset = FFPPFrameDataset(
        FRAME_ROOT,
        "train",
        transform=train_transform,
        max_videos=max_videos,
    )

    val_dataset = FFPPFrameDataset(
        FRAME_ROOT,
        "val",
        transform=val_transform,
        max_videos=max_videos,
    )

    print()

    # DataLoaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    # Load DINOv2

    print("Loading DINOv2...")

    dino = AutoModel.from_pretrained(
        MODEL_NAME
    )

    # Freeze everything
    for param in dino.parameters():
        param.requires_grad = False

    dino = dino.to(DEVICE)

    # Inject LoRA

    print("Injecting LoRA into final block Q/V...")

    last_block = inject_lora_into_last_block(
        dino,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
    )

    # Complete model

    model = LoRABaseline(
        dino=dino,
        hidden_size=768,
    ).to(DEVICE)

    # Parameter report

    print_parameter_report(model)

    # Loss

    criterion = nn.BCEWithLogitsLoss()

    # Optimizer

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # AMP

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

    scaler = torch.cuda.amp.GradScaler(
    enabled=amp_enabled,
    )

    print(
        f"AMP enabled: {amp_enabled}"
    )

    print()

    # Smoke-test forward pass

    print("Running initial forward pass...")

    first_batch = next(
        iter(train_loader)
    )

    first_images, first_labels = first_batch

    first_images = first_images.to(
        DEVICE,
        non_blocking=True,
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp_enabled,
    ):

        first_logits = model(
            first_images
        )

    print(
        f"Input shape : {tuple(first_images.shape)}"
    )

    print(
        f"Output shape: {tuple(first_logits.shape)}"
    )

    assert first_logits.shape == (
        first_images.shape[0],
    )

    print("Forward pass: PASS")

    del first_images
    del first_logits
    del first_batch

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print()

    # Training

    best_val_auc = -float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        total_samples = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, (
            pixel_values,
            labels,
        ) in enumerate(train_loader, start=1):

            pixel_values = pixel_values.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):

                logits = model(
                    pixel_values
                )

                loss = criterion(
                    logits,
                    labels,
                )

                loss_for_backward = (
                    loss
                    / GRADIENT_ACCUMULATION_STEPS
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            if (
                step % GRADIENT_ACCUMULATION_STEPS == 0
                or step == len(train_loader)
            ):

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            if step % 500 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"batch {step}/{len(train_loader)} | "
                    f"loss={loss.item():.5f}",
                    flush=True,
                )

        train_loss = (
            running_loss / total_samples
        )

        val_loss, val_auc = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_AUC={val_auc:.5f}"
        )

        # Last checkpoint

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "config": {
                "model_name": MODEL_NAME,
                "image_size": IMAGE_SIZE,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps":
                    GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "seed": SEED,
            },
        }

        torch.save(
            last_checkpoint,
            OUTPUT_ROOT / "last.pt",
        )

        # Best checkpoint

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            best_epoch = epoch

            torch.save(
                last_checkpoint,
                OUTPUT_ROOT / "best.pt",
            )

            print(
                f"  -> New best model saved "
                f"(epoch {epoch}, "
                f"val AUC={val_auc:.5f})"
            )

        # GPU memory

        if DEVICE.type == "cuda":

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

            print(
                f"  GPU memory: "
                f"{allocated:.2f} GB allocated, "
                f"{reserved:.2f} GB reserved"
            )

    # Final summary

    print()
    print("=" * 60)
    print("CP6 TRAINING COMPLETE")
    print("=" * 60)

    print(f"Best epoch  : {best_epoch}")
    print(f"Best val AUC: {best_val_auc:.5f}")

    print()
    print(
        f"Best checkpoint: "
        f"{OUTPUT_ROOT / 'best.pt'}"
    )

    print(
        f"Last checkpoint: "
        f"{OUTPUT_ROOT / 'last.pt'}"
    )


if __name__ == "__main__":
    main()
