import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


# CONFIG

EMBEDDING_ROOT = Path("embeddings/dinov2_base")
OUTPUT_ROOT = Path("checkpoints/dinov2_linear_probe")

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# REPRODUCIBILITY

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# LOAD EMBEDDINGS

def load_split(split):
    path = EMBEDDING_ROOT / split / "embeddings.pt"

    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")

    data = torch.load(path, map_location="cpu")

    embeddings = data["embeddings"].float()
    labels = data["labels"].float()

    print(f"{split}:")
    print(f"  embeddings : {tuple(embeddings.shape)}")
    print(f"  labels     : {tuple(labels.shape)}")
    print(f"  videos     : {len(data['video_ids'])}")

    return embeddings, labels, data["video_ids"]


# MODEL

class LinearProbe(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()

        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.classifier(x).squeeze(1)


# EVALUATION

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_scores = []

    for embeddings, labels in loader:
        embeddings = embeddings.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(embeddings)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.append(labels.cpu())
        all_scores.append(torch.sigmoid(logits).cpu())

    avg_loss = total_loss / total_samples

    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.cat(all_scores).numpy()

    auc = roc_auc_score(all_labels, all_scores)

    return avg_loss, auc


# MAIN

def main():
    set_seed(SEED)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DINOv2-Base + Linear Probe")
    print("=" * 60)

    print(f"Device       : {DEVICE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay : {WEIGHT_DECAY}")
    print()

    # Load cached embeddings

    train_x, train_y, train_video_ids = load_split("train")
    val_x, val_y, val_video_ids = load_split("val")

    print()

    # Basic integrity checks

    assert train_x.ndim == 2
    assert train_x.shape[1] == 768
    assert val_x.ndim == 2
    assert val_x.shape[1] == 768

    assert len(train_x) == len(train_y)
    assert len(val_x) == len(val_y)

    assert set(train_y.unique().tolist()).issubset({0.0, 1.0})
    assert set(val_y.unique().tolist()).issubset({0.0, 1.0})

    print("Integrity checks: PASS")
    print()

    # DataLoaders

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Model

    model = LinearProbe(input_dim=768).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("Model:")
    print(model)
    print()

    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))
    print()

    # Training

    best_val_auc = -float("inf")
    best_epoch = -1

    history = []

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for embeddings, labels in train_loader:

            embeddings = embeddings.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(embeddings)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / total_samples

        val_loss, val_auc = evaluate(
            model,
            val_loader,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_AUC={val_auc:.5f}"
        )

        # Save last checkpoint

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "config": {
                "input_dim": 768,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "seed": SEED,
            },
        }

        torch.save(
            last_checkpoint,
            OUTPUT_ROOT / "last.pt",
        )

        # Save best checkpoint

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            best_epoch = epoch

            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "config": {
                    "input_dim": 768,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                },
            }

            torch.save(
                best_checkpoint,
                OUTPUT_ROOT / "best.pt",
            )

            print(
                f"  -> New best model saved "
                f"(epoch {epoch}, val AUC={val_auc:.5f})"
            )

    # Save training history

    history_path = OUTPUT_ROOT / "history.pt"

    torch.save(history, history_path)

    # Final summary

    print()
    print("=" * 60)
    print("LINEAR PROBE TRAINING COMPLETE")
    print("=" * 60)

    print(f"Best epoch : {best_epoch}")
    print(f"Best val AUC: {best_val_auc:.5f}")
    print()
    print(f"Best checkpoint : {OUTPUT_ROOT / 'best.pt'}")
    print(f"Last checkpoint : {OUTPUT_ROOT / 'last.pt'}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
