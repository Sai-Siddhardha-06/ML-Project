import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve


# CONFIG

EMBEDDING_PATH = Path(
    "embeddings/dinov2_base/test/embeddings.pt"
)

CHECKPOINT_PATH = Path(
    "checkpoints/dinov2_linear_probe/best.pt"
)


# MODEL

class LinearProbe(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.classifier(x).squeeze(1)


# EER

def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER).

    EER is the point where:
        False Positive Rate ~= False Negative Rate
    """

    fpr, tpr, thresholds = roc_curve(labels, scores)

    fnr = 1.0 - tpr

    difference = np.abs(fpr - fnr)

    idx = np.argmin(difference)

    eer = (fpr[idx] + fnr[idx]) / 2.0

    threshold = thresholds[idx]

    return eer, threshold


# VIDEO AGGREGATION

def aggregate_video_scores(video_ids, labels, scores):
    """
    Aggregate frame predictions into one prediction per video.

    Uses mean frame probability.
    """

    video_scores = {}
    video_labels = {}

    for video_id, label, score in zip(
        video_ids,
        labels,
        scores,
    ):
        if video_id not in video_scores:
            video_scores[video_id] = []

        video_scores[video_id].append(float(score))

        if video_id not in video_labels:
            video_labels[video_id] = int(label)

    final_video_ids = []
    final_labels = []
    final_scores = []

    for video_id in video_scores:

        final_video_ids.append(video_id)

        final_labels.append(
            video_labels[video_id]
        )

        final_scores.append(
            np.mean(video_scores[video_id])
        )

    return (
        final_video_ids,
        np.array(final_labels),
        np.array(final_scores),
    )


# MAIN

def main():

    print("=" * 60)
    print("DINOv2-Base + Linear Probe Evaluation")
    print("=" * 60)

    # Device

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print()

    # Load test embeddings

    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"Missing embeddings: {EMBEDDING_PATH}"
        )

    data = torch.load(
        EMBEDDING_PATH,
        map_location="cpu",
    )

    embeddings = data["embeddings"].float()
    labels = data["labels"].numpy()
    video_ids = data["video_ids"]

    print("Test data:")
    print(f"  embeddings : {tuple(embeddings.shape)}")
    print(f"  labels     : {len(labels)}")
    print(f"  video IDs  : {len(video_ids)}")
    print()

    # Integrity checks

    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 768

    assert len(embeddings) == len(labels)
    assert len(embeddings) == len(video_ids)

    print("Embedding integrity: PASS")
    print()

    # Load checkpoint

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = LinearProbe(input_dim=768)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Checkpoint:")
    print(f"  epoch   : {checkpoint['epoch']}")
    print(f"  val AUC : {checkpoint['val_auc']:.5f}")
    print()

    # Frame-level inference

    print("Running frame-level inference...")

    batch_size = 1024

    all_scores = []

    with torch.inference_mode():

        for start in range(
            0,
            len(embeddings),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(embeddings),
            )

            batch = embeddings[start:end].to(device)

            logits = model(batch)

            scores = torch.sigmoid(logits)

            all_scores.append(
                scores.cpu()
            )

    frame_scores = torch.cat(
        all_scores
    ).numpy()

    # Frame-level metrics

    frame_auc = roc_auc_score(
        labels,
        frame_scores,
    )

    print()
    print("FRAME-LEVEL RESULTS")
    print("-" * 60)
    print(f"Frames evaluated : {len(frame_scores)}")
    print(f"Frame-level AUC  : {frame_auc:.5f}")

    # Video-level aggregation

    (
        final_video_ids,
        video_labels,
        video_scores,
    ) = aggregate_video_scores(
        video_ids,
        labels,
        frame_scores,
    )

    # Video-level metrics

    video_auc = roc_auc_score(
        video_labels,
        video_scores,
    )

    eer, eer_threshold = compute_eer(
        video_labels,
        video_scores,
    )

    # Video counts

    real_videos = int(
        np.sum(video_labels == 0)
    )

    fake_videos = int(
        np.sum(video_labels == 1)
    )

    # Check frames/video distribution
    unique_ids, frame_counts = np.unique(
        np.array(video_ids),
        return_counts=True,
    )

    # Final results

    print()
    print("=" * 60)
    print("VIDEO-LEVEL RESULTS")
    print("=" * 60)

    print(f"Videos evaluated : {len(final_video_ids)}")
    print(f"Real videos      : {real_videos}")
    print(f"Fake videos      : {fake_videos}")
    print(f"Frames/video     : {frame_counts.min()}–{frame_counts.max()}")
    print()

    print(f"Video-level AUC  : {video_auc:.5f}")
    print(f"Video-level EER  : {eer:.5f}")
    print(f"EER threshold    : {eer_threshold:.5f}")

    print()
    print("=" * 60)
    print("CHECKPOINT 5 BASELINE")
    print("=" * 60)

    print("Model: DINOv2-Base + Linear Probe")
    print(f"Best training epoch : {checkpoint['epoch']}")
    print(f"Validation AUC      : {checkpoint['val_auc']:.5f}")
    print(f"Test frame AUC      : {frame_auc:.5f}")
    print(f"Test video AUC      : {video_auc:.5f}")
    print(f"Test video EER      : {eer:.5f}")

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
