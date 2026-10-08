import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# Configuration

ROOT = Path("frames/ffpp")
OUTPUT_ROOT = Path("embeddings/dinov2_base")

MODEL_NAME = "facebook/dinov2-base"

IMAGE_SIZE = 224

# Conservative for RTX 3060 12 GB.
# Increase only if VRAM usage is comfortably below the limit.
BATCH_SIZE = 16

NUM_WORKERS = 0

# Use mixed precision during DINOv2 inference.
USE_AMP = True

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Dataset

class FFPPFrameDataset(Dataset):
    """
    Loads the already-preprocessed FF++ face frames.

    Each sample contains:
        - image
        - label
        - video_id

    No augmentation is used here because this stage only
    extracts deterministic DINOv2 embeddings.
    """

    def __init__(self, root, split, transform):

        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.samples = []

        split_dir = self.root / split

        video_dirs = sorted(
            d for d in split_dir.iterdir()
            if d.is_dir()
        )

        for video_dir in video_dirs:

            meta_path = video_dir / "meta.json"

            if not meta_path.exists():
                continue

            with open(meta_path) as f:
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
                    {
                        "frame_path": frame_path,
                        "label": label,
                        "video_id": video_id,
                    }
                )

        print(
            f"{split}: "
            f"{len(self.samples)} frames from "
            f"{len(set(x['video_id'] for x in self.samples))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(
            sample["frame_path"]
        ).convert("RGB")

        image = self.transform(image)

        return (
            image,
            sample["label"],
            sample["video_id"],
        )


# Model / preprocessing

print("=" * 60)
print("DINOv2 Embedding Extraction")
print("=" * 60)

print(f"Model : {MODEL_NAME}")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU   : "
        f"{torch.cuda.get_device_name(0)}"
    )

    total_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(
        f"VRAM  : "
        f"{total_memory:.2f} GB"
    )


processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()

# DINOv2 normalization.
image_mean = processor.image_mean
image_std = processor.image_std

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_mean,
            std=image_std,
        ),
    ]
)


# Embedding extraction

@torch.inference_mode()
def extract_split(split):

    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / "embeddings.pt"

    # Resume support
    #
    # If the split has already been completely extracted,
    # don't recompute it.
    #
    if output_file.exists():

        print()
        print(
            f"[SKIP] Existing embeddings found:"
            f" {output_file}"
        )

        return

    dataset = FFPPFrameDataset(
        ROOT,
        split,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
    )

    all_embeddings = []
    all_labels = []
    all_video_ids = []

    print()
    print(
        f"Extracting {split} embeddings..."
    )

    progress = tqdm(
        loader,
        desc=f"{split}",
    )

    for images, labels, video_ids in progress:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # Mixed precision

        if USE_AMP and DEVICE.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                outputs = model(
                    pixel_values=images
                )

        else:

            outputs = model(
                pixel_values=images
            )

        # CLS token

        embeddings = (
            outputs.last_hidden_state[:, 0]
        )

        # Store embeddings as FP16.
        #
        # This reduces disk usage substantially while keeping
        # the cached representation compact.

        embeddings = (
            embeddings
            .detach()
            .cpu()
            .to(torch.float16)
        )

        all_embeddings.append(
            embeddings
        )

        all_labels.append(
            labels.cpu()
        )

        all_video_ids.extend(
            list(video_ids)
        )

        progress.set_postfix(
            gpu=(
                f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
                if DEVICE.type == "cuda"
                else "CPU"
            )
        )

    # Combine

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    # Sanity checks

    assert embeddings.shape[0] == len(all_video_ids)

    assert embeddings.shape[0] == labels.shape[0]

    assert embeddings.shape[1] == 768

    print()
    print(f"{split} extraction complete:")

    print(f"  embeddings : {tuple(embeddings.shape)}")

    print(f"  dtype      : {embeddings.dtype}")

    print(f"  labels     : {tuple(labels.shape)}")

    print(f"  video IDs  : {len(all_video_ids)}")

    # Save

    data = {
        "embeddings": embeddings,
        "labels": labels,
        "video_ids": all_video_ids,
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "dtype": "float16",
    }

    torch.save(
        data,
        output_file,
    )

    print(
        f"  saved      : {output_file}"
    )


# Main

def main():

    if DEVICE.type != "cuda":

        print()
        print(
            "WARNING: CUDA is not available."
        )

        print(
            "DINOv2 embedding extraction will run on CPU."
        )

    # Process one split at a time so we don't keep all
    # embeddings in GPU memory.
    for split in ["train", "val", "test"]:

        print()
        print("=" * 60)
        print(f"PROCESSING: {split}")
        print("=" * 60)

        extract_split(split)

        # Release unused CUDA cache between splits.
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("ALL EMBEDDINGS EXTRACTED")
    print("=" * 60)

    for split in ["train", "val", "test"]:

        path = (
            OUTPUT_ROOT
            / split
            / "embeddings.pt"
        )

        if path.exists():

            size_mb = (
                path.stat().st_size
                / (1024 ** 2)
            )

            print(
                f"{split}: "
                f"{path} "
                f"({size_mb:.1f} MB)"
            )


if __name__ == "__main__":
    main()
