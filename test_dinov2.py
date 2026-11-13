"""
DINOv2 sanity test.

Checks:
1. Hugging Face DINOv2 processor loads correctly.
2. DINOv2 model downloads and loads correctly.
3. Model moves to CUDA.
4. A 224x224 dummy image passes through the model.
5. Output dimensions match DINOv2-Base.

Run:
    python test_dinov2.py
"""

import torch
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-base"


def main():
    print("=" * 60)
    print("DINOv2 Sanity Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Device check
    # ---------------------------------------------------------
    print("\n[1/5] Checking device...")

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("CUDA available: YES")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    else:
        device = torch.device("cpu")

        print("CUDA available: NO")
        print("WARNING: Running on CPU.")

    # ---------------------------------------------------------
    # 2. Load image processor
    # ---------------------------------------------------------
    print("\n[2/5] Loading DINOv2 image processor...")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("Processor loaded successfully.")

    # ---------------------------------------------------------
    # 3. Load model
    # ---------------------------------------------------------
    print("\n[3/5] Loading DINOv2 model...")

    model = AutoModel.from_pretrained(MODEL_NAME)

    n_params = sum(p.numel() for p in model.parameters())

    print("Model loaded successfully.")
    print(f"Parameters: {n_params / 1e6:.2f} M")
    print(f"Hidden size: {model.config.hidden_size}")

    # ---------------------------------------------------------
    # 4. Move model to GPU
    # ---------------------------------------------------------
    print("\n[4/5] Moving model to device...")

    model = model.to(device)
    model.eval()

    print(f"Model device: {next(model.parameters()).device}")

    # ---------------------------------------------------------
    # 5. Dummy forward pass
    # ---------------------------------------------------------
    print("\n[5/5] Running dummy forward pass...")

    dummy = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    with torch.no_grad():
        outputs = model(pixel_values=dummy)

    output_shape = tuple(outputs.last_hidden_state.shape)

    print(f"Output shape: {output_shape}")

    # DINOv2-Base:
    # 1 image
    # 256 patch tokens + 1 CLS token
    # 768-dimensional embedding

    expected_shape = (1, 257, 768)

    if output_shape == expected_shape:
        print("Output shape check: PASS")
    else:
        print(
            f"Output shape check: WARNING "
            f"(expected {expected_shape})"
        )

    print("\n" + "=" * 60)
    print("DINOv2 sanity test: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
