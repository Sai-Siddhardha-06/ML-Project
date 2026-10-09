# Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection

Project Summary

This project implements a resource-efficient face forgery detection system based on DINOv2-Small and DFF-style LoRA adapters. It uses dual supervision to classify faces as real or fake and identify the forgery type. The model was trained on FaceForensics++ (FF++) for five epochs, achieving 87.87% validation ROC-AUC. Held-out test evaluation and cross-dataset testing remain future work.

## Setup

**Environment:** Python 3.10.14, PyTorch 2.2.2+cu121, Transformers 4.41.2, Scikit-learn 1.7.2, NVIDIA RTX 3060 (12 GB VRAM).

Create the environment and install dependencies:

```bash
conda create -n dff-adapter python=3.10 -y
conda activate dff-adapter

pip install torch==2.2.2 transformers==4.41.2 scikit-learn==1.7.2 pandas pillow numpy
```

Ensure the FF++ manifest (`splits/ffpp_manifest.csv`) and extracted frames (`frames/ffpp/`) are available.

## How to Run

Run from the project root:

```bash
# Run the smoke test
python cp15_end_to_end_smoke.py

# Run full training
python -u cp16_full_training.py
```

For long-running training, use `tmux` to keep the process running after SSH disconnection:

```bash
tmux new -s cp16
python -u cp16_full_training.py 2>&1 | tee cp16_training.log
```

Detach with `Ctrl+B`, then `D`. Reattach using `tmux attach -t cp16`.

## Team Contributions

- **Moguluri Sai Siddhardha (B24DS018) — Data, Backbone and Baseline Lead:** Dataset preparation, data splits, frame preprocessing, backbone setup, and evaluation infrastructure.
- **Sudhiksha Mattewada (B24CS027) — DFF-Adapter Architecture, Training and Analysis Lead:** LoRA expert bank, task routing, adapter integration, dual-task training, checkpointing, and result analysis.

Both members contribute to integration, experimental interpretation, documentation, and presentation.
