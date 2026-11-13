# Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection

A resource-efficient face forgery detection project based on DINOv2-Small and DFF-style LoRA adapters, developed as a reproduction-oriented implementation with a focus on dual-task supervision.

## Project Summary

This project implements a face forgery detection system using a pretrained **DINOv2-Small** vision transformer with parameter-efficient, LoRA-based adapter components inspired by the DFF-Adapter approach.

The model is designed to learn two related tasks:

- **Binary authenticity classification:** Determine whether a face is real or manipulated.
- **Forgery-type classification:** Predict the manipulation category as an auxiliary training task.

The architecture includes a shared pool of LoRA experts, multi-head task-specific routing, a shared adapter branch, task-fusion components, and a dual-loss training objective. The pretrained backbone is frozen while the adapter and classification components are trained.

The implementation was developed incrementally through CP8–CP16, covering the adapter expert bank, routing, shared components, transformer integration, dual-task training, smoke testing, and full-training pipeline.

### Current Results

| Item | Status |
|---|---|
| Backbone | DINOv2-Small |
| Training dataset | FaceForensics++ (FF++), c23 |
| Training duration | 5 epochs |
| Best reported validation ROC-AUC | **87.87%** |
| Held-out FF++ test evaluation | Pending |
| Cross-dataset evaluation | Pending |

The reported validation ROC-AUC is an experimental result from the current training run. It does not establish generalization to unseen datasets. Held-out test evaluation and cross-dataset evaluation remain future work.

## Architecture Overview

```text
Input face frames (224 × 224)
             |
             v
     Frozen DINOv2-Small
             |
             v
  Transformer blocks with
     DFF-style adapters
             |
     +-------+-------+
     |               |
     v               v
Task-specific      Shared LoRA
expert routing     expert branch
     |               |
     +-------+-------+
             |
             v
      Task-specific fusion
             |
       +-----+------+
       |            |
       v            v
  Authenticity   Forgery-type
  classification classification
       |            |
       v            v
    BCE loss      CE loss
       +-----+------+
             |
             v
       Dual-task loss
```

The training objective combines binary cross-entropy (BCE) for authenticity classification and cross-entropy (CE) for forgery-type classification:

\[
\mathcal{L}=\lambda_0\mathcal{L}_{\mathrm{BCE}}+
\lambda_1\mathcal{L}_{\mathrm{FTC}}
\]

The forgery-type branch provides auxiliary supervision during training. The intended inference task is binary real/fake classification.

## Project Structure

The repository contains implementation scripts, dataset manifests, model configuration files, evaluation artifacts, and experiment logs. Large datasets, extracted frames, virtual environments, model weights, and training checkpoints are excluded from the GitHub repository.

```text
B24_MLP/
├── .gitignore
├── .python-version
├── README.md
├── project_breakdown.txt
│
├── cp8_lora_expert_bank.py
├── cp9_multi_head_router.py
├── cp10_shared_head.py
├── cp11_shared_enhanced_task_fusion.py
├── cp12_dinov2_single_block.py
├── cp13_dinov2_all_blocks.py
├── cp14_dual_loss_training.py
├── cp15_end_to_end_smoke.py
├── cp15_validate_auc.py
├── cp16_full_training.py
│
├── train_dinov2_baseline.py
├── train_dinov2_linear_probe.py
├── train_dinov2_lora_baseline.py
├── evaluate_dinov2_linear_probe.py
├── evaluate_dinov2_lora_baseline.py
├── evaluate_cp7.py
├── test_dinov2.py
│
├── preprocess_videos.py
├── train_test_split_ffpp.py
├── create_ffpp_manifest.py
├── create_celebdf_manifest.py
├── verify_celebdf_test_set.py
├── analyze_ffpp_source_leakage.py
├── extract_dinov2_embeddings.py
├── download_dinov2_small.py
├── download_ff_plus.py
│
├── models/
│   └── dinov2-with-registers-small/
│       ├── config.json
│       └── preprocessor_config.json
│
├── splits/
│   ├── ffpp_manifest.csv
│   ├── celebdf_manifest.csv
│   ├── split_summary.txt
│   ├── train_pairs.txt
│   ├── val_pairs.txt
│   └── test_pairs.txt
│
├── evaluation/
│   ├── cp6_test_predictions.npz
│   └── cp6_test_results.json
│
└── experiment logs
    ├── cp16_training.log
    ├── cp16_resume.log
    ├── ffpp_preprocess.log
    └── other training and evaluation logs
```

The tree above summarizes the principal files; consult the repository for the complete file list. The exact contents of `splits/`, `evaluation/`, and the logs may evolve as experiments progress.

### File and Directory Descriptions

| Component | Purpose |
|---|---|
| `cp8_lora_expert_bank.py` | Implements the shared pool of low-rank LoRA experts. |
| `cp9_multi_head_router.py` | Implements task-specific multi-head routing and Top-3 expert selection. |
| `cp10_shared_head.py` | Implements the shared expert-routing branch. |
| `cp11_shared_enhanced_task_fusion.py` | Combines shared and task-specific adapter representations. |
| `cp12_dinov2_single_block.py` | Tests adapter integration in one transformer block. |
| `cp13_dinov2_all_blocks.py` | Integrates adapters across the DINOv2-Small transformer blocks. |
| `cp14_dual_loss_training.py` | Implements the dual-task training components and loss objective. |
| `cp15_end_to_end_smoke.py` | Runs a small end-to-end integration test. |
| `cp15_validate_auc.py` | Supports validation of ROC-AUC results. |
| `cp16_full_training.py` | Runs the configured FF++ training and validation pipeline. |
| `preprocess_videos.py` | Extracts and preprocesses face frames from videos. |
| `splits/` | Contains dataset manifests and train/validation/test split information. |
| `models/` | Stores model configuration files; pretrained weights must be obtained separately if not present locally. |
| `evaluation/` | Stores selected evaluation outputs from baseline experiments. |
| Experiment logs | Record preprocessing, training, resumption, and evaluation output. |

The earlier CP5–CP7 baseline experiments and their results are retained as experimental history. Their results should not be interpreted as DINOv2-Small DFF-Adapter results unless the corresponding experiment was run using that configuration.

## Dataset Preparation

### FaceForensics++ (FF++)

The primary training dataset is FaceForensics++ using the c23 compression setting.

The existing preprocessing workflow uses face detection and cropping, with frames resized to **224 × 224**. The dataset manifest records paths, split assignments, labels, and relevant video metadata.

Before training, prepare:

- The original FF++ videos and required manipulation categories.
- Extracted face frames for training, validation, and testing.
- `splits/ffpp_manifest.csv`, pointing to the correct local frame paths.
- A compatible pretrained `facebook/dinov2-small` backbone.

Expected local data layout:

```text
B24_MLP/
├── splits/
│   └── ffpp_manifest.csv
└── frames/
    └── ffpp/
        ├── train/
        ├── val/
        └── test/
```

The exact directory and metadata conventions must match the paths and columns expected by the preprocessing and training scripts.

**Important:** Dataset files and extracted frames are not included in this repository. The manifest alone does not contain the image data required for training.

### Celeb-DF-v2

Celeb-DF-v2 is intended for cross-dataset evaluation. Its manifest-generation and test-set verification utilities are included in the repository. Cross-dataset evaluation should be reported only after the corresponding evaluation has actually been completed.

## Environment and Dependencies

The development environment used for the reported experiments was:

| Component | Version / Specification |
|---|---|
| Python | 3.10.14 |
| PyTorch | 2.2.2+cu121 |
| Transformers | 4.41.2 |
| Scikit-learn | 1.7.2 |
| NumPy | 1.26.4 |
| Hardware | NVIDIA RTX 3060, 12 GB VRAM |

These are the recorded development versions, not a guarantee that every dependency combination will work on every system.

### Create the environment

Using Conda:

```bash
conda create -n dff-adapter python=3.10 -y
conda activate dff-adapter
```

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining core dependencies:

```bash
pip install transformers==4.41.2 scikit-learn==1.7.2 pandas pillow numpy
```

Depending on the script and environment, additional dependencies may be required. Install any missing packages identified by the relevant script rather than assuming this list is exhaustive.

Verify the environment:

```bash
python --version

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

The pretrained `facebook/dinov2-small` weights must be accessible through the Hugging Face cache or downloaded separately. The repository's local model configuration files do not, by themselves, provide the pretrained weights.

## How to Run

Run commands from the repository root, with the required environment activated and dataset paths configured.

### 1. Run the integration smoke test

```bash
python cp15_end_to_end_smoke.py
```

This checks the end-to-end integration path using the configuration and test setup implemented in the script. It is not a substitute for full training or held-out evaluation.

### 2. Run full training

```bash
python -u cp16_full_training.py
```

The script expects the FF++ data and manifest to be prepared in the locations expected by its configuration.

### 3. Keep long-running training alive with tmux

Start a named session:

```bash
tmux new -s cp16
```

Inside the session, run:

```bash
python -u cp16_full_training.py 2>&1 | tee cp16_training.log
```

Detach without stopping training:

1. Press `Ctrl+B`.
2. Press `D`.

Reattach later:

```bash
tmux attach -t cp16
```

Check whether the session is still running:

```bash
tmux ls
```

The training script's actual resume behavior depends on its checkpoint-loading implementation. A log file alone does not guarantee that training can resume from the last completed epoch.

## Experimental Status and Evaluation Plan

The project has progressed through the following implementation stages:

1. LoRA expert-bank implementation.
2. Multi-head task-specific routing.
3. Shared expert branch.
4. Shared-enhanced task fusion.
5. Single-block adapter integration.
6. Integration across transformer blocks.
7. Dual-task loss and training integration.
8. End-to-end smoke testing.
9. Full FF++ training and validation.

The current reported result is **87.87% validation ROC-AUC after five training epochs**.

The next evaluation steps are:

- Complete held-out FF++ test evaluation.
- Aggregate frame-level predictions into video-level predictions where appropriate.
- Report test ROC-AUC and other selected metrics, such as EER, consistently with the evaluation protocol.
- Evaluate generalization on Celeb-DF-v2.
- Compare the full adapter approach against the retained baseline experiments.
- Run ablations to measure the effect of shared routing, task-specific routing, dual supervision, and any proposed label-smoothing extension.

These are planned or pending steps unless supported by completed experiment outputs. Do not treat validation metrics as held-out test results.

## Team Contributions

- **Sudhiksha Mattewada (B24CS027) — DFF-Adapter Architecture, Training and Analysis Lead:** LoRA expert bank, task routing, adapter integration, dual-task training, checkpointing, and result analysis.
- **Moguluri Sai Siddhardha (B24DS018) — Data, Backbone and Baseline Lead:** Dataset preparation, data splits, frame preprocessing, backbone setup, and evaluation infrastructure.

Both members contribute to integration, experimental interpretation, documentation, and presentation.

## Limitations

- Current reported performance is based on validation ROC-AUC; held-out and cross-dataset performance remain to be established.
- The pretrained backbone weights and datasets must be obtained separately.
- Reproduction fidelity depends on matching the paper's architecture, hyperparameters, preprocessing, and evaluation protocol. Any implementation-specific interpretations or deviations should be documented.
- Results from different backbone variants or baseline experiments are not directly interchangeable.
