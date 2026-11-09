# Final Results: DFF-Adapter Reproduction vs. Original Paper

The main question is whether our reproduction achieves strong FF++ validation performance **and** generalizes to unseen deepfake datasets. We should evaluate these separately.

The original paper uses a frozen DINOv2-Large backbone and trains for 50 epochs. Our experiments use DINOv2-Small or Base, with 12- or 30-epoch training runs. The paper reports video-level AUC for its benchmark comparison, so AUC is the primary metric for our comparison.

We have now added DFDC to our cross-dataset evaluation. The DFDC evaluation covers 392 videos and 12,544 face crops, with 76 real videos and 316 fake videos.

## 1. Final comparison table — video-level AUC

| Model | Training epochs | FF++ validation AUC ↑ | Celeb-DF-v1 AUC ↑ | Celeb-DF-v2 AUC ↑ | DFDC AUC ↑ |
|---|---:|---:|---:|---:|---:|
| DFF-Adapter Small | 12 | 99.08% | 81.20% | 85.02% | Not evaluated |
| **DFF-Adapter Small** | **30** | **99.08%** | **82.98%** | **86.25%** | **88.03%** |
| DFF-Adapter Small + label smoothing (0.1) | 30 | 99.05% | 81.20% | 84.74% | 87.45% |
| DFF-Adapter Base | 12 | 99.08% | 82.43% | 81.51% | Not evaluated |
| DFF-Adapter Base | 30 | **99.16%** | 81.71% | 84.22% | **84.00%** |
| **Original paper — DINOv2-Large** | **50** | **99.56%** | **96.14%** | **95.26%** | Not reported here |

**How to read the table:**
- Higher AUC is better.
- The paper's values are reported in its benchmark comparison table.
- Our FF++ figures are validation results from our experiments; our Celeb-DF scores come from the saved evaluation runs, and the DFDC scores come from the newly completed evaluation.
- The paper's results are reference benchmarks, not results obtained by running its implementation in our environment.
- The Small and Base DFDC scores were obtained from their respective `best.pt` checkpoints. The corresponding training duration and checkpoint-selection details should be verified before interpreting this as a strictly controlled backbone comparison.
- The 12-epoch models were not evaluated on DFDC in the runs reported here.

## 2. Cross-dataset EER comparison

Equal Error Rate (EER) is another important metric: **lower is better**. It measures the error rate at the threshold where the false-positive and false-negative rates are equal.

| Model | Epochs | Celeb-DF-v1 EER ↓ | Celeb-DF-v2 EER ↓ | DFDC EER ↓ |
|---|---:|---:|---:|---:|
| DFF-Adapter Small | 12 | 30.65% | 22.94% | Not evaluated |
| **DFF-Adapter Small** | **30** | **26.32%** | **22.35%** | **19.74%** |
| DFF-Adapter Small + label smoothing (0.1) | 30 | 26.32% | 25.00% | 19.94% |
| DFF-Adapter Base | 12 | 25.81% | 26.18% | Not evaluated |
| DFF-Adapter Base | 30 | 28.95% | 23.82% | **23.68%** |
| Original paper — DINOv2-Large | 50 | Not reported in the cited comparison table | **12.65%** | Not reported here |

The original paper reports an EER of 1.79% on FF++ validation/testing in its benchmark comparison, alongside an AUC of 99.56%. Our FF++ EER values are approximately 3.2–3.6%, depending on the run. These figures should be compared cautiously because our validation procedure and the paper's benchmark procedure may not be identical.

The DFDC EER values were reported directly by our evaluator. The original paper's DFDC EER is not included here because we do not have a corresponding result in the comparison data.

## 3. DFDC evaluation results

The DFDC evaluation used the following manifest:

- **Videos:** 392
- **Real videos:** 76
- **Fake videos:** 316
- **Face crops:** 12,544
- **Crops per video:** 32

Both models were evaluated on the same manifest using all available crops. The reported results were:

| Metric | DINOv2-Small | DINOv2-Base | Better result |
|---|---:|---:|---|
| Video-level AUC ↑ | **88.03%** | 84.00% | Small |
| EER ↓ | **19.74%** | 23.68% | Small |
| Frame-level AUC ↑ | **79.99%** | 78.10% | Small |
| Videos evaluated | 392 | 392 | Same |
| Frames evaluated | 12,544 | 12,544 | Same |

Small achieved a video-level AUC that was **4.03 percentage points higher** than Base. Its EER was **3.94 percentage points lower**.

These results indicate that Small discriminated between real and fake videos more effectively than Base on this DFDC evaluation subset, according to all three reported metrics.

However, DFDC contains a relatively small and imbalanced evaluation subset. The results should therefore be interpreted as evidence about the evaluated subset, not as definitive evidence of performance on the full DFDC dataset.

## 4. Label-smoothing experiment — completed 30-epoch run

A controlled label-smoothing run was completed using **DINOv2-Small** and the DFF-Adapter variant. Label smoothing was set to **0.1 for the auxiliary forgery-type classification loss only**; the authenticity BCE loss was left unchanged. The run used seed 42 and completed 45,000 training steps.

### Training configuration

| Setting | Value |
|---|---|
| Backbone / variant | DINOv2-Small / DFF-Adapter |
| Epochs / steps | 30 / 45,000 |
| Frames per training video | 10 |
| Validation frames per video | 8 |
| Batch size | 24 |
| Learning rate | 0.0001 |
| Weight decay | 0.01 |
| Authenticity loss weight | 10.0 |
| Forgery-type loss weight | 2.0 |
| Auxiliary forgery-type label smoothing | 0.1 |
| Random seed | 42 |
| Best FF++ validation video AUC | 99.05% |
| Selected checkpoint | `runs/dff_ls01_30ep/best.pt` |

The run's saved metadata reports a best validation AUC of `0.990485` and 45,000 steps. The run arguments and checkpoints are included in the backup ZIP.

### Cross-dataset results

All available crops were evaluated using the same label-smoothed `best.pt` checkpoint.

| Dataset | Videos | Frames | Video AUC ↑ | EER ↓ | Frame AUC ↑ |
|---|---:|---:|---:|---:|---:|
| Celeb-DF-v1 | 100 | 3,200 | 81.20% | 26.32% | 77.42% |
| Celeb-DF-v2 | 518 | 16,576 | 84.74% | 25.00% | 77.52% |
| DFDC subset | 392 | 12,544 | 87.45% | 19.94% | 80.90% |

### Comparison with the existing DINOv2-Small 30-epoch baseline

| Dataset | Metric | Baseline | Label smoothing 0.1 | Change |
|---|---|---:|---:|---:|
| Celeb-DF-v1 | Video AUC ↑ | 82.98% | 81.20% | −1.78 pp |
| Celeb-DF-v1 | EER ↓ | 26.32% | 26.32% | 0.00 pp |
| Celeb-DF-v2 | Video AUC ↑ | 86.25% | 84.74% | −1.51 pp |
| Celeb-DF-v2 | EER ↓ | 22.35% | 25.00% | +2.65 pp |
| DFDC subset | Video AUC ↑ | 88.03% | 87.45% | −0.58 pp |
| DFDC subset | EER ↓ | 19.74% | 19.94% | +0.20 pp |
| DFDC subset | Frame AUC ↑ | 79.99% | 80.90% | +0.91 pp |

**Interpretation:** In this run, applying 0.1 label smoothing to the auxiliary forgery-type loss did **not** improve video-level AUC on any of the three evaluated cross-dataset subsets. Celeb-DF-v1 EER was unchanged; Celeb-DF-v2 EER increased, and DFDC EER increased slightly. DFDC frame-level AUC improved by 0.91 percentage points, so the frame-level and video-level results are not uniformly aligned.

This is evidence about this specific run and evaluation setup, not proof that label smoothing is generally ineffective. Before making a strong causal claim, verify that the baseline and label-smoothed checkpoints use identical evaluation manifests, preprocessing, aggregation, and checkpoint-selection rules. The baseline Celeb-DF results in the existing report were described as final-checkpoint results, while this label-smoothed evaluation used `best.pt`; that checkpoint-selection difference should be disclosed.

### Backup and reproducibility status

A ZIP backup was created on 2026-10-30: `DFF_Adapter_LabelSmoothing_Backup_20261030_112836.zip`. It contains the following relevant artifacts:

- `runs/dff_ls01_30ep/best.pt`
- `runs/dff_ls01_30ep/last.pt`
- `runs/dff_ls01_30ep/args.json`
- `runs/dff_ls01_30ep/done.json`
- `runs/dff_ls01_30ep/log.jsonl`
- `results/dff_ls01_cross_dataset.json`
- `train.py`, `evaluate_dff.py`, and `run_experiments.py`
- `backup_metadata.json`

The ZIP does **not** include the original datasets or extracted face crops. Download or copy the ZIP to persistent storage; files under Kaggle's `/kaggle/working` may be lost when the session environment is reset.

## 5. Inferences we can draw

### A. More training does not guarantee better generalization

For Small, moving from 12 to 30 epochs produced:

| Dataset | AUC change | EER change | Interpretation |
|---|---:|---:|---|
| Celeb-DF-v1 | +1.78 percentage points | −4.33 percentage points | Improved |
| Celeb-DF-v2 | +1.23 percentage points | −0.59 percentage points | Improved |

For Base, moving from 12 to 30 epochs produced:

| Dataset | AUC change | EER change | Interpretation |
|---|---:|---:|---|
| Celeb-DF-v1 | −0.72 percentage points | +3.14 percentage points | Worse |
| Celeb-DF-v2 | +2.71 percentage points | −2.36 percentage points | Improved |

We have not evaluated the 12-epoch checkpoints on DFDC, so the effect of training duration on DFDC generalization cannot yet be assessed.

**Inference:** Longer training helped the Small model on both Celeb-DF evaluations, but its effect on Base depended on the target dataset. Training duration alone is therefore not a reliable predictor of generalization performance.

This is consistent with possible overfitting to the source dataset, although we cannot establish overfitting as the cause from these results alone.

### B. Small currently looks more promising for cross-dataset generalization

Our Small 30-epoch model has the highest AUC among our four configurations on both Celeb-DF datasets:

- Celeb-DF-v1: **82.98%**
- Celeb-DF-v2: **86.25%**

It also has the lowest EER among our configurations on Celeb-DF-v2.

On DFDC, Small achieved:

- Video-level AUC: **88.03%**
- EER: **19.74%**
- Frame-level AUC: **79.99%**

Base achieved lower AUC and higher EER on the same DFDC subset.

**Inference:** Small is the strongest current candidate across the three cross-dataset evaluations represented in our results. However, this does not establish that Small is intrinsically superior to Base. The checkpoints may differ in training configuration, and we need to confirm their comparability before making a stronger claim.

Base achieves the highest FF++ validation AUC at 30 epochs, but it does not achieve the best cross-dataset AUC.

### C. Strong source-dataset performance does not establish robustness

Our models achieve approximately 99% FF++ validation AUC, yet their Celeb-DF AUCs are substantially lower.

For Small at 30 epochs, the difference is:

- FF++ validation to Celeb-DF-v1: approximately **16.10 percentage points**.
- FF++ validation to Celeb-DF-v2: approximately **12.84 percentage points**.

The DFDC AUC of 88.03% is also below the reported FF++ validation AUC of 99.08%, a difference of approximately **11.05 percentage points**.

This illustrates the distinction between **in-domain performance** and **cross-dataset generalization**.

A model can distinguish real and fake videos very effectively on data resembling its training distribution while struggling when the source, manipulation process, compression, identity distribution or other characteristics change.

We cannot identify which factor causes the performance drop without further experiments. Furthermore, the FF++ validation result and the DFDC result may not correspond to the same checkpoint, so the numerical difference should be interpreted with that limitation in mind.

### D. There is a substantial gap between our reproduction and the paper

| Dataset | Small, 30 epochs | Paper | Absolute AUC gap |
|---|---:|---:|---:|
| FF++ validation | 99.08% | 99.56% | 0.48 percentage points |
| Celeb-DF-v1 | 82.98% | 96.14% | 13.16 percentage points |
| Celeb-DF-v2 | 86.25% | 95.26% | 9.01 percentage points |
| DFDC | 88.03% | Not reported here | Not available |

The FF++ gap is comparatively small, while the cross-dataset gaps on Celeb-DF are much larger.

**Inference:** Reproducing high source-dataset performance has not yet reproduced the paper's reported cross-dataset generalization on Celeb-DF.

We cannot quantify the corresponding gap on DFDC without a comparable reference result from the paper.

This does not by itself prove that our architecture is incorrect or that the paper's results are irreproducible. We must first investigate differences in the model configuration, training recipe, data splits, preprocessing and evaluation protocol.

### E. Video-level predictions outperform frame-level predictions in the DFDC evaluation

On DFDC, the Small model's video-level AUC was 88.03%, compared with a frame-level AUC of 79.99%. Base showed a similar pattern: 84.00% video-level AUC versus 78.10% frame-level AUC.

The differences were:

- Small: **8.04 percentage points**
- Base: **5.90 percentage points**

**Inference:** Aggregating predictions across frames is associated with stronger discrimination at the video level in these experiments. This is consistent with video-level aggregation reducing the influence of noisy individual-frame predictions.

However, the results alone do not establish why aggregation helps, and the exact aggregation procedure should be considered when comparing these metrics with results from other implementations.

### F. The Base model does not consistently benefit from its larger backbone

Base achieved a slightly higher FF++ validation AUC than Small in the reported 30-epoch results, but it performed worse on all three reported DFDC metrics.

The DFDC comparison was:

- Video-level AUC: Small 88.03%, Base 84.00%.
- EER: Small 19.74%, Base 23.68%.
- Frame-level AUC: Small 79.99%, Base 78.10%.

**Inference:** A larger backbone does not automatically produce better cross-dataset performance. Possible explanations include differences in learned representations, training dynamics, optimization, or interactions between the model and the target dataset. These remain hypotheses until investigated through controlled experiments.

## 6. Limitations of our current experiments

These are important to acknowledge in a report or research presentation.

### 1. Backbone mismatch

The paper uses **DINOv2-Large**, whereas our experiments use Small and Base.

This changes the backbone's capacity and representation. Consequently, the paper-versus-reproduction comparison does not isolate the effect of our implementation changes.

### 2. Training configuration mismatch

The paper reports 50 epochs, a learning rate of \(2\times10^{-4}\), weight decay of \(10^{-5}\), and a single RTX 4090 GPU. Our runs use different epoch counts and a different training configuration.

Training for 30 epochs instead of 50 does not necessarily explain the performance gap: the learning rate, optimization trajectory, architecture and other details may also matter.

### 3. Evaluation sample size and protocol

Our saved evaluation files report:

| Dataset | Videos evaluated | Frames evaluated |
|---|---:|---:|
| Celeb-DF-v1 | 100 | 3,200 |
| Celeb-DF-v2 | 518 | 16,576 |
| DFDC | 392 | 12,544 |

These are the sample sizes in our current evaluation manifests. We should verify that they match the exact benchmark splits used by the paper before treating the scores as directly comparable.

In particular, we should not assume that our evaluation subsets represent every video in the full datasets.

The DFDC subset is also imbalanced: 76 real videos and 316 fake videos. AUC and EER are useful here, but confidence intervals or repeated evaluations would help quantify uncertainty.

### 4. Checkpoint consistency

Our Small 30-epoch FF++ result is the **best validation AUC**, whereas the saved Celeb-DF results are from the **final checkpoint**.

The newly completed DFDC evaluations use the `best.pt` checkpoints from the Small and Base backup runs. We must verify the exact training durations and checkpoint-selection procedures associated with those files before assuming they correspond to the 30-epoch models in the original table.

Consequently, the AUC figures in the table do not necessarily describe the same model state.

A cleaner experiment would evaluate the same selected checkpoint on FF++, Celeb-DF-v1, Celeb-DF-v2 and DFDC.

### 5. Limited evidence about variability

We have not established the variability of these results across multiple random seeds. A single run cannot tell us how much of the difference between Small and Base comes from random initialization, data ordering or other stochastic factors.

Running multiple seeds and reporting mean and standard deviation would make the conclusions more reliable.

### 6. Limited causal evidence

The observed performance differences do not tell us *why* generalization changes. Potential explanations include differences in learned features, domain shift, training dynamics or preprocessing, but these remain hypotheses.

Controlled ablations would be required to distinguish among them.

### 7. Limited DFDC coverage

Our DFDC evaluation includes 392 videos, not necessarily the entire dataset. The results therefore describe the evaluated subset and should not be presented as definitive performance on full DFDC.

We also do not have a corresponding DFDC reference score from the original paper in the results summarized here.

## 7. What we should do next

I would prioritize the following experiments:

1. **Checkpoint-consistent evaluation:** Evaluate each model's selected best-validation checkpoint on all four datasets using the same scoring procedure.
2. **Protocol verification:** Confirm that the Celeb-DF and DFDC splits, face cropping, frame aggregation and video-level metrics match the paper's protocol as closely as possible.
3. **Multiple-seed experiments:** Run at least three seeds for the most promising configuration if compute permits.
4. **Targeted ablations:** Investigate the adapter configuration, auxiliary forgery-type loss and label smoothing separately, changing one factor at a time.
5. **DFDC subset verification:** Document how the 392 videos were selected and whether the subset is representative of the intended evaluation population.
6. **Document limitations explicitly:** Record the backbone mismatch, available evaluation sample sizes, checkpoint differences and remaining protocol differences when presenting the reproduction results.

## 8. Final conclusion

Our experiments demonstrate that the DFF-Adapter implementation achieves strong FF++ validation AUC, but cross-dataset generalization remains weaker than the results reported in the original paper on Celeb-DF.

Among our current configurations, **Small is the strongest candidate across the three evaluated cross-dataset subsets**, with a video-level AUC of 82.98% on Celeb-DF-v1, 86.25% on Celeb-DF-v2 and 88.03% on the evaluated DFDC subset. On DFDC, it also achieves a lower EER and higher frame-level AUC than Base.

Longer training improved Small's results on both Celeb-DF datasets, whereas Base showed mixed effects. We cannot yet draw a corresponding conclusion about training duration on DFDC because the 12-epoch models have not been evaluated there.

The most defensible research conclusion is that **source-dataset performance alone is insufficient to assess deepfake detector generalization**. Our results show a substantial gap between strong FF++ validation performance and performance on unseen datasets, while the DFDC comparison also demonstrates that a larger backbone does not necessarily perform better on a cross-dataset evaluation.

The completed label-smoothing experiment (0.1 on the auxiliary forgery-type loss) did not improve video-level AUC on Celeb-DF-v1, Celeb-DF-v2 or the evaluated DFDC subset relative to the existing Small 30-epoch baseline. DFDC frame-level AUC improved, but video-level AUC and EER did not. This result should be treated as a single-run ablation, with checkpoint-selection differences noted.

Before attributing the paper-to-reproduction gap or the label-smoothing outcome to a particular architectural or training choice, we need checkpoint-consistent evaluation, protocol verification and controlled ablations.