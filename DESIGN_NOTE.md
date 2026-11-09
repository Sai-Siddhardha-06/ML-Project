# DFF-Adapter reproduction: design note (Stage 0) and deviation log

Paper: arXiv:2511.12107 (Zhang et al.). Backbone: frozen DINOv2. Status key:
**P** = stated in the paper, **A** = our assumption (paper ambiguous), **D** = deliberate deviation.

## 1. One-paragraph summary (say this first in the viva)
A DINOv2 block has query/key/value linears, an attention output linear, and an MLP. We freeze all of
DINOv2 and wrap three linears per block (query, value, attention-output dense) with a small trainable
adapter. The adapter splits the layer's input channels into 4 chunks. Chunk 0 goes to a *shared* head
(softmax mix of all 6 experts); chunks 1-3 go to *task* heads (each mixes its top-3 of 6 experts).
An expert is a LoRA pair (down-project to rank 4, up-project back). The same network is run twice per
batch: task 0 (real/fake, BCE loss) and task 1 (forgery type, cross-entropy). The two tasks share the
experts and the shared head, but use different routing logits for the task heads, so what the
forgery-type task learns reaches the authenticity task through the shared weights.
Only the adapters and two linear heads train. At test time only task 0 is used.

## 2. Shapes (DINOv2-Small: d=384, 12 blocks, 6 attention heads; 224x224 -> 256 patches + CLS)
| tensor | shape |
|---|---|
| token stream in every block | (B, 257, 384) |
| query / value / output-dense input and output | (B, 257, 384) |
| adapter chunk | (B, 257, 4, 96) |
| expert A (down) / B (up) per head | (6, 4, 96) / (6, 96, 4) |
| routing logits Z_task / Z_shared per adapter | (2, 3, 6) / (6,) |
| CLS token to heads | (B, 384) |

## 3. Equation -> code map (`dff_adapter.py`)
| Paper | Meaning | Code |
|---|---|---|
| Eq. 4 | g = softmax(Z_task[t,k]) | `gate_weights`: `F.softmax(self.z_task[t])` |
| Eq. 3 | S = Top3(N, g) | `p.topk(self.cfg.top_k)` |
| Eq. 2 | renormalise over the selected 3 | `top_v / top_v.sum(...)` then `scatter` |
| Eq. 1 | f = beta * sum_j g~_j B_j(A_j x) | `forward`: two `einsum`s, times `self.scale` |
| Eq. 5 | shared head: softmax over all N experts | row 0 of `gate_weights` |
| Eq. 6, 8 | concat(shared, task heads), added residually | head outputs concatenated to width d, added to the frozen linear's output in `DFFLinear.forward` |
| Eq. 7 | L_bce on authenticity head | `train.py` (to be written) |
| Eq. 9 | L_ftc on forgery-type head | `train.py` |
| Eq. 10 | L = l0 L_bce + l1 L_ftc | `train.py` |
| "inserted into every query, value, dense" | injection | `inject_dff` |
| "task flag 0/1, two forward passes" | task switching | `TaskContext.task`, `DFFDetector.forward(task=...)` |

`test_matches_naive_equations` checks the batched code against a literal loop implementation of Eq. 1-5.

## 4. Decisions
| # | Decision | Status | Evidence / reasoning |
|---|---|---|---|
| D1 | Router logits are learned tensors, not computed from the input | P | Eq. 4-5 define Z as parameters. The text says the router "scores adapters that fit the statistics of the sub-space", which suggests input-dependence, but the equations do not. We follow the equations. |
| D2 | H=4 = 1 shared head (chunk 0) + 3 task heads (chunks 1-3) | A | Paper: "4 head adapters"; Eq. 6: concat(f_share, f^1..f^n); X^(0) is the shared head's slice. Concat width = d only if 1 + n = 4. |
| D3 | Each head has its own bank of N=6 experts | A | Table 9: "experts per head". Gives 3.54M adapter params for DINOv2-Large vs paper's 3.5M. One bank shared by all heads would give 0.89M. Supporting evidence, not proof. |
| D4 | Rank per head = r/H = 4; scale beta = alpha/r = 2 | P/A | Paper gives r_h = r/h. Scale = alpha/r is the standard LoRA convention; the paper just says "scaling coefficient". |
| D5 | Fusion is the per-layer residual addition; one CLS token, two forward passes | A | Paper's training section describes two passes with a task flag. Fig. 2 draws two CLS tokens; we read that as one token position evaluated under two task flags. |
| D6 | "dense" = attention output projection | A | Paper says "query, value, and dense projection of each block"; Fig. 2 shows q, v and MLP. We did not adapt MLP layers. |
| D7 | Routing logits initialised N(0, 0.5^2); B zero-init, A uniform(+-1/sqrt(d_h)) | A | Paper silent. Zero B makes the adapter an exact no-op at init (tested). |
| D8 | 5 forgery-type classes: real + Deepfakes, Face2Face, FaceSwap, NeuralTextures | A | From Fig. 1 (Real, FaceSwap, Face2Face, ...). |

## 5. Known behaviour worth knowing before the viva
- **Dead experts.** Top-k routing gives zero gradient to experts outside the selected 3 (the softmax
  Jacobian cancels after renormalising). An expert unselected by both tasks in a given head/layer never
  trains. `routing_report` counts these. This also affects the paper's own formulation (it is a property
  of top-k with static logits), so we report it as an observation, not fix it silently.
- **Why query/value/dense, not key?** The paper does not justify it. LoRA literature commonly adapts
  q and v; do not claim the paper proves this.

## 6. Deviation log (fill in as we go)
| Item | Paper | Ours | Reason | Expected effect |
|---|---|---|---|---|
| Backbone | DINOv2-Large with registers | Small or Base (no registers) | Free-tier GPU budget | Lower absolute AUC; compare with the paper's Table 6 Base rows |
| Face preprocessing | DeepfakeBench pipeline | MTCNN, 1.3x square crop, 224, 32 frames/video | Tooling available; 0.02% frames had no detection | Small domain shift |
| Frames per epoch | 32 per video, 50 epochs | sampled subset, few epochs | GPU budget | Lower AUC |
| Batch size / lr / lambdas | 24 / 2e-4 / 10, 2 | TBD | Memory | TBD |
| Cross-dataset sets | CDF-v1, CDF-v2, DFDC, DFDCP | CDF-v1 (100 test videos), CDF-v2 (518) | DFDCP access unavailable | No DFDCP column; CDF-v1 AUC is noisy |
| Splits | DeepfakeBench / official | Official FF++ 720/140/140 | Same | None |

## 7. Problems found in the Phase-2 submission (fixed in this rebuild)
1. One expert bank, router and shared vector shared across all 36 wrapped layers (about 14k adapter
   parameters vs about 0.67M here).
2. Fusion used input chunk 0 twice and never used chunk 3.
3. Hyperparameters: lambda 1/1, lr 1e-4, batch 2 (paper: 10/2, 2e-4, 24).
4. Custom 400/50/50 random pair split instead of the official 360/70/70 pairs.
