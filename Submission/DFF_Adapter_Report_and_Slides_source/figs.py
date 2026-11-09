import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

# ---------- Fig A: cross-dataset video AUC ----------
models = ["Small-12", "Small-30", "Small-30+LS", "Base-12", "Base-30", "Paper (Base)", "Paper (Large)"]
cdf1 = [81.20, 82.98, 81.20, 82.43, 81.71, np.nan, 96.14]
cdf2 = [85.02, 86.25, 84.74, 81.51, 84.22, 93.35, 95.26]
dfdc = [np.nan, 88.03, 87.45, np.nan, 84.00, 82.00, 89.96]
x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for i,(v,l,c) in enumerate([(cdf1,"Celeb-DF-v1","#4C72B0"),(cdf2,"Celeb-DF-v2","#55A868"),(dfdc,"DFDC subset","#C44E52")]):
    b = ax.bar(x+(i-1)*w, v, w, label=l, color=c)
    for r,val in zip(b,v):
        if not np.isnan(val): ax.text(r.get_x()+r.get_width()/2, val+0.4, f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
ax.set_ylim(70,102); ax.set_ylabel("Video-level AUC (%)")
ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
ax.legend(ncol=3, fontsize=8, loc="upper left", frameon=False)
ax.spines[["top","right"]].set_visible(False)
ax.text(5-w, 71.3, "n/r", fontsize=6.5, ha="center", color="gray")
plt.tight_layout(); plt.savefig("fig_auc.png", dpi=220); plt.close()

# ---------- Fig B: in-domain vs cross-dataset (Small-30) ----------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
labs = ["FF++\n(val)", "CDF-v1", "CDF-v2", "DFDC"]
ours = [99.08, 82.98, 86.25, 88.03]
paper = [99.56, 96.14, 95.26, 89.96]
x = np.arange(4); w = 0.38
ax.bar(x-w/2, ours, w, label="Ours (Small-30)", color="#4C72B0")
ax.bar(x+w/2, paper, w, label="Paper (Large-50)", color="#8C8C8C")
for xi,v in zip(x-w/2, ours): ax.text(xi, v+0.4, f"{v:.1f}", ha="center", fontsize=6.5)
for xi,v in zip(x+w/2, paper):
    if not np.isnan(v): ax.text(xi, v+0.4, f"{v:.1f}", ha="center", fontsize=6.5)
ax.set_ylim(70,103); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("Video AUC (%)")
ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5,1.0), ncol=2); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_gap.png", dpi=220); plt.close()

# ---------- Fig C: pipeline ----------
fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.set_xlim(0,102); ax.set_ylim(0,34); ax.axis("off")
def box(x,y,w,h,t,fc="#E8EEF7",ec="#4C72B0",fs=7.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,lw=1,color="#333"))
box(1,12,12.5,10,"Video\n(FF++ c23)",fs=7)
box(17.5,12,16.5,10,"MTCNN face crop\n224$\\times$224\n32 frames/video",fs=7)
box(36,12,15,10,"Frozen DINOv2\n(Small / Base)\nViT blocks 1..L",fc="#EDEDED",ec="#777")
box(36,25,15,7,"DFF-Adapter in\nevery block (LoRA)",fc="#FBE9D0",ec="#DD8452")
box(56,22,16,8,"Authenticity\nbranch",fc="#DDF0E1",ec="#55A868")
box(56,12,16,8,"Shared\nbranch",fc="#FBE9D0",ec="#DD8452")
box(56,2,16,8,"Forgery-type\nbranch",fc="#F6DADA",ec="#C44E52")
box(76,17,24,10,"Real / Fake\n(inference: auth.+shared)\nBCE, weight 10",fc="#DDF0E1",ec="#55A868",fs=6.5)
box(76,2,24,10,"5-way forgery type\n(training only)\nCE, weight 2 (+LS 0.1)",fc="#F6DADA",ec="#C44E52",fs=6.5)
arr(14,17,17.1,17); arr(34.4,17,35.6,17); arr(43.5,25,43.5,22.4)
arr(51.4,19,55.6,25); arr(51.4,17,55.6,16); arr(51.4,15,55.6,6)
arr(72.4,26,75.6,23); arr(72.4,16,75.6,21); arr(72.4,6,75.6,7)
arr(64,12.4,64,9.6)
plt.savefig("fig_pipeline.png", dpi=220, bbox_inches="tight"); plt.close()
print("ok")
