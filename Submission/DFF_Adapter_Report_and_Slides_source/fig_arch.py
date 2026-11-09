import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix","font.size":7})
fig, ax = plt.subplots(figsize=(7.2, 5.3)); ax.set_xlim(0,100); ax.set_ylim(-7,72); ax.axis("off")
W="#FFFFFF"; G="#DEDEDE"; G2="#B8B8B8"
def box(x,y,w,h,t,fill=W,lw=0.8,fs=6.6,ls="-",bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.0,rounding_size=0.5",fc=fill,ec="black",lw=lw,ls=ls))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,linespacing=1.15,fontweight="bold" if bold else "normal")
def arr(x1,y1,x2,y2,ls="-",lw=0.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=6.5,lw=lw,color="black",ls=ls,shrinkA=0,shrinkB=0))
def line(xs,ys,ls="-",lw=0.8): ax.plot(xs,ys,color="black",lw=lw,ls=ls,solid_capstyle="butt")
def title(x,y,t): ax.text(x,y,t,fontsize=7.6,fontweight="bold",va="top")

# ---------- (a) ----------
title(0,71.5,"(a) Overall model. Small: $d{=}384$, Base: $d{=}768$; 12 blocks each")
ya,hh=55.5,11
box(0.5,ya,11.5,hh,"Face crop\n$224{\\times}224{\\times}3$")
box(15,ya,19,hh,"Patch embedding (14)\n+ [CLS] + position\n256 patches, 257 tokens")
box(37.5,ya,19.5,hh,"$12\\times$ transformer block\nDINOv2 weights frozen,\nDFF-Adapters trained\n(panels b, c)",fill=G,lw=1.3)
box(60.5,ya+2,9,hh-4,"[CLS]\nfeature\n($d$)")
box(73,ya+6.2,15,4.6,"Authenticity head\nLinear($d\\to1$), task 0",fill=G,lw=1.3,fs=6.3)
box(73,ya-0.8,15,4.6,"Forgery-type head\nLinear($d\\to5$), task 1",fill=G,lw=1.3,fs=6.3)
box(90.5,ya+5.4,9.5,6.2,"Real/fake\nscore",fs=6.4)
box(90.5,ya-1.6,9.5,6.2,"Type logits\n(training)",fs=6.4,ls="--")
cy=ya+hh/2
arr(12,cy,14.7,cy); arr(34.3,cy,37.2,cy); arr(57.3,cy,60.2,cy)
arr(69.8,cy,72.7,ya+8.5); arr(69.8,cy,72.7,ya+1.5)
arr(88.3,ya+8.5,90.2,ya+8.5); arr(88.3,ya+1.5,90.2,ya+1.5,ls="--")
ax.text(50,ya-3.6,"Each batch is passed twice: task 0 $\\rightarrow$ authenticity (BCE), task 1 $\\rightarrow$ forgery type (CE). Inference uses task 0 only.",fontsize=6.3,ha="center",va="top",style="italic")

# ---------- (b) ----------
title(0,47.6,"(b) One transformer block ($\\ell=1..12$)")
bx,bw=14,17
ys=[3.5,11.5,19.5,27.5,35.5]   # LN1? build bottom-up
box(bx,3.5,bw,4.4,"Layer norm")
box(bx,10.6,bw,9.2,"Multi-head self-\nattention (frozen):\n$W_Q,W_K,W_V,W_O$")
ax.add_patch(Circle((bx+bw/2,23.6),1.3,fc="white",ec="black",lw=0.8)); ax.text(bx+bw/2,23.6,"+",ha="center",va="center",fontsize=7)
box(bx,27.0,bw,4.0,"Layer norm")
box(bx,33.4,bw,4.0,"MLP (frozen)")
ax.add_patch(Circle((bx+bw/2,40.4),1.3,fc="white",ec="black",lw=0.8)); ax.text(bx+bw/2,40.4,"+",ha="center",va="center",fontsize=7)
mx=bx+bw/2
arr(mx,0.7,mx,3.3); arr(mx,7.9,mx,10.4); arr(mx,19.8,mx,22.2); arr(mx,24.9,mx,26.8); arr(mx,31.0,mx,33.2); arr(mx,37.4,mx,39.0)
ax.text(mx,0.2,"$x_\\ell$",ha="center",va="top",fontsize=7.5)
ax.text(mx,42.2,"$x_{\\ell+1}$",ha="center",va="bottom",fontsize=7.5)
# residuals
rx=bx+bw+2.6
line([mx,mx],[1.8,1.8]); line([mx+0.0,rx],[1.6,1.6]); line([rx,rx],[1.6,23.6]); arr(rx,23.6,mx+1.4,23.6)
rx2=bx+bw+5.6
line([mx,mx],[24.6,24.6]); line([mx+1.4,rx2],[23.6+0.0,23.6]); 
line([rx2,rx2],[23.6,40.4]); arr(rx2,40.4,mx+1.4,40.4)
# adapter box
box(0.4,12.2,10.6,6.0,"DFF-Adapter\non $W_Q,W_V,W_O$",fill=G,lw=1.3,fs=6.2)
arr(11.0,15.2,13.8,15.2,ls="--")

# ---------- (c) ----------
title(44,47.6,"(c) One DFF-Adapter (width $d$, task $t\\in\\{0,1\\}$)")
ax.text(44.2,41.2,"$x$",fontsize=8,va="center")
cx0=47.5; cw=8.0; hx=61.5; hw=25.5
ys=[34.8,26.2,17.6,9.0]; hh2=6.4
labels=["chunk 0","chunk 1","chunk 2","chunk 3"]
heads=["Shared head: all 6 experts,\nweights softmax($Z_{shared}$)",
       "Task head 1: top-3 of 6 experts,\nweights from $Z_{task}[t,1]$",
       "Task head 2: top-3 of 6 experts,\nweights from $Z_{task}[t,2]$",
       "Task head 3: top-3 of 6 experts,\nweights from $Z_{task}[t,3]$"]
for i,(y,l,h) in enumerate(zip(ys,labels,heads)):
    box(cx0,y,cw,hh2,f"{l}\n$d/4$")
    box(hx,y,hw,hh2,h,fill=G2 if i==0 else G,lw=1.3,fs=6.1)
    arr(cx0+cw,y+hh2/2,hx-0.2,y+hh2/2)
    arr(hx+hw,y+hh2/2,hx+hw+2.0,y+hh2/2) if False else None
# split bracket from x
line([46.2,46.2],[ys[3]+hh2/2,ys[0]+hh2/2]); 
for y in ys: arr(46.2,y+hh2/2,cx0-0.2,y+hh2/2)
line([44.8,46.2],[41.2-0.0,41.2]) if False else None
line([46.2,46.2],[ys[0]+hh2/2,41.2]); line([45.2,46.2],[41.2,41.2])
# concat
box(90.0,ys[3],5.0,ys[0]+hh2-ys[3],"concat\n$\\times\\beta$",fs=6.2)
for y in ys: line([hx+hw,90.0],[y+hh2/2,y+hh2/2])
arr(95.0,(ys[3]+ys[0]+hh2)/2,99.8,(ys[3]+ys[0]+hh2)/2)
ax.text(97.4,(ys[3]+ys[0]+hh2)/2+1.2,"$\\Delta Wx$",fontsize=7,ha="center",va="bottom")
ax.text(44.2,6.4,"Every head owns its own bank of 6 LoRA experts; expert $j$ is $B_jA_j$\nwith $A_j: d/4\\to4$, $B_j: 4\\to d/4$ ($r_h=r/4$).\n"
        "$r=16$, $\\alpha=32$, $\\beta=\\alpha/r=2$. $\\Delta Wx$ is added to the frozen layer output: $y=Wx+\\Delta Wx$.",fontsize=6.0,va="top",ha="left")
# legend
ax.add_patch(FancyBboxPatch((44.2,-5.6),3,2.2,boxstyle="round,pad=0,rounding_size=0.3",fc=W,ec="black",lw=0.8)); ax.text(48.2,-4.5,"frozen",fontsize=6.3,va="center")
ax.add_patch(FancyBboxPatch((57,-5.6),3,2.2,boxstyle="round,pad=0,rounding_size=0.3",fc=G,ec="black",lw=1.3)); ax.text(61,-4.5,"trainable",fontsize=6.3,va="center")
ax.add_patch(FancyBboxPatch((72,-5.6),3,2.2,boxstyle="round,pad=0,rounding_size=0.3",fc=G2,ec="black",lw=1.3)); ax.text(76,-4.5,"trainable, shared across tasks",fontsize=6.3,va="center")
plt.savefig("fig_arch.pdf",bbox_inches="tight"); plt.savefig("fig_arch.png",dpi=250,bbox_inches="tight"); print("saved")
