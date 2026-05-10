"""
CLEAR-MoE++ pipeline figure.
Two-row layout for improved readability on page.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.join(BASE, "data", "imagenette2-320", "val")
SCORES_JSON = os.path.join(
    BASE,
    "outputs",
    "logs",
    "scores_deit_small_patch16_224_classification_E4_clear_moe",
    "layer_scores.json",
)
ROUTING_JSON = os.path.join(
    BASE,
    "outputs",
    "logs",
    "routers_deit_small_patch16_224_classification_E4_clear_moe",
    "routing_stats.json",
)
OUT_PDF = os.path.join(BASE, "figures", "clear_moe_pipeline_real.pdf")
OUT_PNG = os.path.join(BASE, "figures", "clear_moe_pipeline_real.png")

# Palette
BG = "#F1F5F9"
WHITE = "#FFFFFF"
DARK = "#0F172A"
GRAY = "#64748B"
LGRAY = "#E2E8F0"
C_IN = "#334155"
C_P1 = "#0284C7"
C_P2 = "#D97706"
C_P3 = "#059669"
C_P4 = "#7C3AED"
C_OUT = "#334155"
C_SEL = "#10B981"
C_SKIP = "#CBD5E1"
C_THR = "#EF4444"
EXPERT_COLORS = ["#EF4444", "#0284C7", "#10B981", "#F59E0B"]

PANEL_COLORS = [C_IN, C_P1, C_P2, C_P3, C_P4, C_OUT]
PANEL_TITLES = [
    "1  Input Images",
    "2  Calibration Pass",
    "3  Layer Scoring",
    "4  Expert Extraction",
    "5  Router & Dispatch",
    "6  MoE Output",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.facecolor": WHITE,
})

# Data
CLASSES = ["n01440764", "n03028079", "n03445777", "n03888257"]
CLABELS = ["Tench", "Church", "Golf Ball", "Parachute"]


def load_img(cls_id, size=128):
    d = os.path.join(VAL_DIR, cls_id)
    f = sorted(
        fn for fn in os.listdir(d)
        if fn.lower().endswith((".jpeg", ".jpg", ".png"))
    )[0]
    img = Image.open(os.path.join(d, f)).convert("RGB")
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


imgs = [load_img(c) for c in CLASSES]

with open(SCORES_JSON, "r", encoding="utf-8") as f:
    sd = json.load(f)
scores_sorted = sorted(sd["scores"], key=lambda x: x["layer_index"])
L_idx = [s["layer_index"] for s in scores_sorted]
L_comp = [s["composite"] for s in scores_sorted]
L_spar = [s["sparsity"] for s in scores_sorted]
L_clust = [s["multimodality"] for s in scores_sorted]
L_sens = [s["sensitivity"] for s in scores_sorted]
L_sel = [s["selected"] for s in scores_sorted]
threshold = np.median(L_comp)

with open(ROUTING_JSON, "r", encoding="utf-8") as f:
    rd = json.load(f)
fracs = rd["blocks.8.mlp"]["expert_load_fractions"]

# Figure
fig = plt.figure(figsize=(16.2, 11.2))
fig.patch.set_facecolor(BG)
outer = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    left=0.02,
    right=0.985,
    top=0.90,
    bottom=0.04,
    hspace=0.18,
    wspace=0.08,
)


def build_panel(spec, title, color):
    panel = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=spec,
        height_ratios=[0.18, 0.82],
        hspace=0.0,
    )
    header = fig.add_subplot(panel[0, 0])
    header.set_facecolor(color)
    header.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=WHITE,
    )
    header.set_xticks([])
    header.set_yticks([])
    for spine in header.spines.values():
        spine.set_visible(False)

    content = fig.add_subplot(panel[1, 0])
    content.set_facecolor(WHITE)
    for spine in content.spines.values():
        spine.set_edgecolor(LGRAY)
        spine.set_linewidth(0.9)
    return header, content


# Row 1: Input, calibration, scoring
_, ax_in = build_panel(outer[0, 0], PANEL_TITLES[0], PANEL_COLORS[0])
_, ax_cal = build_panel(outer[0, 1], PANEL_TITLES[1], PANEL_COLORS[1])
_, ax_score = build_panel(outer[0, 2], PANEL_TITLES[2], PANEL_COLORS[2])

# Row 2: extraction, router, output
_, ax_extract = build_panel(outer[1, 0], PANEL_TITLES[3], PANEL_COLORS[3])
_, ax_route = build_panel(outer[1, 1], PANEL_TITLES[4], PANEL_COLORS[4])
_, ax_out = build_panel(outer[1, 2], PANEL_TITLES[5], PANEL_COLORS[5])

# Input panel
ax_in.axis("off")
inner_in = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax_in.get_subplotspec(), hspace=0.07, wspace=0.05)
for i, (img_arr, lbl) in enumerate(zip(imgs, CLABELS)):
    ax = fig.add_subplot(inner_in[i // 2, i % 2])
    ax.imshow(img_arr)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(lbl, fontsize=7, color=DARK, labelpad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(LGRAY)

# Calibration panel
x = np.array(L_idx)
ax_cal.plot(x, L_spar, "o-", color=C_P1, lw=1.6, ms=4.5, label="Sparsity", markerfacecolor=WHITE, markeredgewidth=1.2)
ax_cal.plot(x, L_clust, "s-", color=C_P3, lw=1.6, ms=4.5, label="Clusterability", markerfacecolor=WHITE, markeredgewidth=1.2)
ax_cal.plot(x, L_sens, "^-", color=C_P4, lw=1.6, ms=4.5, label="Sensitivity", markerfacecolor=WHITE, markeredgewidth=1.2)
ax_cal.set_xlim(-0.5, 11.5)
ax_cal.set_xticks([0, 3, 6, 9, 11])
ax_cal.set_xticklabels(["L0", "L3", "L6", "L9", "L11"], fontsize=7)
ax_cal.set_ylabel("Score", fontsize=7.5, color=DARK)
ax_cal.set_xlabel("FFN layer index", fontsize=7.5, color=DARK)
ax_cal.set_title("Activation statistics\n(200 calib. images, DeiT-Small)", fontsize=8.5, color=DARK, pad=6)
ax_cal.axvspan(5.5, 11.5, alpha=0.08, color=C_SEL, zorder=0)
ax_cal.text(8.5, 0.95, "selected\nlayers", ha="center", va="top", fontsize=6.2, color=C_SEL, transform=ax_cal.get_xaxis_transform(), style="italic")
leg = ax_cal.legend(fontsize=6.4, loc="upper left", framealpha=0.92, edgecolor=LGRAY, handlelength=1.4)
leg.get_frame().set_linewidth(0.6)
ax_cal.tick_params(colors=GRAY, labelsize=6.8)

# Scoring panel
bar_c = [C_SEL if s else C_SKIP for s in L_sel]
y_pos = np.arange(len(L_idx))
bars = ax_score.barh(y_pos, L_comp, color=bar_c, edgecolor=WHITE, linewidth=0.4, height=0.72)
ax_score.axvline(threshold, color=C_THR, lw=1.4, ls="--", zorder=5)
ax_score.text(threshold + 0.003, 5.8, f"θ={threshold:.3f}", color=C_THR, fontsize=6, va="center")
for bar, val in zip(bars, L_comp):
    ax_score.text(val + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha="left", fontsize=5.6, color=DARK)
ax_score.set_yticks(y_pos)
ax_score.set_yticklabels([f"L{i}" for i in L_idx], fontsize=6.5)
ax_score.set_xlabel("Composite score $\\mathcal{S}(l)$", fontsize=7.5, color=DARK)
ax_score.set_title("Layer scoring & selection\n(last-half policy, L6–L11 selected)", fontsize=8.5, color=DARK, pad=6)
ax_score.set_xlim(0, max(L_comp) + 0.08)
ax_score.invert_yaxis()
sel_p = mpatches.Patch(facecolor=C_SEL, label="Selected", edgecolor="none")
skip_p = mpatches.Patch(facecolor=C_SKIP, label="Skipped", edgecolor="none")
ax_score.legend(handles=[sel_p, skip_p], fontsize=6.4, loc="lower right", framealpha=0.92, edgecolor=LGRAY).get_frame().set_linewidth(0.6)
ax_score.spines["left"].set_visible(False)
ax_score.tick_params(left=False, colors=GRAY, labelsize=6.8)

# Extraction panel
ax_extract.axis("off")
ax_extract.set_xlim(0, 10)
ax_extract.set_ylim(0, 10)


def rbox(ax, x, y, w, h, fc, ec, lw=1.3, radius=0.18):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=3,
        )
    )


def arrow3(ax, x0, y0, x1, y1, color=DARK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=10))

rbox(ax_extract, 0.20, 3.95, 1.45, 2.55, "#EFF6FF", C_P1, lw=1.3, radius=0.16)
ax_extract.text(0.93, 5.20, "$W$", ha="center", va="center", fontsize=15, fontweight="bold", color=C_P1)
ax_extract.text(0.93, 4.55, "dense FFN", ha="center", va="center", fontsize=7.0, color=C_P1)

arrow3(ax_extract, 1.65, 5.85, 3.00, 7.25)
ax_extract.text(2.20, 6.95, "SVD", ha="center", fontsize=7.0, color=DARK, style="italic")

rbox(ax_extract, 3.05, 7.25, 2.65, 1.65, "#ECFDF5", C_P3, lw=1.3, radius=0.16)
ax_extract.text(4.38, 8.10, "$W_{\\mathrm{shared}}$", ha="center", va="center", fontsize=9.6, fontweight="bold", color=C_P3)
ax_extract.text(4.38, 7.58, "rank-$r$ basis", ha="center", va="center", fontsize=7.0, color=C_P3)

arrow3(ax_extract, 1.65, 4.55, 3.00, 5.35, C_P2)
ax_extract.text(2.25, 4.75, "$k$-means\n$E{=}4$", ha="center", fontsize=6.8, color=C_P2)

for ei, ey in enumerate([1.05, 2.55, 4.05, 5.55]):
    fc = f"{EXPERT_COLORS[ei]}18"
    rbox(ax_extract, 3.05, ey, 2.65, 1.15, fc, EXPERT_COLORS[ei], lw=1.2, radius=0.13)
    ax_extract.text(4.38, ey + 0.58, f"$W_{{{ei+1}}}^{{\\mathrm{{res}}}}$  Expert {ei+1}", ha="center", va="center", fontsize=6.8, color=EXPERT_COLORS[ei], fontweight="bold")

arrow3(ax_extract, 5.72, 5.0, 7.15, 5.0)
ax_extract.text(6.43, 5.30, "combine", ha="center", fontsize=6.7, color=DARK, style="italic")

rbox(ax_extract, 7.15, 3.55, 2.45, 2.75, "#FDF4FF", C_P4, lw=1.3, radius=0.20)
ax_extract.text(8.38, 5.15, "MoE", ha="center", va="center", fontsize=12.8, fontweight="bold", color=C_P4)
ax_extract.text(8.38, 4.45, "layer", ha="center", va="center", fontsize=9.6, color=C_P4)
ax_extract.set_title("SVD + $k$-means decomposition\n(recon. error 0.47–0.52 across L6–L11)", fontsize=8.4, color=DARK, pad=6)

# Router panel
ax_route.axis("off")
inner_route = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax_route.get_subplotspec(), hspace=0.44)
ax_donut = fig.add_subplot(inner_route[0, 0])
ax_bar = fig.add_subplot(inner_route[1, 0])
for ax in (ax_donut, ax_bar):
    ax.set_facecolor(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(LGRAY)
        spine.set_linewidth(0.9)

wedges, texts, autotexts = ax_donut.pie(
    fracs,
    colors=EXPERT_COLORS,
    autopct="%1.0f%%",
    startangle=110,
    pctdistance=0.72,
    wedgeprops=dict(width=0.52, edgecolor=WHITE, linewidth=1.5),
    textprops=dict(fontsize=6.4),
)
for at in autotexts:
    at.set_fontsize(6.4)
    at.set_fontweight("bold")
    at.set_color(WHITE)
for t, lbl in zip(texts, [f"E{i+1}" for i in range(4)]):
    t.set_text(lbl)
    t.set_fontsize(7)
    t.set_color(DARK)
ax_donut.set_title("Expert load distribution\n(block L8, real routing stats)", fontsize=8.4, color=DARK, pad=5)

backends = ["Naive", "Grouped", "Stream", "Triton\nS-G", "cuBLAS\nGEMM"]
tput = [131.6, 148.5, 152.8, 176.1, 243.0]
b_colors = [C_SKIP, C_SKIP, C_P1, C_P4, C_P3]
yp = np.arange(len(backends))
hb = ax_bar.barh(yp, tput, color=b_colors, edgecolor=WHITE, linewidth=0.4, height=0.65)
for bar, val in zip(hb, tput):
    ax_bar.text(val + 2, bar.get_y() + bar.get_height() / 2, f"{val:.0f}k", va="center", ha="left", fontsize=6.4, color=DARK)
ax_bar.set_yticks(yp)
ax_bar.set_yticklabels(backends, fontsize=6.8)
ax_bar.set_xlabel("Throughput (tok/s ×10³)", fontsize=7.3, color=DARK)
ax_bar.set_xlim(0, 285)
ax_bar.set_title("Dispatch backend throughput\n(0% imbalance, $E{=}4$, $d{=}384$)", fontsize=8.4, color=DARK, pad=5)
ax_bar.spines["left"].set_visible(False)
ax_bar.tick_params(left=False, colors=GRAY, labelsize=6.8)

# Output panel
ax_out.axis("off")
inner_out = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax_out.get_subplotspec(), hspace=0.07, wspace=0.05)
expert_assign = [2, 1, 4, 3]
for i, (img_arr, lbl) in enumerate(zip(imgs, CLABELS)):
    ax = fig.add_subplot(inner_out[i // 2, i % 2])
    ax.imshow(img_arr)
    ea = expert_assign[i]
    ec = EXPERT_COLORS[ea - 1]
    ax.text(
        0.96,
        0.96,
        f"E{ea}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=WHITE,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=ec, alpha=0.94, edgecolor=WHITE, linewidth=0.8),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(lbl, fontsize=7, color=DARK, labelpad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(LGRAY)

# Titles
fig.text(
    0.5,
    0.965,
    "CLEAR-MoE++  ·  Calibration-Driven Post-Training Expert Extraction from Pretrained Vision Transformers",
    ha="center",
    va="center",
    fontsize=12.2,
    fontweight="bold",
    color=DARK,
)
fig.text(
    0.5,
    0.945,
    "DeiT-Small on Imagenette  ·  $E{=}4$ experts  ·  6 of 12 FFN layers expertised  ·  cuBLAS dispatch: 243k tok/s",
    ha="center",
    va="center",
    fontsize=8.4,
    color=GRAY,
)
# Save
fig.savefig(OUT_PDF, dpi=220, bbox_inches="tight", facecolor=BG)
fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
plt.close(fig)
