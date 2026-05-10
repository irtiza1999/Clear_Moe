from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_card(ax, x, y, width, height, accent, title, subtitle, lines, badge, body_fontsize=8.6):
    shadow = FancyBboxPatch(
        (x + 0.03, y - 0.03),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0,
        facecolor="#0f172a",
        alpha=0.08,
        zorder=1,
    )
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4,
        edgecolor=accent,
        facecolor="#ffffff",
        zorder=2,
    )
    ax.add_patch(shadow)
    ax.add_patch(card)

    badge_circle = Circle((x + 0.18, y + height - 0.22), 0.11, facecolor=accent, edgecolor="none", zorder=3)
    ax.add_patch(badge_circle)
    ax.text(x + 0.18, y + height - 0.22, badge, ha="center", va="center", fontsize=9, color="white", weight="bold", zorder=4)

    ax.text(x + 0.38, y + height - 0.14, title, ha="left", va="top", fontsize=11, color="#0f172a", weight="bold", zorder=4)
    ax.text(x + 0.38, y + height - 0.40, subtitle, ha="left", va="top", fontsize=8.5, color=accent, weight="semibold", zorder=4)

    text_y = y + height - 0.67
    for line in lines:
        ax.text(x + 0.18, text_y, f"• {line}", ha="left", va="top", fontsize=body_fontsize, color="#334155", zorder=4)
        text_y -= 0.22


def add_arrow(ax, start, end, color):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.2,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3,rad=0.0",
        zorder=3,
    )
    ax.add_patch(arrow)


def main():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#f8fafc",
            "savefig.facecolor": "#f8fafc",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(11.8, 4.25), dpi=300)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    # Background atmosphere.
    ax.add_patch(FancyBboxPatch((0.16, 0.16), 11.35, 3.72, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=1.0, zorder=0))
    ax.add_patch(Circle((10.95, 3.16), 0.65, facecolor="#dbeafe", edgecolor="none", alpha=0.86, zorder=0))
    ax.add_patch(Circle((1.1, 0.72), 0.48, facecolor="#dcfce7", edgecolor="none", alpha=0.72, zorder=0))
    ax.add_patch(FancyBboxPatch((0.55, 3.74), 3.2, 0.05, boxstyle="round,pad=0.0,rounding_size=0.03", facecolor="#2563eb", edgecolor="none", zorder=0.5, alpha=0.8))

    ax.text(0.55, 3.58, "CLEAR-MoE++ pipeline", fontsize=17, weight="bold", color="#0f172a", ha="left", va="center")
    ax.text(0.55, 3.28, "From a frozen dense backbone to routed conditional inference", fontsize=9.6, color="#475569", ha="left", va="center")

    cards = [
        (0.7, 1.05, 2.45, 1.88, "#2563eb", "Calibration", "Phase 1", ["Hook FFN activations", "Collect 200 images", "Store calibration traces"], "1"),
        (3.48, 1.05, 2.45, 1.88, "#0ea5e9", "Scoring", "Phase 2", ["Rank by sparsity", "clusterability, sensitivity", "Select top layers"], "2"),
        (6.26, 1.05, 2.45, 1.88, "#14b8a6", "Extraction", "Phase 3", ["Shared SVD basis", "$k$-means residual experts", "Freeze dense weights"], "3"),
    ]

    for x, y, width, height, accent, title, subtitle, lines, badge in cards:
        add_card(ax, x, y, width, height, accent, title, subtitle, lines, badge)

    add_card(ax, 8.82, 1.05, 2.12, 1.88, "#8b5cf6", "Deployment", "Phase 4", ["Fit routers", "Select backend", "Latency / quality eval"], "4", body_fontsize=8.15)

    add_arrow(ax, (3.17, 1.98), (3.42, 1.98), "#64748b")
    add_arrow(ax, (5.95, 1.98), (6.20, 1.98), "#64748b")
    add_arrow(ax, (8.73, 1.98), (8.80, 1.98), "#64748b")

    ax.text(0.7, 0.72, "Output: conditional MoE inference with measurable latency gains", fontsize=9.6, color="#0f172a", weight="bold")
    ax.text(0.7, 0.42, "Router variants + dispatch backends are selected per workload, while dense weights stay frozen.", fontsize=8.8, color="#475569")

    out_pdf = OUT_DIR / "clear_moe_pipeline_modern.pdf"
    out_png = OUT_DIR / "clear_moe_pipeline_modern.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()