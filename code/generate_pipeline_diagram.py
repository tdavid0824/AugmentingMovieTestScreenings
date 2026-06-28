#!/usr/bin/env python3
"""
Generate the thesis pipeline-flow diagram as a high-resolution PNG.

Output: ml_dataset/data/model_ready/movie_success_v6/tables/pipeline_diagram.png

Usage:
    python generate_pipeline_diagram.py

The diagram shows the data flow from raw multimodal recordings through to
modelling, with the column count at each stage annotated on the right.
Style constants at the top of the file can be edited to change colours,
fonts, or sizing.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).resolve().parent.parent.parent
OUT_DIR  = PROJECT / "ml_dataset" / "data" / "model_ready" / "movie_success_v6" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG  = OUT_DIR / "pipeline_diagram.png"

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONSTANTS — edit here to change overall aesthetic
# ─────────────────────────────────────────────────────────────────────────────
BOX_BG          = '#FFFFFF'   # box fill
BOX_EDGE        = '#2C3E50'   # box border (dark slate blue)
BOX_BG_ACCENT   = '#EAF1F8'   # subtle blue tint for emphasis boxes
TITLE_BG        = '#2C3E50'   # title bar background
TITLE_FG        = '#FFFFFF'   # title bar text
TEXT_PRIMARY    = '#1F2328'   # main text
TEXT_SECONDARY  = '#5C6B7A'   # sub-label text
ANNOTATION_FG   = '#5C6B7A'   # side-annotation text colour
ARROW_COLOR     = '#2C3E50'   # arrow colour
ARROW_LW        = 1.4
BOX_RADIUS      = 0.04        # rounded-corner radius for boxes
FONT_PRIMARY    = 10.5
FONT_SECONDARY  = 9
FONT_ANNOTATION = 9
FONT_TITLE      = 12
DPI             = 300

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGES — edit text here to change diagram content
# Each entry: (main_label, sub_label, side_annotation, emphasis)
#   emphasis=True highlights the box with the accent background
# ─────────────────────────────────────────────────────────────────────────────
STAGES = [
    (
        "Raw data from the Emognition dataset",
        "OpenFace · Quantum · Empatica · Muse · Samsung Watch · Self-report",
        "Per participant × amount of clips",
        True,
    ),
    (
        "Feature extraction",
        "Less resource-consuming and model-ready representation of data",
        "Per participant × amount of clips",
        False,
    ),
    (
        "Participant baseline correction",
        "stim − baseline → variables with _bc tags",
        "Inclusion of new baseline-adjusted variables",
        False,
    ),
    (
        "Participant-level feature rows",
        "Per (participant × clip)",
        "163 features",
        True,
    ),
    (
        "Aggregation across 43 viewers per movie",
        "Mean and standard deviation per base feature",
        "43 rows → 1 row per scene (generalized to movie-level in this research)",
        False,
    ),
    (
        "Movie-level feature row",
        "Mean + std per base feature",
        "327 columns",
        True,
    ),
    (
        "Manually collected metadata + dependent variables added",
        "Scene-level features · financial figures · IMDb / WOM",
        "All variables are now included",
        False,
    ),
    (
        "Exclude features to avoid data leakage",
        "imdb_rating · wom_multiplier_* · revenue_usd_* · opening_weekend_*\nroi_percent · success_class · budget_usd",
        "7 columns dropped\nfrom predictors",
        False,
    ),
    (
        "Modelling / Analysis",
        "Classical ML models and DL",
        "Testing the pipeline and evaluating performance",
        True,
    ),
]


def draw_pipeline_diagram(stages, save_path):
    """Render the vertical pipeline-flow diagram and save as PNG."""
    n_stages = len(stages)

    # Layout constants in axes coordinates
    fig_w, fig_h = 11, 14
    box_w  = 0.62                 # box width as fraction of axes
    box_h  = 0.075                # base box height
    box_x  = 0.18                 # left edge of boxes
    gap    = 0.020                # vertical gap between boxes
    y_top  = 0.92                 # top y for the first box
    arrow_offset = 0.005          # small vertical offset for arrows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ─── Title bar ───
    title_h = 0.04
    title_rect = FancyBboxPatch(
        (0.10, y_top + 0.025), 0.80, title_h,
        boxstyle=f"round,pad=0.005,rounding_size={BOX_RADIUS}",
        facecolor=TITLE_BG, edgecolor=TITLE_BG, linewidth=0,
        transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(title_rect)
    ax.text(0.50, y_top + 0.025 + title_h / 2,
            "Full overview of the data pipeline",
            fontsize=FONT_TITLE, fontweight='bold', color=TITLE_FG,
            ha='center', va='center', transform=ax.transAxes, zorder=2)

    # ─── Draw each stage box ───
    box_centres_y = []
    for i, (main, sub, annot, emphasis) in enumerate(stages):
        # Box height adjusts slightly if sub-label has wrapping
        h = box_h + (0.012 if '\n' in sub else 0)
        y = y_top - i * (box_h + gap + 0.012)
        bg = BOX_BG_ACCENT if emphasis else BOX_BG
        edge_w = 1.5 if emphasis else 1.0

        box = FancyBboxPatch(
            (box_x, y - h), box_w, h,
            boxstyle=f"round,pad=0.005,rounding_size={BOX_RADIUS}",
            facecolor=bg, edgecolor=BOX_EDGE, linewidth=edge_w,
            transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(box)

        # Main label
        ax.text(box_x + box_w / 2, y - h * 0.35,
                main,
                fontsize=FONT_PRIMARY, fontweight='bold', color=TEXT_PRIMARY,
                ha='center', va='center', transform=ax.transAxes, zorder=3)

        # Sub-label
        ax.text(box_x + box_w / 2, y - h * 0.72,
                sub,
                fontsize=FONT_SECONDARY, color=TEXT_SECONDARY,
                ha='center', va='center', transform=ax.transAxes, zorder=3)

        # Side annotation on the right
        if annot:
            ax.text(box_x + box_w + 0.04, y - h / 2,
                    annot,
                    fontsize=FONT_ANNOTATION, color=ANNOTATION_FG,
                    ha='left', va='center', style='italic',
                    transform=ax.transAxes, zorder=3)

        box_centres_y.append((y, y - h))

    # ─── Draw arrows between boxes ───
    for i in range(n_stages - 1):
        y_bottom_of_box_i      = box_centres_y[i][1]
        y_top_of_box_i_plus_1  = box_centres_y[i + 1][0]
        x_arrow = box_x + box_w / 2
        arrow = FancyArrowPatch(
            (x_arrow, y_bottom_of_box_i - arrow_offset),
            (x_arrow, y_top_of_box_i_plus_1 + arrow_offset),
            arrowstyle='-|>', mutation_scale=14,
            color=ARROW_COLOR, linewidth=ARROW_LW,
            transform=ax.transAxes, zorder=1,
        )
        ax.add_patch(arrow)

    # ─── Save ───
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  PNG saved → {save_path}")


if __name__ == "__main__":
    draw_pipeline_diagram(STAGES, OUT_PNG)
