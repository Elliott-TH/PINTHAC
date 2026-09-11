"""
Shared matplotlib style for the Phase 7 portfolio figures (docs/brief/PHASE67_BRIEF.md part B).

Every figures/*.py script imports this module and calls apply() once, then figure() to
get a dark-background Axes matching docs/reference/index.html's own palette -- the exact
colors are pinned by that page's own <!-- PROMPT FOR CLAUDE CODE --> comments (grep it
for the hex codes): background #0a0d12, accent #6fd3f7, muted grey #8b93a1, light
gridlines, no title baked into the image (the caption lives in the HTML page, not the
SVG -- see docs/FIGURE_CAPTIONS.md).

Kept to one small module rather than a matplotlib style sheet file so every script's
`import style` line is explicit about where the look comes from, and so per-figure code
can still override a color locally (a highlighted series, say) without fighting a global
rcParam.
"""
import matplotlib.pyplot as plt

BG = "#0a0d12"          # page background
PANEL = "#10141c"       # slightly lighter than BG, for axes facecolor separation
ACCENT = "#6fd3f7"       # the one accent color, used for the single most important series
MUTED = "#8b93a1"        # secondary series / reference lines / CPU-side comparisons
GRID = "#2a3341"         # light gridlines against the dark background
TEXT = "#c7ccd4"         # body text / tick labels
TEXT_DIM = "#8b93a1"     # axis labels, less emphasis than tick text

# A small qualitative palette for figures needing more than accent+muted (e.g. three
# liquid-metal species): accent first, then desaturated blues/greens/ambers that stay
# legible on the dark background without competing with the one true accent color.
SERIES = [ACCENT, "#f7b96f", "#8bd39a", MUTED]


def apply():
    """
    Set the shared rcParams. Call once, near the top of a figure script, before creating
    any figure.

    Inputs: none
    Returns: none (mutates matplotlib's global rcParams)
    """
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_DIM,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 11,
        "legend.labelcolor": TEXT,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.framealpha": 0.9,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "font.size": 11,
        "font.family": "sans-serif",
        "svg.fonttype": "none",   # keep text as text in the SVG, not paths
    })


def figure(figsize=(7.0, 4.5), ncols=1, nrows=1, **kwargs):
    """
    A dark-styled Figure/Axes pair (or array of Axes), with no title set -- captions live
    in docs/FIGURE_CAPTIONS.md / the HTML page, never baked into the image per the site's
    own instructions.

    Inputs:
        figsize      : (width, height) in inches
        ncols, nrows : subplot grid shape
        **kwargs     : passed through to plt.subplots
    Returns:
        (fig, ax) : ax is a single Axes if nrows==ncols==1, else the array subplots()
                    returns
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    axes = ax.flat if hasattr(ax, "flat") else [ax]
    for a in axes:
        a.tick_params(colors=TEXT, which="both")
        for spine in a.spines.values():
            spine.set_color(GRID)
    return fig, ax


def finish(fig, out_path):
    """
    Tight-layout and save as SVG with the dark background preserved.

    Inputs:
        fig      : matplotlib Figure
        out_path : destination path, string -- should end in .svg
    Returns: none
    """
    fig.tight_layout()
    fig.savefig(out_path, format="svg", facecolor=BG)
    # A PNG beside the SVG so the figure can actually be looked at while it is being
    # built. The SVG is the deliverable; the PNG is never referenced by the page.
    fig.savefig(str(out_path).replace(".svg", ".png"), dpi=130,
                facecolor=fig.get_facecolor())
    print(f"Saved {out_path}")
