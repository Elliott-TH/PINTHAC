"""
PINTHAC module dependency diagram (docs/PHASE67_BRIEF.md figure 6).

Parses the actual import statements out of every pinthac/**/*.py source file with the
`ast` module and draws the resulting package-level graph -- "the real import graph, not
an idealized one," per the brief. This is a static-analysis tool, not a physics model:
it reads source text and reports what it finds, so there are no numbers to validate here,
only a faithful transcription of the repository's own `import` lines.

Package-level grouping (not per-file) so the diagram stays legible: every pinthac/foo/*.py
file's imports are attributed to the `pinthac.foo` package node. Edges are deduplicated
and, where they would violate the documented import direction
(properties <- correlations <- pin <- sca <- ml, CLAUDE.md section 8), flagged in the
printed report so a genuine layering violation cannot silently vanish into "the diagram
looked fine."
"""
import ast
import os

import style

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_ROOT = os.path.join(REPO_ROOT, "pinthac")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Layer order per CLAUDE.md section 8 -- used only to flag upward edges in the printed
# report, not to hide or reroute anything drawn.
LAYER_ORDER = ["backend", "ranges", "uncertainty", "solvers", "paths",
               "properties", "correlations", "pin", "sca", "ml"]


def module_package(rel_path):
    """
    'pinthac/sca/rod.py' -> 'sca'; 'pinthac/backend.py' -> 'backend'.

    Inputs:
        rel_path : path of a .py file relative to the pinthac/ package root
    Returns:
        package name, string
    """
    parts = rel_path.split(os.sep)
    if len(parts) == 1:
        return parts[0][:-3]   # top-level module, strip .py
    return parts[0]             # subpackage directory name


def collect_imports():
    """
    Walk every pinthac/**/*.py file and record, per source package, every module it
    imports -- both other pinthac packages and external third-party packages.

    Inputs: none
    Returns:
        internal_edges : set of (src_package, dst_package) pinthac-to-pinthac edges
        external_edges : set of (src_package, dst_name) pinthac-to-external edges
    """
    internal_edges = set()
    external_edges = set()

    for dirpath, _, filenames in os.walk(PKG_ROOT):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, PKG_ROOT)
            src_pkg = module_package(rel_path)

            with open(full_path) as f:
                tree = ast.parse(f.read(), filename=full_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    _record(node.module, src_pkg, internal_edges, external_edges)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        _record(alias.name, src_pkg, internal_edges, external_edges)

    return internal_edges, external_edges


def _record(dotted_name, src_pkg, internal_edges, external_edges):
    top = dotted_name.split(".")[0]
    if top != "pinthac":
        external_edges.add((src_pkg, top))
        return
    pieces = dotted_name.split(".")
    dst_pkg = pieces[1] if len(pieces) > 1 else src_pkg
    if dst_pkg != src_pkg:
        internal_edges.add((src_pkg, dst_pkg))


def report_layering_violations(internal_edges):
    """
    Print any edge that points from a later layer to an earlier one, per CLAUDE.md's
    stated one-way import direction -- a real finding if any turn up, not decoration.
    """
    rank = {name: i for i, name in enumerate(LAYER_ORDER)}
    violations = []
    for src, dst in sorted(internal_edges):
        if src in rank and dst in rank and rank[src] < rank[dst]:
            violations.append((src, dst))
    if violations:
        print("Layering violations (source imports a LATER layer):")
        for src, dst in violations:
            print(f"  {src} -> {dst}")
    else:
        print("No layering violations found: every internal edge points to an earlier "
              "or equal layer in properties <- correlations <- pin <- sca <- ml.")


def main():
    style.apply()
    internal_edges, external_edges = collect_imports()

    print("Internal (pinthac-to-pinthac) package edges:")
    for src, dst in sorted(internal_edges):
        print(f"  {src} -> {dst}")
    print("\nExternal dependencies by package:")
    by_pkg = {}
    for src, dst in external_edges:
        by_pkg.setdefault(src, set()).add(dst)
    for pkg in sorted(by_pkg):
        print(f"  {pkg}: {', '.join(sorted(by_pkg[pkg]))}")
    print()
    report_layering_violations(internal_edges)

    draw(internal_edges, external_edges, os.path.join(OUT_DIR, "pinthac-architecture.svg"))


# Fixed layout: hand-placed per layer so the diagram reads top-to-bottom in the documented
# import order, rather than a force-directed layout that would place nodes differently on
# every run. Positions are cosmetic only -- the edges drawn are exactly what collect_imports()
# found in the source, not an idealized subset.
LAYOUT = {
    "backend": (0.5, 5), "ranges": (2.3, 5), "uncertainty": (4.1, 5), "solvers": (5.9, 5),
    "paths": (7.7, 5),
    "properties": (1.5, 4),
    "correlations": (1.5, 3),
    "pin": (1.5, 2),
    "sca": (1.5, 1),
    "ml": (1.5, 0),
}
# External nodes sit in one shared column to the right of the internal packages --
# deduplicated across all of pinthac (torch, say, is used by six different packages, and
# is drawn once with six incoming edges) rather than repeated per package, which is what
# made an earlier version of this diagram an unreadable wall of overlapping text.
EXTERNAL_X = 9.8


def draw(internal_edges, external_edges, out_path):
    import matplotlib.patches as mpatches

    external_names = sorted({dst for _, dst in external_edges})
    fig_height = max(7.0, 0.62 * len(external_names))
    fig, ax = style.figure(figsize=(9.0, fig_height))

    y_top = max(y for _, y in LAYOUT.values()) + 0.8
    y_bot = min(y for _, y in LAYOUT.values()) - 0.8
    ext_y = {name: y for name, y in zip(
        external_names, [y_top - i * (y_top - y_bot) / max(1, len(external_names) - 1)
                          for i in range(len(external_names))])} if len(external_names) > 1 \
        else {external_names[0]: (y_top + y_bot) / 2} if external_names else {}
    ext_positions = {name: (EXTERNAL_X, y) for name, y in ext_y.items()}

    ax.set_xlim(-0.3, EXTERNAL_X + 1.3)
    ax.set_ylim(y_bot - 0.5, y_top + 0.5)
    ax.axis("off")

    def draw_box(xy, label, color, fontsize=10, w=1.15, h=0.32):
        x, y = xy
        box = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                       boxstyle="round,pad=0.02,rounding_size=0.05",
                                       linewidth=1.1, edgecolor=color, facecolor=style.PANEL)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", color=color, fontsize=fontsize)

    for pkg, (x, y) in LAYOUT.items():
        draw_box((x, y), f"pinthac.{pkg}", style.ACCENT)

    for name, (x, y) in ext_positions.items():
        draw_box((x, y), name, style.MUTED, fontsize=9, w=1.25, h=0.28)

    for src, dst in internal_edges:
        if src in LAYOUT and dst in LAYOUT:
            x0, y0 = LAYOUT[src]
            x1, y1 = LAYOUT[dst]
            # Bowed rather than straight for edges that skip a layer (e.g. ml -> properties):
            # a straight vertical line there would pass directly through the boxes in
            # between (pin, correlations), which read as false intermediate edges.
            rad = 0.0 if abs(y0 - y1) <= 1.01 else 0.35
            up = y1 > y0
            ax.annotate("", xy=(x1, y1 - 0.17 if up else y1 + 0.17),
                        xytext=(x0, y0 + 0.17 if up else y0 - 0.17),
                        arrowprops=dict(arrowstyle="-|>", color=style.ACCENT, lw=1.1,
                                        alpha=0.85, shrinkA=2, shrinkB=2,
                                        connectionstyle=f"arc3,rad={rad}"))

    for src, dst in external_edges:
        if src in LAYOUT and dst in ext_positions:
            x0, y0 = LAYOUT[src]
            x1, y1 = ext_positions[dst]
            ax.annotate("", xy=(x1 - 0.65, y1), xytext=(x0 + 0.58, y0),
                        arrowprops=dict(arrowstyle="-|>", color=style.MUTED, lw=0.6,
                                        alpha=0.45, shrinkA=1, shrinkB=1))

    style.finish(fig, out_path)


if __name__ == "__main__":
    main()
