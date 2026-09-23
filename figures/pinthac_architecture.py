"""PINTHAC module dependency diagram

Package-level grouping (not per-file) so the diagram stays legible: every pinthac/foo/*.py
file's imports are attributed to the `pinthac.foo` package node. Edges are deduplicated
and, where they would violate the documented import direction
(properties <- correlations <- pin <- sca <- ml, CONTRIBUTING.md section 8), flagged in the
printed report so a genuine layering violation cannot silently vanish into "the diagram
looked fine."
"""
import ast
import os

import style

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_ROOT = os.path.join(REPO_ROOT, "pinthac")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Layer order per CONTRIBUTING.md section 8 -- used only to flag upward edges in the printed
# report, not to hide or reroute anything drawn.
LAYER_ORDER = ["backend", "ranges", "uncertainty", "solvers", "paths",
               "properties", "correlations", "pin", "sca", "ml"]


def module_package(rel_path):
    """'pinthac/sca/rod.py' -> 'sca'; 'pinthac/backend.py' -> 'backend'.

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
    """Walk every pinthac/**/*.py file and record, per source package, every module it
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
    """Print any edge that points from a later layer to an earlier one, per CONTRIBUTING.md's
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


LAYOUT = {
    "backend": (0.5, 5), "ranges": (2.3, 5), "uncertainty": (4.1, 5), "solvers": (5.9, 5),
    "paths": (7.7, 5),
    "properties": (1.5, 4),
    "correlations": (1.5, 3),
    "pin": (1.5, 2),
    "sca": (1.5, 1),
    "ml": (1.5, 0),
}
EXTERNAL_X = 9.8


def layer_stats():
    """Module names and line counts per layer, read off the repository.

    Why this is here: the first version of this figure showed only which layer imports
    which, which is true but tells a reader almost nothing they could not guess from the
    directory listing. Where the code actually *is* -- that properties is three times the
    size of pin, that the whole foundation is under 700 lines -- is the part that conveys
    the shape of the library. These numbers are counted at draw time, so the figure cannot
    drift out of date the way a hand-written caption would.

    Returns:
        dict of layer name -> (module name list, file count, total line count)
    """
    out = {}
    for layer in ("properties", "correlations", "pin", "sca", "ml"):
        d = os.path.join(PKG_ROOT, layer)
        files = sorted(f for f in os.listdir(d)
                       if f.endswith(".py") and f != "__init__.py")
        lines = sum(len(open(os.path.join(d, f), encoding="utf-8").readlines())
                    for f in files)
        out[layer] = ([f[:-3] for f in files], len(files), lines)

    root_files = sorted(f for f in os.listdir(PKG_ROOT)
                        if f.endswith(".py") and f != "__init__.py")
    root_lines = sum(len(open(os.path.join(PKG_ROOT, f), encoding="utf-8").readlines())
                     for f in root_files)
    out["foundation"] = ([f[:-3] for f in root_files], len(root_files), root_lines)
    return out


def draw(internal_edges, external_edges, out_path):
    """Draw the layer stack, with each layer's contents and size.

    The drawing is deliberately lossy where the printed report is not. Internal edges
    become the layer chain; anything that skips a layer is still counted and printed by
    report_layering_violations(), so a genuine violation cannot vanish into "the diagram
    looked fine". External dependencies are listed beside the layer that pulls them in
    rather than wired to it -- an earlier version drew one arrow per dependency per
    package, and forty muted arrows crossing unrelated boxes buried the one thing the
    figure exists to show.

    Inputs:
        internal_edges : set of (src_pkg, dst_pkg) inside pinthac
        external_edges : set of (src_pkg, external_module)
        out_path       : where to write the SVG
    Returns:
        None
    """
    import matplotlib.patches as mpatches

    stats = layer_stats()
    stack = ["properties", "correlations", "pin", "sca", "ml"]
    role = {"properties": "water, liquid metal and solid material properties",
            "correlations": "heat transfer, friction, rod-bundle factors",
            "pin": "gap, clad and fuel radial conduction",
            "sca": "single-channel solvers and the run driver",
            "ml": "DeepONet surrogate, PINN, training data"}

    fig, ax = style.figure(figsize=(10.2, 6.4))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(-0.5, 8.1)
    ax.axis("off")

    box_x, box_w, box_h = 0.5, 5.35, 0.82
    y0, dy = 1.55, 1.16

    total_lines = sum(v[2] for v in stats.values())
    ax.text(box_x, 7.72, "PINTHAC module architecture",
            ha="left", va="center", color=style.TEXT, fontsize=13, fontweight="bold")
    ax.text(box_x, 7.36,
            f"{sum(v[1] for v in stats.values())} modules, {total_lines:,} lines "
            "\u2014 imports run upward only",
            ha="left", va="center", color=style.TEXT_DIM, fontsize=8.5)

    for i, pkg in enumerate(stack):
        y = y0 + i * dy
        mods, nfile, nline = stats[pkg]
        ax.add_patch(mpatches.FancyBboxPatch(
            (box_x, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.3, edgecolor=style.ACCENT, facecolor=style.PANEL))
        ax.text(box_x + 0.26, y + 0.24, pkg, ha="left", va="center",
                color=style.ACCENT, fontsize=11.5, fontweight="bold")
        ax.text(box_x + box_w - 0.26, y + 0.24, f"{nline:,} lines",
                ha="right", va="center", color=style.TEXT_DIM, fontsize=8)
        ax.text(box_x + 0.26, y - 0.02, role[pkg], ha="left", va="center",
                color=style.TEXT, fontsize=8.2)
        ax.text(box_x + 0.26, y - 0.26, "  ".join(mods), ha="left", va="center",
                color=style.TEXT_DIM, fontsize=7.4, family="monospace")

        if i < len(stack) - 1:
            ax.annotate("", xy=(box_x + box_w / 2, y + dy - box_h / 2 - 0.02),
                        xytext=(box_x + box_w / 2, y + box_h / 2 + 0.02),
                        arrowprops=dict(arrowstyle="-|>", color=style.ACCENT,
                                         lw=1.4, alpha=0.9))

    mods, nfile, nline = stats["foundation"]
    found_h = 0.72
    ax.add_patch(mpatches.FancyBboxPatch(
        (box_x, 0.28), box_w, found_h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.3, edgecolor=style.ACCENT, facecolor=style.PANEL, linestyle="--"))
    ax.text(box_x + 0.26, 0.80, "  ".join(mods), ha="left", va="center",
            color=style.ACCENT, fontsize=8.6, family="monospace")
    ax.text(box_x + box_w - 0.26, 0.80, f"{nline:,} lines",
            ha="right", va="center", color=style.TEXT_DIM, fontsize=8)
    ax.text(box_x + 0.26, 0.50,
            "float / numpy / torch dispatch, range checks, Monte Carlo, root finding",
            ha="left", va="center", color=style.TEXT_DIM, fontsize=7.8)

    # External dependencies, against the layer that pulls each one in. Standard-library
    # imports are filtered out: they are not dependencies in any sense that affects how
    # this library is installed or run, and listing them crowds out the ones that are.
    STDLIB = {"math", "os", "time", "warnings", "functools", "ast", "json", "sys",
              "itertools", "collections", "typing", "pathlib", "random", "copy"}
    ext_by_pkg = {}
    for src, dst in external_edges:
        if dst not in STDLIB:
            ext_by_pkg.setdefault(src, set()).add(dst)

    ext_x = box_x + box_w + 0.45
    ax.text(ext_x, 7.72, "third-party dependencies", ha="left", va="center",
            color=style.MUTED, fontsize=8.5, fontweight="bold")
    for i, pkg in enumerate(stack):
        names = sorted(ext_by_pkg.get(pkg, set()))
        if names:
            # One line per layer, not a stacked column: stacking six names vertically
            # from a box's centre runs straight into the neighbouring layer's list.
            ax.text(ext_x, y0 + i * dy, "  ".join(names), ha="left", va="center",
                    color=style.MUTED, fontsize=7.6)
    found_ext = sorted(set().union(*[ext_by_pkg.get(p, set()) for p in mods] or [set()]))
    if found_ext:
        ax.text(ext_x, 0.64, "  ".join(found_ext), ha="left", va="center",
                color=style.MUTED, fontsize=7.6)

    style.finish(fig, out_path)


if __name__ == "__main__":
    main()
