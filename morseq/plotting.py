"""Optional plotting helpers (require matplotlib).

The figures follow the conventions of the report: blue arrows for
vertex-edge pairs, orange arrows for edge-square pairs, red for critical
cells, one colour per basin.
"""
from __future__ import annotations

from typing import Dict, Optional

from .complex import Complex, vertex_coords
from .sequence import MorseSequence

BLUE, ORANGE, RED, GRAY = "#1f66b4", "#e07b00", "#cc1f1f", "#666666"
PALETTE = ["#5b9bd5", "#e6a23c", "#70ad47", "#b07aa1", "#4dc9c0", "#d98880"]


def _mid(K: Complex, c):
    vs = [vertex_coords(v) for v in K.closure([c]) if K.dim[v] == 0]
    return tuple(sum(x[i] for x in vs) / len(vs) for i in range(len(vs[0])))


def plot_field(seq: MorseSequence, path: Optional[str] = None, ax=None,
               title: str = "gradient vector field"):
    """Draw a 2-D cubical gradient field with its critical cells."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    K = seq.K
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    verts = [vertex_coords(v) for v in K.cells_of_dim(0)]
    nx = max(v[0] for v in verts)
    ny = max(v[1] for v in verts)
    for x in range(nx + 1):
        for y in range(ny + 1):
            if x < nx:
                ax.plot([x, x + 1], [y, y], color=GRAY, lw=1, zorder=2)
            if y < ny:
                ax.plot([x, x], [y, y + 1], color=GRAY, lw=1, zorder=2)
    for v in verts:
        ax.plot(v[0], v[1], "o", color="k", ms=3, zorder=5)
    for c in seq.critical:
        p = _mid(K, c)
        if K.dim[c] == 2:
            ax.add_patch(Rectangle((p[0] - .5, p[1] - .5), 1, 1,
                                   facecolor="#f6c8c8", edgecolor=RED, lw=1.6, zorder=1))
        elif K.dim[c] == 1:
            vs = [vertex_coords(v) for v in K.facets[c]]
            ax.plot([vs[0][0], vs[1][0]], [vs[0][1], vs[1][1]], color=RED, lw=3.5, zorder=4)
        else:
            ax.plot(p[0], p[1], "o", color=RED, ms=9, mfc="none", mew=2, zorder=8)
            ax.plot(p[0], p[1], "o", color=RED, ms=4, zorder=8)
    for s, t in seq.V.items():
        a, b = _mid(K, s), _mid(K, t)
        col = BLUE if K.dim[s] == 0 else ORANGE
        ax.annotate("", xytext=a,
                    xy=(a[0] + 1.2 * (b[0] - a[0]), a[1] + 1.2 * (b[1] - a[1])),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7), zorder=7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    if path:
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
    return ax


def plot_segmentation(seg, path: Optional[str] = None):
    """Two panels: the gradient field, and the basins with their labels."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .segmentation import basin_sets, vertex_partition

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_field(seg.sequence, ax=axes[0], title="(a) gradient vector field")
    plot_field(seg.sequence, ax=axes[1], title="(b) basins")
    part = vertex_partition(seg.sequence)
    roots = sorted({m for m in part.values()}, key=str)
    colour = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(roots)}
    for v, m in part.items():
        p = vertex_coords(v)
        axes[1].plot(p[0], p[1], "o", color=colour[m], ms=10, alpha=.9, zorder=4)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=180, bbox_inches="tight")
    return fig
