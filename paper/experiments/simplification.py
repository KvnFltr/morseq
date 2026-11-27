"""Simplification of a noisy image by minimum-saddle cancellations (Sec. 7.3).

    python paper/experiments/simplification.py

A smooth image with six wells is quantized to 8 bits and corrupted by noise.
Minimum-saddle pairs are cancelled in increasing order of persistence, and
the stack is updated by raise_stack after each cancellation, so that the
field remains an F-sequence field. Writes results/simplification.json and
figures/simplification.pdf.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER.parent))

from morseq import (basin_sets, cubical_complex, cube_vertex,  # noqa: E402
                    flooding_order, gradient_from_vertex_map, is_acyclic,
                    is_flooding_sequence, is_stack, sequence_from_field)
from morseq.simplification import (is_level_flat, minimum_saddle_pairs,  # noqa: E402
                                   simplify_minima, vertex_flows)

N = 64
WELLS = [(14, 14, 70, 6), (16, 48, 60, 7), (32, 30, 55, 6),
         (48, 12, 65, 6), (50, 46, 75, 7), (30, 56, 50, 5)]
NOISE = 7
THRESHOLDS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40]
SHOW = 12


def make_image():
    rng = random.Random(2026)
    img = []
    for i in range(N):
        row = []
        for j in range(N):
            z = 200.0
            for (ci, cj, depth, r) in WELLS:
                z -= depth * math.exp(-((i - ci) ** 2 + (j - cj) ** 2) / (2 * r * r))
            z += rng.uniform(-NOISE, NOISE)
            row.append(max(0, min(255, int(round(z)))))
        img.append(row)
    return img


def labels_grid(K, V):
    flow = vertex_flows(K, V)
    roots = sorted(set(flow.values()))
    idx = {m: k for k, m in enumerate(roots)}
    return [[idx[flow[cube_vertex((i, j))]] for j in range(N)] for i in range(N)]


def minima(K, V):
    heads = set(V.values())
    return [v for v in K.cells_of_dim(0) if v not in V and v not in heads]


def count_minima(K, V):
    return len(minima(K, V))


def main():
    img = make_image()
    K = cubical_complex((N, N))
    f = {v: img[v[0][0]][v[1][0]] for v in K.cells_of_dim(0)}
    t0 = time.perf_counter()
    V, F = gradient_from_vertex_map(K, f)
    F0 = dict(F)
    curve, snapshots = [], {}
    total_steps = 0
    for h in THRESHOLDS:
        V, F, steps = simplify_minima(K, V, F, threshold=h)
        total_steps += steps
        curve.append({"threshold": h, "minima": count_minima(K, V)})
        if h in (0, SHOW):
            snapshots[h] = (labels_grid(K, V),
                            [(m[0][0], m[1][0]) for m in minima(K, V)])
        print(f"h={h:3d}: {curve[-1]['minima']} minima", flush=True)
    elapsed = time.perf_counter() - t0

    seq = sequence_from_field(K, V, key=flooding_order(F))
    checks = {
        "stack": is_stack(K, F),
        "raised_only": all(F[c] >= F0[c] for c in K.cells),
        "level_flat": is_level_flat(V, F),
        "acyclic": is_acyclic(K, V),
        "flooding": is_flooding_sequence(seq, F),
        "basins": len(basin_sets(seq)),
        "remaining_pairs_min_persistence": min(
            (p for p, _, _ in minimum_saddle_pairs(K, V, F)), default=None),
    }
    res = {"size": N, "wells": len(WELLS), "noise": NOISE, "shown": SHOW, "curve": curve,
           "cancellations": total_steps, "seconds": round(elapsed, 1),
           "checks": checks}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "simplification.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    figure(img, snapshots, curve)


def boundaries(lab):
    """Pixels whose right or lower neighbour carries another label."""
    import numpy as np
    a = np.asarray(lab)
    b = np.zeros(a.shape, dtype=bool)
    b[:, :-1] |= a[:, :-1] != a[:, 1:]
    b[:-1, :] |= a[:-1, :] != a[1:, :]
    return b


def figure(img, snapshots, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    import numpy as np
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    boundary = ListedColormap([(0, 0, 0, 0), "#eb6834"])
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.25),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, h, title in [(axes[0], 0, "(a) before simplification"),
                         (axes[1], SHOW, f"(b) after cancellations, $h={SHOW}$")]:
        labels, mins = snapshots[h]
        ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest")
        if h:  # before simplification nearly every pixel is a boundary
            ax.imshow(boundaries(labels), cmap=boundary, interpolation="nearest")
        ax.scatter([j for _, j in mins], [i for i, _ in mins], s=4 if h == 0 else 12,
                   color="#2a78d6", edgecolors="white", linewidths=0.3)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[2]
    xs = [c["threshold"] for c in curve]
    ys = [c["minima"] for c in curve]
    ax.plot(xs, ys, color="#2a78d6", linewidth=1.5, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("persistence threshold $h$")
    ax.set_ylabel("minima")
    ax.axhline(len(WELLS), color="#8a8984", linewidth=0.8, linestyle="--")
    ax.text(xs[-1], len(WELLS) * 1.15, f"{len(WELLS)} wells", ha="right",
            va="bottom", fontsize=7, color="#52514e")
    ax.set_title("(c) minima left", fontsize=8)
    ax.grid(True, which="major", color="#e6e5e1", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    (PAPER / "figures").mkdir(exist_ok=True)
    fig.savefig(PAPER / "figures" / "simplification.pdf")
    fig.savefig(HERE / "results" / "simplification.png", dpi=200)


if __name__ == "__main__":
    main()
