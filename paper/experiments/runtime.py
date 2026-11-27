"""Running time of the three stages of the pipeline (Section 7).

    python paper/experiments/runtime.py

Stages: lower-star gradient, flooding schedule of the sequence, one-pass
basin labeling. Random 8-bit images, pure Python. Each size runs in its own
process (so that memory from a previous size cannot slow it down), twice;
the best time of each stage is kept. Writes results/runtime.json and
results/runtime_table.tex.
"""
from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

CASES = [(64, 64), (128, 128), (256, 256), (512, 512),
         (16, 16, 16), (24, 24, 24), (32, 32, 32)]
REPEAT = 2


def run(shape):
    from morseq import (basins, cubical_complex, flooding_order,
                        gradient_from_vertex_map, sequence_from_field)
    rng = random.Random(0)
    K = cubical_complex(shape)
    f = {v: rng.randrange(256) for v in K.cells_of_dim(0)}
    t0 = time.perf_counter()
    V, F = gradient_from_vertex_map(K, f)
    t1 = time.perf_counter()
    seq = sequence_from_field(K, V, key=flooding_order(F))
    t2 = time.perf_counter()
    basins(seq)
    t3 = time.perf_counter()
    n_min = sum(1 for c in seq.critical if K.dim[c] == 0)
    return {"shape": list(shape), "cells": len(K), "minima": n_min,
            "gradient_s": t1 - t0, "sequence_s": t2 - t1, "basins_s": t3 - t2}


def measure(shape):
    best = None
    for _ in range(REPEAT):
        out = subprocess.run([sys.executable, __file__, "--one", "x".join(map(str, shape))],
                             capture_output=True, text=True, check=True)
        r = json.loads(out.stdout)
        if best is None:
            best = r
        else:
            for k in ("gradient_s", "sequence_s", "basins_s"):
                best[k] = min(best[k], r[k])
    return best


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "\\,")


def main():
    rows = []
    for shape in CASES:
        r = measure(shape)
        rows.append(r)
        tot = r["gradient_s"] + r["sequence_s"] + r["basins_s"]
        print(f"{'x'.join(map(str, shape)):>10} {r['cells']:>9} cells  "
              f"{r['gradient_s']:7.2f} {r['sequence_s']:7.2f} {r['basins_s']:6.2f}  "
              f"{1e6 * tot / r['cells']:5.1f} us/cell", flush=True)
    info = {"python": platform.python_version(), "repeat": REPEAT, "rows": rows}
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "runtime.json").write_text(json.dumps(info, indent=2))
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"grid & cells & minima & gradient & sequence & basins & $\mu$s/cell \\",
             r"\midrule"]
    for r in rows:
        tot = r["gradient_s"] + r["sequence_s"] + r["basins_s"]
        grid = "$" + r"\times".join(map(str, r["shape"])) + "$"
        lines.append(f"{grid} & {thousands(r['cells'])} & {thousands(r['minima'])} & "
                     f"{r['gradient_s']:.1f}\\,s & {r['sequence_s']:.1f}\\,s & "
                     f"{r['basins_s']:.1f}\\,s & {1e6 * tot / r['cells']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "runtime_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print(json.dumps(run(tuple(int(x) for x in sys.argv[2].split("x")))))
    else:
        main()
