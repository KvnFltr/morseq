"""Large-scale checks of the statements of the paper (Section 7.1).

    python paper/experiments/verification.py      # about 10 minutes

Writes paper/experiments/results/verification.json. Every count reported in
the paper comes from this file.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from morseq import (basin, basin_sets, basins, cubical_complex,  # noqa: E402
                    flooding_order, gradient_from_vertex_map, is_acyclic,
                    is_flooding_sequence, is_stack, random_morse_sequence,
                    sequence_from_field, simplicial_complex, stable_set,
                    coextension, reference_map)
from morseq.simplification import (cancel, gradient_path_count,  # noqa: E402
                                   is_level_flat, lower_stack,
                                   paths_from_boundary,
                                   minimum_saddle_pairs, raise_stack,
                                   vertex_flows)
from test_rws_equivalence import process_lower_stars  # noqa: E402


def image(shape, levels, seed):
    rng = random.Random(seed)
    K = cubical_complex(shape)
    f = {v: rng.randrange(levels) for v in K.cells_of_dim(0)}
    return K, f


IMAGE_SETS = [((8, 8), 4, 300), ((10, 12), 256, 200), ((16, 16), 8, 100),
              ((4, 4, 4), 3, 150), ((5, 5, 4), 256, 80)]


def check_rws():
    out = {"images": 0, "cells": 0, "differences": 0}
    for shape, levels, n in IMAGE_SETS:
        for seed in range(n):
            K, f = image(shape, levels, seed)
            tie = {v: i for i, v in enumerate(K.cells_of_dim(0))}
            V, _ = gradient_from_vertex_map(K, f, tie)
            W, _ = process_lower_stars(K, f, tie)
            out["images"] += 1
            out["cells"] += len(K)
            out["differences"] += int(V != W)
    return out


def critical(K, V):
    heads = set(V.values())
    return [c for c in K.cells if c not in V and c not in heads]


def check_cancellation():
    out = {"stacks": 0, "pairs": 0, "not_flat_for_F": 0, "flat_raised": 0,
           "flat_lowered": 0, "flooding": 0, "acyclic": 0,
           "min_saddle_merges": 0, "merge_ok": 0}
    for shape, levels, n in IMAGE_SETS:
        for seed in range(min(n, 60)):
            K, f = image(shape, levels, seed)
            V, F = gradient_from_vertex_map(K, f)
            out["stacks"] += 1
            crit = critical(K, V)
            crit_set = set(crit)
            for beta in crit:
                if K.dim[beta] == 0:
                    continue
                counts = paths_from_boundary(K, V, beta)
                for alpha in (c for c, n in counts.items()
                              if n == 1 and c in crit_set):
                    assert gradient_path_count(K, V, beta, alpha) == 1
                    W = cancel(K, V, alpha, beta)
                    out["pairs"] += 1
                    out["acyclic"] += is_acyclic(K, W)
                    out["not_flat_for_F"] += not is_level_flat(W, F)
                    G1 = raise_stack(K, F, alpha, F[beta])
                    G2 = lower_stack(K, F, beta, F[alpha])
                    ok1 = is_stack(K, G1) and is_level_flat(W, G1)
                    ok2 = is_stack(K, G2) and is_level_flat(W, G2)
                    out["flat_raised"] += ok1
                    out["flat_lowered"] += ok2
                    seq = sequence_from_field(K, W, key=flooding_order(G1))
                    out["flooding"] += is_flooding_sequence(seq, G1)
            # basin merging for minimum-saddle cancellations
            before = basin_sets(sequence_from_field(K, V, key=flooding_order(F)))
            flow = vertex_flows(K, V)
            for _, m, s in minimum_saddle_pairs(K, V, F):
                a, b = K.facets[s]
                m2 = flow[b] if flow[a] == m else flow[a]
                W = cancel(K, V, m, s)
                G = raise_stack(K, F, m, F[s])
                after = basin_sets(sequence_from_field(K, W, key=flooding_order(G)))
                out["min_saddle_merges"] += 1
                ok = after[m2] >= before.get(m, {m}) | before.get(m2, {m2}) | {s}
                ok &= all(after[x] == B for x, B in before.items() if x not in (m, m2))
                out["merge_ok"] += ok
    return out


def sequences():
    complexes = [cubical_complex((6, 5)), cubical_complex((4, 4, 3)),
                 simplicial_complex([(i, i + 1, i + 2) for i in range(8)])]
    for K in complexes:
        for seed in range(40):
            yield K, random_morse_sequence(K, seed=seed)
    for shape, levels, n in IMAGE_SETS:
        for seed in range(20):
            K, f = image(shape, levels, seed)
            V, F = gradient_from_vertex_map(K, f)
            yield K, sequence_from_field(K, V, key=flooding_order(F))


def check_segmentation():
    out = {"sequences": 0, "cells": 0, "closure_equals_scan": 0,
           "bridges_equal_saddle_stable_sets": 0, "existence_strictly_larger": 0}
    for K, seq in sequences():
        out["sequences"] += 1
        out["cells"] += len(K)
        lab = basins(seq)
        sets = basin_sets(seq)
        mins = [c for c in seq.critical if K.dim[c] == 0]
        out["closure_equals_scan"] += all(basin(seq, m) == sets.get(m, {m}) for m in mins)
        br = {c for c in K.cells if K.dim[c] == 1 and lab[c] is None}
        union = set()
        for s in (c for c in seq.critical if K.dim[c] == 1):
            union |= stable_set(seq, s)
        out["bridges_equal_saddle_stable_sets"] += br == union
        gam = reference_map(seq)
        strict = any(len(stable_set(seq, k)) > len(coextension(seq, k, gam))
                     for k in seq.critical if K.dim[k] >= 1)
        out["existence_strictly_larger"] += strict
    return out


def main():
    t0 = time.perf_counter()
    res = {"rws": check_rws()}
    print("rws", res["rws"], flush=True)
    res["cancellation"] = check_cancellation()
    print("cancellation", res["cancellation"], flush=True)
    res["segmentation"] = check_segmentation()
    print("segmentation", res["segmentation"], flush=True)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "verification.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
