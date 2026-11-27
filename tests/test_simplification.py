"""Cancellation of critical pairs, compatible stacks, bridges.

These tests check the statements of the paper that go beyond the
translation of DFRS: after a Forman cancellation, the field is level-flat
for the raised and for the lowered stack; cancelling a minimum with a saddle
merges two basins; bridges are the stable sets of the critical edges.
"""
from __future__ import annotations

import random

import pytest

from morseq import (basin_sets, basins, cubical_complex, flooding_order,
                    gradient_from_vertex_map, is_acyclic, is_flooding_sequence,
                    is_stack, is_valid_field, random_morse_sequence,
                    sequence_from_field, simplicial_complex, stable_set)
from morseq.simplification import (cancel, gradient_path_count,
                                   is_level_flat, lower_stack,
                                   minimum_saddle_pairs, raise_stack,
                                   simplify_minima, vertex_flows)


def image_field(shape, levels, seed):
    rng = random.Random(seed)
    K = cubical_complex(shape)
    f = {v: rng.randrange(levels) for v in K.cells_of_dim(0)}
    V, F = gradient_from_vertex_map(K, f)
    return K, V, F


def critical_cells(K, V):
    heads = set(V.values())
    return [c for c in K.cells if c not in V and c not in heads]


def cancellable_pairs(K, V):
    crit = critical_cells(K, V)
    for beta in crit:
        for alpha in crit:
            if K.dim[beta] == K.dim[alpha] + 1 and \
                    gradient_path_count(K, V, beta, alpha) == 1:
                yield alpha, beta


IMAGES = [((5, 5), 3, s) for s in range(12)] + \
         [((6, 7), 256, s) for s in range(8)] + \
         [((3, 3, 3), 2, s) for s in range(6)] + \
         [((4, 3, 3), 4, s) for s in range(4)]


@pytest.mark.parametrize("shape,levels,seed", IMAGES)
def test_cancellation_is_level_flat_for_raised_and_lowered_stacks(shape, levels, seed):
    K, V, F = image_field(shape, levels, seed)
    for alpha, beta in cancellable_pairs(K, V):
        W = cancel(K, V, alpha, beta)
        assert is_valid_field(K, W) and is_acyclic(K, W)
        for G in (raise_stack(K, F, alpha, F[beta]),
                  lower_stack(K, F, beta, F[alpha])):
            assert is_stack(K, G)
            assert is_level_flat(W, G)
            seq = sequence_from_field(K, W, key=flooding_order(G))
            seq.validate()
            assert is_flooding_sequence(seq, G)


def test_original_stack_is_not_enough():
    """Without modifying the stack, flatness fails for some cancellation."""
    failures = 0
    for shape, levels, seed in IMAGES:
        K, V, F = image_field(shape, levels, seed)
        for alpha, beta in cancellable_pairs(K, V):
            if not is_level_flat(cancel(K, V, alpha, beta), F):
                failures += 1
    assert failures > 0


def test_raised_stack_changes_only_one_component():
    K, V, F = image_field((7, 7), 5, 3)
    for alpha, beta in cancellable_pairs(K, V):
        G = raise_stack(K, F, alpha, F[beta])
        changed = {c for c in K.cells if G[c] != F[c]}
        assert all(F[c] < F[beta] and G[c] == F[beta] for c in changed)
        assert all(G[c] >= F[c] for c in K.cells)
        H = lower_stack(K, F, beta, F[alpha])
        assert all(H[c] <= F[c] for c in K.cells)


@pytest.mark.parametrize("shape,levels,seed", IMAGES)
def test_cancelling_a_minimum_merges_two_basins(shape, levels, seed):
    K, V, F = image_field(shape, levels, seed)
    seq = sequence_from_field(K, V, key=flooding_order(F))
    before = basin_sets(seq)
    flow = vertex_flows(K, V)
    for _, m, s in minimum_saddle_pairs(K, V, F):
        a, b = K.facets[s]
        m2 = flow[b] if flow[a] == m else flow[a]
        W = cancel(K, V, m, s)
        G = raise_stack(K, F, m, F[s])
        after = basin_sets(sequence_from_field(K, W, key=flooding_order(G)))
        assert m not in after
        assert after[m2] >= before.get(m, {m}) | before.get(m2, {m2}) | {s}
        verts = lambda X: {c for c in X if K.dim[c] == 0}
        assert verts(after[m2]) == verts(before.get(m, {m})) | verts(before.get(m2, {m2}))
        for m3, B in before.items():
            if m3 not in (m, m2):
                assert after[m3] == B


def test_simplify_minima_removes_low_persistence_pairs():
    K, V, F = image_field((12, 12), 50, 1)
    n0 = len([c for c in critical_cells(K, V) if K.dim[c] == 0])
    W, G, steps = simplify_minima(K, V, F, threshold=10)
    assert steps > 0
    assert is_stack(K, G) and is_level_flat(W, G) and is_acyclic(K, W)
    assert all(G[c] >= F[c] for c in K.cells)
    assert all(p >= 10 for p, _, _ in minimum_saddle_pairs(K, W, G))
    n1 = len([c for c in critical_cells(K, W) if K.dim[c] == 0])
    assert n1 == n0 - steps


def test_paths_from_boundary_matches_pairwise_counts():
    from morseq.simplification import paths_from_boundary
    for shape, levels, seed in IMAGES[::3]:
        K, V, F = image_field(shape, levels, seed)
        crit = critical_cells(K, V)
        for beta in crit:
            if K.dim[beta] == 0:
                continue
            counts = paths_from_boundary(K, V, beta)
            for alpha in crit:
                if K.dim[alpha] == K.dim[beta] - 1:
                    assert counts.get(alpha, 0) == gradient_path_count(K, V, beta, alpha)


def test_cancel_rejects_invalid_pairs():
    K, V, F = image_field((5, 5), 3, 0)
    crit = critical_cells(K, V)
    paired = next(iter(V))
    with pytest.raises(ValueError):
        cancel(K, V, paired, next(c for c in crit if K.dim[c] == 1))
    mins = [c for c in crit if K.dim[c] == 0]
    with pytest.raises(ValueError):
        cancel(K, V, mins[0], mins[-1])


# ----------------------------------------------------------------------
# bridges
# ----------------------------------------------------------------------
def bridge_cases():
    for shape, levels, seed in IMAGES:
        K, V, F = image_field(shape, levels, seed)
        yield f"image{shape}/{seed}", sequence_from_field(K, V, key=flooding_order(F))
    for name, K in [("cubical 5x4", cubical_complex((5, 4))),
                    ("cubical 3x3x3", cubical_complex((3, 3, 3))),
                    ("simplicial band", simplicial_complex(
                        [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]))]:
        for seed in range(5):
            yield f"{name}/{seed}", random_morse_sequence(K, seed=seed)


@pytest.mark.parametrize("name,seq", list(bridge_cases()))
def test_bridges_are_stable_sets_of_critical_edges(name, seq):
    K = seq.K
    lab = basins(seq)
    bridges = {c for c in K.cells if K.dim[c] == 1 and lab[c] is None}
    saddles = [c for c in seq.critical if K.dim[c] == 1]
    union = set()
    for s in saddles:
        union |= stable_set(seq, s)
    assert bridges == union
    # vertex-edge pairs always lie in a basin
    for v, e in seq.V.items():
        if K.dim[v] == 0:
            assert lab[v] is not None and lab[e] == lab[v]
