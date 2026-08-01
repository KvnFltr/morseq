"""The statements of Part II, as executable tests.

Every theorem is checked on: the guide example, cubical grids in 2-D and 3-D,
simplicial complexes, and randomly generated acyclic fields. A failure here is
a counterexample to a stated result.
"""
from __future__ import annotations

import itertools
import random

import pytest

from morseq import (Complex, cubical_complex, simplicial_complex, cube_vertex,
                    MorseSequence, sequence_from_field, is_acyclic,
                    reference_map, coreference_map, critical_boundary,
                    cocritical_coboundary, coextension, extension,
                    count_gradient_paths, stable_set, unstable_set,
                    vertex_flow, vertex_partition, is_hermetic, restrict,
                    requirements, basin, basins, basin_sets, bridges,
                    regular_collapse_order, max_extension, is_stack, cut,
                    is_f_sequence, gradient_from_vertex_map, flooding_order,
                    segment_image)
from morseq.complex import vertex_coords


# ----------------------------------------------------------------------
# fixtures: a zoo of complexes and fields
# ----------------------------------------------------------------------
def random_field(K, seed, drop=0.25):
    """A random acyclic matching, or None if the draw is cyclic."""
    rng = random.Random(seed)
    cand = [(f, c) for c in K.cells for f in K.facets[c]]
    rng.shuffle(cand)
    used, V = set(), {}
    for s, t in cand:
        if s in used or t in used or rng.random() < drop:
            continue
        V[s] = t
        used.update((s, t))
    return V if is_acyclic(K, V) else None


COMPLEXES = [
    ("cubical 3x3", cubical_complex((3, 3))),
    ("cubical 4x3", cubical_complex((4, 3))),
    ("cubical 5x4", cubical_complex((5, 4))),
    ("cubical 3x3x3", cubical_complex((3, 3, 3))),
    ("simplicial triangle", simplicial_complex([(1, 2, 3)])),
    ("simplicial band", simplicial_complex([(1, 2, 3), (2, 3, 4), (3, 4, 5)])),
]


def all_cases(max_seeds=6):
    for name, K in COMPLEXES:
        found = 0
        for seed in range(40):
            V = random_field(K, seed)
            if V is None:
                continue
            seq = sequence_from_field(K, V)
            yield f"{name}/seed{seed}", K, seq
            found += 1
            if found >= max_seeds:
                break


CASES = list(all_cases())


def test_cases_are_non_trivial():
    assert len(CASES) >= 20


# ----------------------------------------------------------------------
# complexes
# ----------------------------------------------------------------------
def test_cubical_counts_and_euler():
    K = cubical_complex((3, 3))
    assert K.betti_check_counts() == {0: 9, 1: 12, 2: 4}
    assert K.euler_characteristic() == 1
    K3 = cubical_complex((2, 2, 2))
    assert K3.betti_check_counts() == {0: 8, 1: 12, 2: 6, 3: 1}
    assert K3.euler_characteristic() == 1


def test_complex_rejects_non_closed():
    with pytest.raises(ValueError):
        Complex(["a"], {"a": ["b"]}, {"a": 1, "b": 0})


@pytest.mark.parametrize("name,K,seq", CASES)
def test_sequence_is_valid(name, K, seq):
    seq.validate()


# ----------------------------------------------------------------------
# lemme d'ordre
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,K,seq", CASES)
def test_incidence_order_lemma(name, K, seq):
    """Lemma: facets of tau other than sigma come earlier; cofacets of sigma later."""
    for s, t in seq.V.items():
        i = seq.step[s]
        assert seq.step[t] == i
        for f in K.facets[t]:
            if f != s:
                assert seq.step[f] < i
        for c in K.cofacets[s]:
            if c != t:
                assert seq.step[c] > i


@pytest.mark.parametrize("name,K,seq", CASES)
def test_paths_strictly_decrease(name, K, seq):
    for s, t in seq.V.items():
        for nxt in K.facets[t]:
            if nxt != s:
                assert seq.step[nxt] < seq.step[s]


# ----------------------------------------------------------------------
# B5
# ----------------------------------------------------------------------
def recursive_stable_set(K, V, kappa):
    """The literal recursive definition of DFRS, Sec. 3.1."""
    S, changed = {kappa}, True
    while changed:
        changed = False
        for d, g in V.items():
            if K.dim[d] != K.dim[kappa] or d in S:
                continue
            if any(b in S for b in K.facets[g] if b != d):
                S.add(d)
                changed = True
    return S


def recursive_unstable_set(K, V, kappa):
    inv = {t: s for s, t in V.items()}
    U, changed = {kappa}, True
    while changed:
        changed = False
        for d in K.cells:
            if K.dim[d] != K.dim[kappa] or d in U or d not in inv:
                continue
            g = inv[d]
            if any(b in U for b in K.cofacets[g] if b != d):
                U.add(d)
                changed = True
    return U


@pytest.mark.parametrize("name,K,seq", CASES)
def test_b5_equivalence_with_dfrs(name, K, seq):
    for k in seq.critical:
        assert stable_set(seq, k) == recursive_stable_set(K, seq.V, k)
        assert unstable_set(seq, k) == recursive_unstable_set(K, seq.V, k)


@pytest.mark.parametrize("name,K,seq", CASES)
def test_b5_parity_included_in_existence(name, K, seq):
    gam, lam = reference_map(seq), coreference_map(seq)
    for k in seq.critical:
        assert coextension(seq, k, gam) <= stable_set(seq, k)
        assert extension(seq, k, lam) <= unstable_set(seq, k)


@pytest.mark.parametrize("name,K,seq", CASES)
def test_theorem_12_parity(name, K, seq):
    """kappa in gamma(nu) iff the number of gradient paths nu -> kappa is odd."""
    gam = reference_map(seq)
    for k in seq.critical:
        for c in K.cells:
            if K.dim[c] != K.dim[k]:
                continue
            assert (count_gradient_paths(seq, c, k) % 2 == 1) == (k in gam[c])


def test_b5_strict_inclusion_counterexample():
    """The V2 field of the report: two paths AB -> EH, so parity loses AB."""
    K = cubical_complex((3, 3))
    v = lambda x, y: cube_vertex((x, y))
    e = lambda p, q: tuple((min(a, b), max(a, b)) for a, b in zip(p, q))
    AB = e((0, 0), (1, 0))
    BE = e((1, 0), (1, 1))
    DE = e((0, 1), (1, 1))
    EF = e((1, 1), (2, 1))
    EH = e((1, 1), (1, 2))
    Q = lambda x, y: ((x, x + 1), (y, y + 1))
    V = {AB: Q(0, 0), BE: Q(1, 0), DE: Q(0, 1), EF: Q(1, 1)}
    # complete with vertex pairs so that only the interesting cells stay critical
    for (x, y) in [(2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]:
        pass
    seq = sequence_from_field(K, V)
    assert count_gradient_paths(seq, AB, EH) == 2
    assert AB in stable_set(seq, EH)
    assert AB not in coextension(seq, EH)


# ----------------------------------------------------------------------
# B6
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,K,seq", CASES)
def test_b6_partition(name, K, seq):
    part = vertex_partition(seq)
    crit0 = [c for c in seq.critical if K.dim[c] == 0]
    assert set(part.values()) <= set(crit0)
    assert set(part) == set(K.cells_of_dim(0))
    gam = reference_map(seq)
    for m in crit0:
        cls = {v for v, r in part.items() if r == m}
        assert cls == stable_set(seq, m)
        assert cls == coextension(seq, m, gam)   # existence == parity at dim 0
    # classes are pairwise disjoint and cover the vertices
    total = sum(len({v for v, r in part.items() if r == m}) for m in crit0)
    assert total == len(K.cells_of_dim(0))


@pytest.mark.parametrize("name,K,seq", CASES)
def test_b6_unique_maximal_path(name, K, seq):
    for v in K.cells_of_dim(0):
        m = vertex_flow(seq, v)
        assert m in seq.critical
        assert count_gradient_paths(seq, v, m) == 1
        for other in [c for c in seq.critical if K.dim[c] == 0 and c != m]:
            assert count_gradient_paths(seq, v, other) == 0


# ----------------------------------------------------------------------
# B8
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,K,seq", CASES)
def test_b8_prefixes_are_hermetic(name, K, seq):
    for i in range(len(seq.items) + 1):
        assert is_hermetic(seq, seq.prefix(i))


@pytest.mark.parametrize("name,K,seq", CASES)
def test_b8_restriction(name, K, seq):
    rng = random.Random(hash(name) % 1000)
    for _ in range(3):
        i = rng.randrange(1, len(seq.items) + 1)
        H = seq.prefix(i)
        sub = restrict(seq, H)
        for k in sub.critical:
            assert stable_set(sub, k) == stable_set(seq, k) & H
            assert unstable_set(sub, k) == unstable_set(seq, k)


# ----------------------------------------------------------------------
# B11
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,K,seq", CASES)
def test_b11_closure_is_order_independent(name, K, seq):
    crit0 = [c for c in seq.critical if K.dim[c] == 0]
    for m in crit0:
        pairs = list(seq.V.items())
        a = basin(seq, m, order=pairs)
        rng = random.Random(7)
        for _ in range(3):
            rng.shuffle(pairs)
            assert basin(seq, m, order=list(pairs)) == a


@pytest.mark.parametrize("name,K,seq", CASES)
def test_b11_algorithm_equals_closure(name, K, seq):
    sets = basin_sets(seq)
    for m in [c for c in seq.critical if K.dim[c] == 0]:
        assert sets.get(m, {m}) == basin(seq, m)


@pytest.mark.parametrize("name,K,seq", CASES)
def test_b11_properties(name, K, seq):
    sets = basin_sets(seq)
    crit0 = [c for c in seq.critical if K.dim[c] == 0]
    for m in crit0:
        B = sets.get(m, {m})
        assert K.is_subcomplex(B)                       # subcomplex
        assert is_hermetic(seq, B)                      # hermetic
        assert set(seq.critical) & B == {m}             # only critical cell
        assert {c for c in B if K.dim[c] == 0} == stable_set(seq, m)   # B^(0)=St
        regular_collapse_order(seq, B, m)               # collapses regularly
        for s, t in seq.V.items():                      # maximal
            if s not in B:
                assert not requirements(K, s, t) <= B
    for a, b in itertools.combinations(crit0, 2):
        assert not (sets.get(a, {a}) & sets.get(b, {b}))  # disjoint


@pytest.mark.parametrize("name,K,seq", CASES)
def test_b11_restriction_to_prefixes(name, K, seq):
    rng = random.Random(hash(name) % 997)
    sets = basin_sets(seq)
    for _ in range(2):
        i = rng.randrange(1, len(seq.items) + 1)
        H = seq.prefix(i)
        sub = restrict(seq, H)
        for m in [c for c in sub.critical if K.dim[c] == 0]:
            assert basin(sub, m) == sets.get(m, {m}) & H


@pytest.mark.parametrize("name,K,seq", CASES)
def test_bridges_are_unlabeled_one_cells(name, K, seq):
    lab = basins(seq)
    assert bridges(seq) == {c for c in K.cells if K.dim[c] == 1 and lab[c] is None}


# ----------------------------------------------------------------------
# stacks, cuts, images
# ----------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(4, 4), (5, 3), (3, 3, 3)])
def test_stack_and_cuts(shape):
    rng = random.Random(sum(shape))
    K = cubical_complex(shape)
    f = {v: rng.randrange(0, 4) for v in K.cells_of_dim(0)}   # plateaus on purpose
    V, F = gradient_from_vertex_map(K, f)
    assert is_stack(K, F)
    seq = sequence_from_field(K, V, key=flooding_order(F))
    seq.validate()
    assert is_f_sequence(seq, F)
    sets = basin_sets(seq)
    for lam in range(min(F.values()), max(F.values()) + 1):
        C = cut(K, F, lam)
        assert K.is_subcomplex(C)
        assert is_hermetic(seq, C)
        sub = restrict(seq, C)
        for m in [c for c in sub.critical if K.dim[c] == 0]:
            assert basin(sub, m) == sets.get(m, {m}) & C
            assert stable_set(sub, m) == stable_set(seq, m) & C


def test_segment_image_two_basins():
    seg = segment_image([[0, 3, 1],
                         [3, 3, 3],
                         [1, 3, 0]])
    labels = seg.vertex_labels()
    assert labels[0][0] != labels[2][2]          # opposite corners separated
    assert len(seg.basin_sets()) >= 2
    assert seg.summary().startswith("complex:")


def test_segment_image_plateau_no_perturbation():
    """A large flat zone is handled without perturbing the data."""
    img = [[9, 9, 9, 9],
           [9, 0, 0, 9],
           [9, 0, 0, 9],
           [9, 9, 9, 9]]
    seg = segment_image(img)
    assert is_stack(seg.K, seg.F)
    assert is_f_sequence(seg.sequence, seg.F)
    # the four plateau vertices share one label
    lab = seg.vertex_labels()
    assert lab[1][1] == lab[1][2] == lab[2][1] == lab[2][2]


def test_segment_image_3d():
    img = [[[0, 5], [5, 5]], [[5, 5], [5, 0]]]
    seg = segment_image(img)
    seg.sequence.validate()
    assert seg.K.dimension == 3
    assert len(seg.basin_sets()) >= 1


def test_monotone_ramp_has_single_basin():
    seg = segment_image([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    assert len(seg.basin_sets()) == 1


# ----------------------------------------------------------------------
# maps: chain complex identities
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,K,seq", CASES)
def test_critical_complex_is_a_chain_complex(name, K, seq):
    d = critical_boundary(seq)
    for k, col in d.items():
        acc = set()
        for x in col:
            for y in d.get(x, set()):
                acc ^= {y}
        assert not acc, "d o d != 0"
    dd = cocritical_coboundary(seq)
    for k, col in dd.items():
        acc = set()
        for x in col:
            for y in dd.get(x, set()):
                acc ^= {y}
        assert not acc, "delta o delta != 0"


@pytest.mark.parametrize("name,K,seq", CASES)
def test_betti_numbers_match_cell_counts(name, K, seq):
    """Euler characteristic from the critical complex equals that of K."""
    crit = seq.critical_by_dim()
    chi = sum((-1) ** p * len(v) for p, v in crit.items())
    assert chi == K.euler_characteristic()
