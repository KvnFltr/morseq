"""morseq's lower-star gradient against Algorithm 1 of Robins, Wood, Sheppard.

``process_lower_stars`` below is a line-by-line transcription of
Algorithm 1 (ProcessLowerStars) of V. Robins, P. J. Wood, A. P. Sheppard,
*Theory and algorithms for constructing discrete Morse complexes from
grayscale digital images*, IEEE TPAMI 33(8):1646-1658, 2011. Line numbers in
the comments refer to that algorithm. It counts unclassified *faces* of all
dimensions and updates the queues only around the cells just classified, as
in the paper; :func:`morseq.gradient_from_vertex_map` counts facets and
rescans the lower star. The tests check that both return the same field.
"""
from __future__ import annotations

import heapq
import itertools
import random

import pytest

from morseq import cubical_complex, simplicial_complex, gradient_from_vertex_map


def _faces(K, c, memo):
    if c not in memo:
        out = set()
        for f in K.facets[c]:
            out.add(f)
            out |= _faces(K, f, memo)
        memo[c] = out
    return memo[c]


def _cofaces(K, c, memo):
    if c not in memo:
        out = set()
        for g in K.cofacets[c]:
            out.add(g)
            out |= _cofaces(K, g, memo)
        memo[c] = out
    return memo[c]


def process_lower_stars(K, g, tie):
    """Algorithm 1 of RWS 2011. Returns ``(V, C)``."""
    order = lambda v: (g[v], tie[v])  # distinct values, as after Eq. (8)
    fm, cm = {}, {}
    verts = {c: [x for x in _faces(K, c, fm) | {c} if K.dim[x] == 0]
             for c in K.cells}
    # G(sigma): vertex values in decreasing order, compared lexicographically
    G = {c: tuple(sorted((order(x) for x in verts[c]), reverse=True))
         for c in K.cells}
    lower_star = {}
    for c in K.cells:
        lower_star.setdefault(max(verts[c], key=order), set()).add(c)

    V, C = {}, set()
    for x in K.cells_of_dim(0):                                  # line 1
        L = lower_star[x]
        if L == {x}:                                             # line 2
            C.add(x)                                             # line 3
            continue
        classified = set()  # cells already inserted in V or C

        def num_unpaired_faces(a):
            return [b for b in _faces(K, a, fm) if b in L and b not in classified]

        tick = itertools.count()
        PQone, PQzero = [], []
        delta = min((c for c in L if K.dim[c] == 1), key=G.get)  # line 5
        V[x] = delta                                             # line 6
        classified |= {x, delta}
        for c in L:                                              # line 7
            if K.dim[c] == 1 and c != delta:
                heapq.heappush(PQzero, (G[c], next(tick), c))
        for a in _cofaces(K, delta, cm):                         # line 8
            if a in L and len(num_unpaired_faces(a)) == 1:
                heapq.heappush(PQone, (G[a], next(tick), a))
        while PQone or PQzero:                                   # line 9
            while PQone:                                         # line 10
                _, _, a = heapq.heappop(PQone)                   # line 11
                if a in classified:
                    continue
                unpaired = num_unpaired_faces(a)
                if not unpaired:                                 # line 12
                    heapq.heappush(PQzero, (G[a], next(tick), a))  # line 13
                    continue
                assert len(unpaired) == 1
                p = unpaired[0]
                assert K.dim[p] == K.dim[a] - 1, "pair(alpha) is not a facet"
                V[p] = a                                         # line 15
                classified |= {p, a}
                PQzero = [e for e in PQzero if e[2] != p]        # line 16
                heapq.heapify(PQzero)
                for b in _cofaces(K, a, cm) | _cofaces(K, p, cm):  # line 17
                    if b in L and b not in classified \
                            and len(num_unpaired_faces(b)) == 1:
                        heapq.heappush(PQone, (G[b], next(tick), b))
            while PQzero and PQzero[0][2] in classified:
                heapq.heappop(PQzero)
            if PQzero:                                           # line 20
                _, _, gam = heapq.heappop(PQzero)                # line 21
                C.add(gam)                                       # line 22
                classified.add(gam)
                for a in _cofaces(K, gam, cm):                   # line 23
                    if a in L and a not in classified \
                            and len(num_unpaired_faces(a)) == 1:
                        heapq.heappush(PQone, (G[a], next(tick), a))
        assert classified == L, "RWS Proposition 4"
    return V, C


CUBICAL = [((5, 5), 3, 60), ((6, 7), 256, 40), ((8, 8), 4, 30),
           ((3, 3, 3), 2, 30), ((4, 4, 3), 5, 20), ((4, 4, 4), 256, 8)]


@pytest.mark.parametrize("shape,levels,trials", CUBICAL)
def test_same_field_as_rws_on_cubical_images(shape, levels, trials):
    K = cubical_complex(shape)
    vs = K.cells_of_dim(0)
    tie = {v: i for i, v in enumerate(vs)}
    for seed in range(trials):
        rng = random.Random(seed)
        f = {v: rng.randrange(levels) for v in vs}
        V, F = gradient_from_vertex_map(K, f, tie)
        V_rws, C_rws = process_lower_stars(K, f, tie)
        assert V == V_rws, f"shape={shape} levels={levels} seed={seed}"
        paired = set(V) | set(V.values())
        assert {c for c in K.cells if c not in paired} == C_rws


def test_same_field_as_rws_on_simplicial_complex():
    # a triangulated 4x4 grid
    faces = []
    for i in range(3):
        for j in range(3):
            a, b, c, d = 4 * i + j, 4 * i + j + 1, 4 * (i + 1) + j, 4 * (i + 1) + j + 1
            faces += [(a, b, d), (a, c, d)]
    K = simplicial_complex(faces)
    vs = K.cells_of_dim(0)
    tie = {v: i for i, v in enumerate(vs)}
    for seed in range(40):
        rng = random.Random(seed)
        f = {v: rng.randrange(3) for v in vs}
        V, _ = gradient_from_vertex_map(K, f, tie)
        V_rws, _ = process_lower_stars(K, f, tie)
        assert V == V_rws, f"seed={seed}"
