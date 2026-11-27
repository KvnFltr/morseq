"""Cancellation of critical pairs, and stacks that remain compatible with it.

Forman's cancellation reverses the unique gradient path between two critical
cells and returns another gradient field. When the field comes from an
F-sequence, the reversed pairs usually join cells of different values, so the
new field is no longer level-flat for F. This module builds two explicit
stacks for which it is:

* :func:`raise_stack` -- the connected component of ``{F < F(beta)}`` that
  contains ``alpha`` is raised to ``F(beta)`` (for a minimum and a saddle,
  this fills the "lake" of the minimum up to the saddle);
* :func:`lower_stack` -- the connected component of ``{F > F(alpha)}`` that
  contains ``beta`` is lowered to ``F(alpha)`` (for a saddle and a maximum,
  this cuts the "dome" of the maximum down to the saddle).

Connectivity is taken in the incidence graph (a cell is adjacent to its
facets and cofacets).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .complex import Cell, Complex


# ----------------------------------------------------------------------
# gradient paths on a field
# ----------------------------------------------------------------------
def _path_counts(K: Complex, V: Dict[Cell, Cell], alpha: Cell,
                 sources: Sequence[Cell]) -> Dict[Cell, int]:
    """Number of gradient paths from each visited cell to ``alpha``.

    Iterative depth-first search with memoization: gradient paths can be
    long on images, so recursion is avoided.
    """
    count: Dict[Cell, int] = {}
    for src in sources:
        if src in count:
            continue
        stack: List[Tuple[Cell, bool]] = [(src, False)]
        while stack:
            cell, expanded = stack.pop()
            if cell in count:
                continue
            nxt = ([f for f in K.facets[V[cell]] if f != cell]
                   if cell in V else [])
            if not expanded:
                stack.append((cell, True))
                stack.extend((f, False) for f in nxt if f not in count)
            else:
                count[cell] = (1 if cell == alpha else 0) + sum(count[f] for f in nxt)
    return count


def paths_from_boundary(K: Complex, V: Dict[Cell, Cell], beta: Cell
                        ) -> Dict[Cell, int]:
    """Number of gradient paths from the boundary of ``beta`` to every cell.

    One pass over the cells reachable from the facets of ``beta``, in
    topological order of the (acyclic) path graph. The critical cells
    ``alpha`` with count 1 are those that can be cancelled with ``beta``.
    """
    def succ(c: Cell) -> List[Cell]:
        return [f for f in K.facets[V[c]] if f != c] if c in V else []

    starts = list(K.facets[beta])
    order: List[Cell] = []
    seen: set = set()
    for s in starts:
        if s in seen:
            continue
        seen.add(s)
        stack = [(s, iter(succ(s)))]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                order.append(node)
                stack.pop()
            elif nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(succ(nxt))))
    count = {c: 0 for c in seen}
    for s in starts:
        count[s] += 1
    for c in reversed(order):  # every cell before the cells it leads to
        if count[c]:
            for f in succ(c):
                count[f] += count[c]
    return count


def gradient_path_count(K: Complex, V: Dict[Cell, Cell], beta: Cell,
                        alpha: Cell) -> int:
    """Number of gradient paths from the boundary of ``beta`` to ``alpha``."""
    facets = list(K.facets[beta])
    count = _path_counts(K, V, alpha, facets)
    return sum(count[f] for f in facets)


def cancel(K: Complex, V: Dict[Cell, Cell], alpha: Cell, beta: Cell
           ) -> Dict[Cell, Cell]:
    """Forman cancellation of the critical pair ``(alpha, beta)``.

    ``alpha`` and ``beta`` must be critical, ``dim beta = dim alpha + 1``,
    and exactly one gradient path must join a facet of ``beta`` to
    ``alpha``. The path ``sigma_0, tau_0, ..., sigma_r = alpha`` is reversed:
    the pairs ``(sigma_i, tau_i)`` are replaced by ``(sigma_{i+1}, tau_i)``
    and ``(sigma_0, beta)`` is added. The result is again a gradient field
    (Forman 1998, Th. 11.1). ``V`` is not modified.
    """
    heads = set(V.values())
    for c in (alpha, beta):
        if c in V or c in heads:
            raise ValueError(f"{c!r} is not critical")
    if K.dim[beta] != K.dim[alpha] + 1:
        raise ValueError("dim beta must be dim alpha + 1")
    facets = list(K.facets[beta])
    count = _path_counts(K, V, alpha, facets)
    total = sum(count[f] for f in facets)
    if total != 1:
        raise ValueError(f"{total} gradient paths join beta to alpha; exactly one is needed")
    (cur,) = [f for f in facets if count[f] == 1]
    path = [cur]
    while cur != alpha:
        tau = V[cur]
        (cur,) = [f for f in K.facets[tau] if f != cur and count.get(f, 0) == 1]
        path += [tau, cur]
    W = dict(V)
    sigmas, taus = path[0::2], path[1::2]
    for s in sigmas[:-1]:
        del W[s]
    for i, t in enumerate(taus):
        W[sigmas[i + 1]] = t
    W[sigmas[0]] = beta
    return W


# ----------------------------------------------------------------------
# stacks compatible with the cancelled field
# ----------------------------------------------------------------------
def _component(K: Complex, start: Cell, inside) -> set:
    """Cells reachable from ``start`` in the incidence graph through ``inside``."""
    if not inside(start):
        return set()
    seen, stack = {start}, [start]
    while stack:
        c = stack.pop()
        for d in (*K.facets[c], *K.cofacets[c]):
            if d not in seen and inside(d):
                seen.add(d)
                stack.append(d)
    return seen


def raise_stack(K: Complex, F: Dict[Cell, float], alpha: Cell, level: float
                ) -> Dict[Cell, float]:
    """Raise to ``level`` the component of ``{F < level}`` containing ``alpha``."""
    G = dict(F)
    for c in _component(K, alpha, lambda c: F[c] < level):
        G[c] = level
    return G


def lower_stack(K: Complex, F: Dict[Cell, float], beta: Cell, level: float
                ) -> Dict[Cell, float]:
    """Lower to ``level`` the component of ``{F > level}`` containing ``beta``."""
    G = dict(F)
    for c in _component(K, beta, lambda c: F[c] > level):
        G[c] = level
    return G


def is_level_flat(V: Dict[Cell, Cell], F: Dict[Cell, float]) -> bool:
    """True when both cells of every pair of ``V`` have the same value."""
    return all(F[s] == F[t] for s, t in V.items())


# ----------------------------------------------------------------------
# simplification of minima
# ----------------------------------------------------------------------
def vertex_flows(K: Complex, V: Dict[Cell, Cell]) -> Dict[Cell, Cell]:
    """Critical vertex reached from each vertex by its maximal gradient path."""
    out: Dict[Cell, Cell] = {}
    for v in K.cells_of_dim(0):
        trail, cur = [], v
        while cur not in out and cur in V:
            trail.append(cur)
            (cur,) = [f for f in K.facets[V[cur]] if f != cur]
        root = out.get(cur, cur)
        out[cur] = root
        for x in trail:
            out[x] = root
    return out


def minimum_saddle_pairs(K: Complex, V: Dict[Cell, Cell], F: Dict[Cell, float]
                         ) -> List[Tuple[float, Cell, Cell]]:
    """Cancellable pairs ``(persistence, minimum, saddle)``.

    A critical edge whose endpoints flow to two different minima can be
    cancelled with either of them; the one with the larger value (the
    younger, by the elder rule; ties broken by the cell order) is retained,
    and the persistence is ``F(saddle) - F(minimum)``.
    """
    flow = vertex_flows(K, V)
    heads = set(V.values())
    order = {c: i for i, c in enumerate(K.cells)}
    out = []
    for s in K.cells_of_dim(1):
        if s in V or s in heads:
            continue
        a, b = K.facets[s]
        ma, mb = flow[a], flow[b]
        if ma == mb:
            continue
        m = max(ma, mb, key=lambda x: (F[x], order[x]))
        out.append((F[s] - F[m], m, s))
    out.sort(key=lambda x: (x[0], order[x[1]], order[x[2]]))
    return out


def simplify_minima(K: Complex, V: Dict[Cell, Cell], F: Dict[Cell, float],
                    threshold: float, max_steps: Optional[int] = None
                    ) -> Tuple[Dict[Cell, Cell], Dict[Cell, float], int]:
    """Cancel minimum-saddle pairs of persistence below ``threshold``.

    Pairs are cancelled one at a time, lowest persistence first; after each
    cancellation the stack is updated with :func:`raise_stack`, so the field
    stays level-flat (Theorem in the accompanying paper) and the pairs are
    recomputed. Returns ``(V, F, number of cancellations)``.
    """
    steps = 0
    while max_steps is None or steps < max_steps:
        pairs = [p for p in minimum_saddle_pairs(K, V, F) if p[0] < threshold]
        if not pairs:
            break
        _, m, s = pairs[0]
        V = cancel(K, V, m, s)
        F = raise_stack(K, F, m, F[s])
        steps += 1
    return V, F, steps
