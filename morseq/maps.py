"""Reference, coreference, extension and coextension maps, and the critical
and cocritical complexes.

Reference: G. Bertrand, *Morse sequences: a simple approach to discrete Morse
theory* (arXiv:2410.14227), Def. 6, Prop. 7, Th. 12, Def. 16/21, Def. 32.

All chains are subsets of cells, i.e. sums over the field with two elements.
"""
from __future__ import annotations

from typing import Dict, List, Set

from .complex import Cell
from .sequence import MorseSequence


def _sym(acc: Set[Cell], x: Cell) -> None:
    if x in acc:
        acc.discard(x)
    else:
        acc.add(x)


def reference_map(seq: MorseSequence) -> Dict[Cell, Set[Cell]]:
    """The reference map, computed by a left-to-right scan (Prop. 7)."""
    K = seq.K
    gam: Dict[Cell, Set[Cell]] = {}
    for it in seq.items:
        if isinstance(it, tuple) and it[0] in seq.V and seq.V.get(it[0]) == it[1]:
            s, t = it
            gam[t] = set()
            acc: Set[Cell] = set()
            for f in K.facets[t]:
                if f == s:
                    continue
                for x in gam[f]:
                    _sym(acc, x)
            gam[s] = acc
        else:
            gam[it] = {it}
    return gam


def coreference_map(seq: MorseSequence) -> Dict[Cell, Set[Cell]]:
    """The coreference map, computed by a right-to-left scan (Prop. 7)."""
    K = seq.K
    lam: Dict[Cell, Set[Cell]] = {}
    for it in reversed(seq.items):
        if isinstance(it, tuple) and it[0] in seq.V and seq.V.get(it[0]) == it[1]:
            s, t = it
            lam[s] = set()
            acc: Set[Cell] = set()
            for c in K.cofacets[s]:
                if c == t:
                    continue
                for x in lam[c]:
                    _sym(acc, x)
            lam[t] = acc
        else:
            lam[it] = {it}
    return lam


def critical_boundary(seq: MorseSequence, gam: Dict[Cell, Set[Cell]] | None = None
                      ) -> Dict[Cell, Set[Cell]]:
    """:math:`\\widehat{\\partial}\\kappa = \\gamma(\\partial\\kappa)` for critical cells."""
    gam = reference_map(seq) if gam is None else gam
    out: Dict[Cell, Set[Cell]] = {}
    for k in seq.critical:
        acc: Set[Cell] = set()
        for f in seq.K.facets[k]:
            for x in gam[f]:
                _sym(acc, x)
        out[k] = acc
    return out


def cocritical_coboundary(seq: MorseSequence, lam: Dict[Cell, Set[Cell]] | None = None
                          ) -> Dict[Cell, Set[Cell]]:
    """:math:`\\widehat{\\delta}\\kappa = \\lambda(\\delta\\kappa)` for critical cells."""
    lam = coreference_map(seq) if lam is None else lam
    out: Dict[Cell, Set[Cell]] = {}
    for k in seq.critical:
        acc: Set[Cell] = set()
        for c in seq.K.cofacets[k]:
            for x in lam[c]:
                _sym(acc, x)
        out[k] = acc
    return out


def critical_boundary_matrices(seq: MorseSequence) -> Dict[int, List[Set[Cell]]]:
    """Columns of :math:`\\widehat{\\partial}_p`, indexed by dimension ``p``."""
    d = critical_boundary(seq)
    out: Dict[int, List[Set[Cell]]] = {}
    for k, col in d.items():
        p = seq.K.dim[k]
        if p >= 1:
            out.setdefault(p, []).append(col)
    return out


def coextension(seq: MorseSequence, kappa: Cell,
                gam: Dict[Cell, Set[Cell]] | None = None) -> Set[Cell]:
    """:math:`\\widehat{\\Upsilon}(\\kappa)`: cells whose reference contains ``kappa``."""
    gam = reference_map(seq) if gam is None else gam
    p = seq.K.dim[kappa]
    return {c for c in seq.K.cells if seq.K.dim[c] == p and kappa in gam[c]}


def extension(seq: MorseSequence, kappa: Cell,
              lam: Dict[Cell, Set[Cell]] | None = None) -> Set[Cell]:
    """:math:`\\widehat{\\Lambda}(\\kappa)`: cells whose coreference contains ``kappa``."""
    lam = coreference_map(seq) if lam is None else lam
    p = seq.K.dim[kappa]
    return {c for c in seq.K.cells if seq.K.dim[c] == p and kappa in lam[c]}


def count_gradient_paths(seq: MorseSequence, source: Cell, target: Cell) -> int:
    """Number of gradient paths from ``source`` to ``target`` (exact count).

    Used to check Theorem 12 (parity) against the algebraic maps, and to
    exhibit the gap between path existence and path parity.
    """
    K, V = seq.K, seq.V
    memo: Dict[Cell, int] = {}

    def rec(cur: Cell) -> int:
        if cur in memo:
            return memo[cur]
        total = 1 if cur == target else 0
        if cur in V:
            t = V[cur]
            for nxt in K.facets[t]:
                if nxt != cur:
                    total += rec(nxt)
        memo[cur] = total
        return total

    return rec(source)
