"""Morse sequences, gradient vector fields, stacks and F-sequences.

Reference: G. Bertrand, *Morse sequences: a simple approach to discrete Morse
theory*, JMIV 2025 (arXiv:2410.14227), and *Morse sequences on stacks and
flooding sequences* (arXiv:2509.01384).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .complex import Cell, Complex


@dataclass
class MorseSequence:
    """A cell-wise Morse sequence on a complex.

    ``items`` is the sequence of steps: a bare cell for a *filling* (critical
    step) and a pair ``(sigma, tau)`` for an *expansion* (regular step).
    """

    K: Complex
    items: List[object] = field(default_factory=list)

    # -- derived data ---------------------------------------------------
    def __post_init__(self) -> None:
        self.V: Dict[Cell, Cell] = {}
        self.V_inv: Dict[Cell, Cell] = {}
        self.critical: List[Cell] = []
        self.step: Dict[Cell, int] = {}
        for i, it in enumerate(self.items):
            if isinstance(it, tuple) and len(it) == 2 and it[0] in self.K and it[1] in self.K \
                    and self.K.dim[it[1]] == self.K.dim[it[0]] + 1:
                s, t = it
                self.V[s] = t
                self.V_inv[t] = s
                self.step[s] = self.step[t] = i
            else:
                self.critical.append(it)
                self.step[it] = i

    # -- validation -----------------------------------------------------
    def validate(self) -> None:
        """Check that the sequence is a genuine Morse sequence.

        Verifies that every prefix is a subcomplex, that each regular step is
        an elementary expansion (the pair is free at insertion time) and that
        each critical step is an elementary filling.
        """
        present: set = set()
        for i, it in enumerate(self.items):
            if isinstance(it, tuple) and it in [(s, t) for s, t in self.V.items()]:
                s, t = it
                for f in self.K.facets[t]:
                    if f != s and f not in present:
                        raise ValueError(f"step {i}: face {f!r} of {t!r} missing")
                for f in self.K.facets[s]:
                    if f not in present:
                        raise ValueError(f"step {i}: face {f!r} of {s!r} missing")
                for c in self.K.cofacets[s]:
                    if c in present:
                        raise ValueError(f"step {i}: pair ({s!r},{t!r}) not free")
                present.update((s, t))
            else:
                for f in self.K.facets[it]:
                    if f not in present:
                        raise ValueError(f"step {i}: face {f!r} of {it!r} missing")
                present.add(it)
        if present != set(self.K.cells):
            missing = set(self.K.cells) - present
            raise ValueError(f"sequence does not cover the complex ({len(missing)} cells missing)")

    # -- accessors ------------------------------------------------------
    def critical_by_dim(self) -> Dict[int, List[Cell]]:
        out: Dict[int, List[Cell]] = {}
        for c in self.critical:
            out.setdefault(self.K.dim[c], []).append(c)
        return out

    def prefix(self, i: int) -> set:
        """Cells of the prefix :math:`K_i` (first ``i`` steps)."""
        out: set = set()
        for it in self.items[:i]:
            if isinstance(it, tuple) and it[0] in self.V and self.V[it[0]] == it[1]:
                out.update(it)
            else:
                out.add(it)
        return out

    def betti_numbers_mod2(self) -> Dict[int, int]:
        """Betti numbers over F2, computed from the critical complex."""
        from .maps import critical_boundary_matrices

        mats = critical_boundary_matrices(self)
        crit = self.critical_by_dim()
        ranks = {p: _rank_f2(m) for p, m in mats.items()}
        out = {}
        for p in range(self.K.dimension + 1):
            n = len(crit.get(p, []))
            out[p] = n - ranks.get(p, 0) - ranks.get(p + 1, 0)
        return out


def _rank_f2(rows: List[set]) -> int:
    """Rank over F2 of a matrix given as a list of column supports."""
    basis: List[set] = []
    rank = 0
    for col in rows:
        cur = set(col)
        for b in basis:
            piv = min(b)
            if piv in cur:
                cur ^= b
        if cur:
            basis.append(cur)
            rank += 1
    return rank


# ----------------------------------------------------------------------
# Building sequences from a gradient vector field
# ----------------------------------------------------------------------
def is_valid_field(K: Complex, V: Dict[Cell, Cell]) -> bool:
    """True when ``V`` is a matching of facet pairs (not necessarily acyclic)."""
    seen: set = set()
    for s, t in V.items():
        if s not in K or t not in K:
            return False
        if K.dim[t] != K.dim[s] + 1 or s not in K.facets[t]:
            return False
        if s in seen or t in seen:
            return False
        seen.update((s, t))
    return True


def is_acyclic(K: Complex, V: Dict[Cell, Cell]) -> bool:
    """True when ``V`` has no non-trivial closed gradient path."""
    adj: Dict[Cell, List[Cell]] = {}
    for s, t in V.items():
        adj[s] = [n for n in K.facets[t] if n != s]
    colour: Dict[Cell, int] = {}

    def visit(root: Cell) -> bool:
        stack = [(root, iter(adj.get(root, ())))]
        colour[root] = 1
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                c = colour.get(nxt, 0)
                if c == 1:
                    return False
                if c == 0 and nxt in adj:
                    colour[nxt] = 1
                    stack.append((nxt, iter(adj.get(nxt, ()))))
                    advanced = True
                    break
                colour.setdefault(nxt, 2)
            if not advanced:
                colour[node] = 2
                stack.pop()
        return True

    for u in adj:
        if colour.get(u, 0) == 0 and not visit(u):
            return False
    return True


def sequence_from_field(K: Complex, V: Dict[Cell, Cell],
                        key=None) -> MorseSequence:
    """Build a Morse sequence inducing the acyclic field ``V``.

    Cells are inserted as soon as their required faces are present, which is
    always possible when ``V`` is acyclic (Bertrand, Theorem 51). ``key`` is an
    optional priority function on items, used to make the output deterministic
    or to follow a prescribed order (see :func:`morseq.stacks.flooding_order`).
    """
    if not is_valid_field(K, V):
        raise ValueError("V is not a valid discrete vector field")
    if not is_acyclic(K, V):
        raise ValueError("V is not acyclic: no Morse sequence induces it")

    crit = [c for c in K.cells if c not in V and c not in set(V.values())]
    pending: List[object] = [(s, t) for s, t in V.items()] + list(crit)
    if key is not None:
        pending.sort(key=key)

    present: set = set()
    items: List[object] = []
    remaining = list(pending)
    while remaining:
        progress = False
        still: List[object] = []
        for it in remaining:
            if isinstance(it, tuple) and it[0] in V and V.get(it[0]) == it[1]:
                s, t = it
                ok = all(f in present for f in K.facets[t] if f != s) and \
                     all(f in present for f in K.facets[s])
            else:
                ok = all(f in present for f in K.facets[it])
            if ok:
                items.append(it)
                present.update(it if isinstance(it, tuple) and it[0] in V and V.get(it[0]) == it[1] else (it,))
                progress = True
            else:
                still.append(it)
        if not progress:
            raise RuntimeError("cannot schedule the field (should not happen for acyclic V)")
        remaining = still
    seq = MorseSequence(K, items)
    return seq
