"""Segmentation layer: stable/unstable sets, vertex partition, basins.

Implements the statements proved in Part II of the report:

* :func:`stable_set` / :func:`unstable_set` -- path-existence definitions,
  equivalent to the recursive sets of Delgado-Friedrichs, Robins and Sheppard
  (DFRS), *Skeletonization and partitioning of digital images using discrete
  Morse theory*, IEEE TPAMI 37(3), 2015, Sec. 3.1;
* :func:`vertex_partition` -- deterministic flow at dimension zero (DFRS
  Lemma 5);
* :func:`is_hermetic` -- DFRS Sec. 3.1;
* :func:`basin` / :func:`basins` -- basins as unique maximal closures under
  regular expansions, and the one-pass labeling algorithm.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .complex import Cell, Complex
from .sequence import MorseSequence


# ----------------------------------------------------------------------
# B5 -- stable and unstable sets
# ----------------------------------------------------------------------
def stable_set(seq: MorseSequence, kappa: Cell) -> Set[Cell]:
    """Cells joined to ``kappa`` by a gradient path (trivial path included).

    Equal to the recursive stable set :math:`S_K(\\kappa)` of DFRS.
    """
    K, V = seq.K, seq.V
    back: Dict[Cell, List[Cell]] = {}
    for s, t in V.items():
        for nxt in K.facets[t]:
            if nxt != s:
                back.setdefault(nxt, []).append(s)
    out, stack = {kappa}, [kappa]
    while stack:
        u = stack.pop()
        for w in back.get(u, ()):
            if w not in out:
                out.add(w)
                stack.append(w)
    return out


def unstable_set(seq: MorseSequence, kappa: Cell) -> Set[Cell]:
    """Cells reached from ``kappa`` by a cogradient path.

    Equal to the recursive unstable set :math:`U_K(\\kappa)` of DFRS.
    """
    K, V = seq.K, seq.V
    fwd: Dict[Cell, List[Cell]] = {}
    for s, t in V.items():
        for c in K.cofacets[s]:
            if c != t:
                fwd.setdefault(c, []).append(t)
    out, stack = {kappa}, [kappa]
    while stack:
        u = stack.pop()
        for w in fwd.get(u, ()):
            if w not in out:
                out.add(w)
                stack.append(w)
    return out


def unstable_complex(seq: MorseSequence, kappa: Cell) -> Set[Cell]:
    """Closure of the unstable set (DFRS :math:`W_K(\\kappa)`)."""
    return seq.K.closure(unstable_set(seq, kappa))


def morse_skeleton(seq: MorseSequence) -> Set[Cell]:
    """Union of the unstable complexes of all critical cells (DFRS Sec. 3.2)."""
    out: Set[Cell] = set()
    for c in seq.critical:
        out |= unstable_complex(seq, c)
    return out


# ----------------------------------------------------------------------
# B6 -- deterministic vertex flow
# ----------------------------------------------------------------------
def vertex_flow(seq: MorseSequence, v: Cell) -> Cell:
    """Endpoint of the unique maximal (0,1) gradient path starting at ``v``."""
    K, V = seq.K, seq.V
    if K.dim[v] != 0:
        raise ValueError("vertex_flow expects a 0-cell")
    cur, seen = v, {v}
    while cur in V:
        e = V[cur]
        nxts = [f for f in K.facets[e] if f != cur]
        if len(nxts) != 1:
            raise RuntimeError("an edge should have exactly two vertices")
        cur = nxts[0]
        if cur in seen:
            raise RuntimeError("cyclic field: not a Morse sequence")
        seen.add(cur)
    return cur


def vertex_partition(seq: MorseSequence) -> Dict[Cell, Cell]:
    """Map every vertex to the critical vertex it descends to (DFRS Lemma 5)."""
    return {v: vertex_flow(seq, v) for v in seq.K.cells_of_dim(0)}


# ----------------------------------------------------------------------
# B8 -- hermetic subcomplexes
# ----------------------------------------------------------------------
def is_hermetic(seq: MorseSequence, H: Iterable[Cell]) -> bool:
    """True when ``H`` is a subcomplex closed under the field (DFRS Sec. 3.1)."""
    S = set(H)
    if not seq.K.is_subcomplex(S):
        return False
    return all((s in S) == (t in S) for s, t in seq.V.items())


def restrict(seq: MorseSequence, H: Iterable[Cell]) -> MorseSequence:
    """Restriction of the sequence to a hermetic subcomplex ``H``."""
    S = set(H)
    if not is_hermetic(seq, S):
        raise ValueError("H is not hermetic for this field")
    sub = Complex(
        [c for c in seq.K.cells if c in S],
        {c: [f for f in seq.K.facets[c]] for c in seq.K.cells if c in S},
        {c: seq.K.dim[c] for c in seq.K.cells if c in S},
    )
    items = []
    for it in seq.items:
        if isinstance(it, tuple) and it[0] in seq.V and seq.V.get(it[0]) == it[1]:
            if it[0] in S:
                items.append(it)
        elif it in S:
            items.append(it)
    return MorseSequence(sub, items)


# ----------------------------------------------------------------------
# B11 -- basins
# ----------------------------------------------------------------------
def requirements(K: Complex, sigma: Cell, tau: Cell) -> Set[Cell]:
    """Requirement set :math:`\\mathrm{req}(\\sigma,\\tau)` of a regular pair."""
    return set(K.facets[sigma]) | {f for f in K.facets[tau] if f != sigma}


def basin(seq: MorseSequence, m: Cell, order: Optional[Sequence] = None) -> Set[Cell]:
    """Maximal closure of ``{m}`` under available regular expansions.

    By the closure theorem this set is independent of ``order`` and equals the
    DFRS basin :math:`B_K(m)`. ``order`` exists only so that tests can check
    order-independence explicitly.
    """
    K, V = seq.K, seq.V
    if K.dim[m] != 0 or m in V or m in seq.V_inv:
        raise ValueError("m must be a critical vertex")
    pairs = list(V.items()) if order is None else list(order)
    B: Set[Cell] = {m}
    changed = True
    while changed:
        changed = False
        for s, t in pairs:
            if s in B:
                continue
            if requirements(K, s, t) <= B:
                B.update((s, t))
                changed = True
    return B


def basins(seq: MorseSequence) -> Dict[Cell, Optional[Cell]]:
    """One-pass labeling: ``cell -> critical vertex`` or ``None``.

    Single left-to-right scan of the sequence; the label of a cell is the
    minimum whose basin contains it, or ``None`` when the cell belongs to no
    basin (in dimension one these are the *bridges* of DFRS).
    """
    K = seq.K
    lab: Dict[Cell, Optional[Cell]] = {c: None for c in K.cells}
    for it in seq.items:
        if isinstance(it, tuple) and it[0] in seq.V and seq.V.get(it[0]) == it[1]:
            s, t = it
            labels = {lab[x] for x in requirements(K, s, t)}
            if len(labels) == 1:
                (only,) = tuple(labels)
                if only is not None:
                    lab[s] = lab[t] = only
        else:
            if K.dim[it] == 0:
                lab[it] = it
    return lab


def basin_sets(seq: MorseSequence) -> Dict[Cell, Set[Cell]]:
    """Basins as a dictionary ``critical vertex -> set of cells``."""
    lab = basins(seq)
    out: Dict[Cell, Set[Cell]] = {}
    for c, m in lab.items():
        if m is not None:
            out.setdefault(m, set()).add(c)
    return out


def bridges(seq: MorseSequence) -> Set[Cell]:
    """One-cells belonging to no basin (DFRS Sec. 3.3)."""
    lab = basins(seq)
    return {c for c, m in lab.items() if m is None and seq.K.dim[c] == 1}


def regular_collapse_order(seq: MorseSequence, B: Iterable[Cell], m: Cell) -> List[Tuple[Cell, Cell]]:
    """A regular collapse of the basin ``B`` onto ``{m}``.

    Returns the pairs in removal order and raises if ``B`` does not collapse
    regularly onto ``m`` -- a useful independent check of the closure theorem.
    """
    K, V = seq.K, seq.V
    cur = set(B)
    removed: List[Tuple[Cell, Cell]] = []
    added = [it for it in seq.items
             if isinstance(it, tuple) and it[0] in V and V.get(it[0]) == it[1] and it[0] in cur]
    for s, t in reversed(added):
        cofs = [c for c in K.cofacets[s] if c in cur]
        if cofs != [t]:
            raise ValueError(f"pair ({s!r},{t!r}) is not free during the collapse")
        cur -= {s, t}
        removed.append((s, t))
    if cur != {m}:
        raise ValueError(f"collapse ended on {cur!r} instead of {{{m!r}}}")
    return removed
