"""Stacks, cuts, F-sequences and grayscale images.

A *stack* is a monotone function on a complex; the max-extension of a vertex
map is the canonical example. Cuts of a stack are hermetic for an
F-sequence, which is what makes a single computation valid at every grayscale
level -- with no injectivity assumption, hence with plateaus.

Reference: G. Bertrand, *Morse sequences on stacks and flooding sequences*
(arXiv:2509.01384), Sec. 2.3, Def. 4-5, Prop. 7.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .complex import Cell, Complex, cubical_complex, cube_vertex, vertex_coords
from .sequence import MorseSequence, sequence_from_field


def max_extension(K: Complex, f: Dict[Cell, int]) -> Dict[Cell, int]:
    """Extend a vertex map to all cells by ``F(sigma) = max f(v), v <= sigma``.

    The result is a stack: monotone for the face relation.
    """
    F: Dict[Cell, int] = {}
    for c in sorted(K.cells, key=lambda c: K.dim[c]):
        if K.dim[c] == 0:
            if c not in f:
                raise KeyError(f"missing value for vertex {c!r}")
            F[c] = f[c]
        else:
            F[c] = max(F[x] for x in K.facets[c])
    return F


def is_stack(K: Complex, F: Dict[Cell, int]) -> bool:
    """True when ``F`` is monotone for the face relation."""
    return all(F[f] <= F[c] for c in K.cells for f in K.facets[c])


def cut(K: Complex, F: Dict[Cell, int], level: int) -> Set[Cell]:
    """Cut :math:`F_\\lambda = \\{\\nu : F(\\nu) \\le \\lambda\\}`."""
    return {c for c in K.cells if F[c] <= level}


def is_f_sequence(seq: MorseSequence, F: Dict[Cell, int]) -> bool:
    """True when every regular pair is level-flat (Bertrand, Def. 4)."""
    return all(F[s] == F[t] for s, t in seq.V.items())


def is_flooding_sequence(seq: MorseSequence, F: Dict[Cell, int]) -> bool:
    """True when values never decrease along the sequence (Bertrand, Def. 5)."""
    if not is_f_sequence(seq, F):
        return False
    last = None
    for it in seq.items:
        cells = it if isinstance(it, tuple) and it[0] in seq.V and seq.V.get(it[0]) == it[1] else (it,)
        val = max(F[c] for c in cells)
        if last is not None and val < last:
            return False
        last = val
    return True


# ----------------------------------------------------------------------
# Building a gradient field from grayscale data (lower-star processing)
# ----------------------------------------------------------------------
def lower_star(K: Complex, F: Dict[Cell, int], v: Cell,
               tie: Dict[Cell, int]) -> List[Cell]:
    """Cells whose maximal vertex, for the total order ``tie``, is ``v``."""
    out = []
    for c in K.cells:
        verts = [x for x in K.closure([c]) if K.dim[x] == 0]
        if max(verts, key=lambda x: (F[x], tie[x])) == v:
            out.append(c)
    return out


def gradient_from_vertex_map(K: Complex, f: Dict[Cell, int],
                             tie: Optional[Dict[Cell, int]] = None
                             ) -> Tuple[Dict[Cell, Cell], Dict[Cell, int]]:
    """Discrete gradient field from a vertex map, by lower-star processing.

    Returns ``(V, F)`` where ``F`` is the max-extension of ``f``. The pairing
    is built inside each lower star by repeated *coreductions* (pair a cell
    that has exactly one unprocessed cofacet), the free-face rule of Bertrand's
    increasing scheme; a cell is made critical only when no coreduction is
    available. The construction is level-flat by design, so the resulting
    sequence is an F-sequence and plateaus need no perturbation of the data:
    ``tie`` only orders cells *inside* one level.

    Complexity is linear in the incidence structure of each lower star.
    """
    F = max_extension(K, f)
    verts = K.cells_of_dim(0)
    if tie is None:
        tie = {v: i for i, v in enumerate(verts)}

    # G(c) = vertex values of c in decreasing order; the total order used to
    # compare cells inside one lower star (RWS Sec. 3).
    vert_of: Dict[Cell, List[Cell]] = {}
    G: Dict[Cell, tuple] = {}
    for c in K.cells:
        vs = [x for x in K.closure([c]) if K.dim[x] == 0]
        vert_of[c] = vs
        G[c] = tuple(sorted(((F[x], tie[x]) for x in vs), reverse=True))
    owner: Dict[Cell, Cell] = {c: max(vert_of[c], key=lambda x: (F[x], tie[x]))
                              for c in K.cells}
    stars: Dict[Cell, List[Cell]] = {v: [] for v in verts}
    for c in K.cells:
        stars[owner[c]].append(c)

    V: Dict[Cell, Cell] = {}
    for v in verts:
        L = set(stars[v])
        if len(L) == 1:
            continue  # v is critical
        paired: set = set()
        critical: set = set()

        def unpaired_facets(a: Cell) -> List[Cell]:
            return [f for f in K.facets[a]
                    if f in L and f not in paired and f not in critical]

        edges = [c for c in L if K.dim[c] == 1]
        delta = min(edges, key=lambda e: (G[e], str(e)))
        V[v] = delta
        paired.update((v, delta))

        pq_one = [a for a in L if a not in paired and len(unpaired_facets(a)) == 1]
        pq_zero = [a for a in L if a not in paired and not unpaired_facets(a)]
        while pq_one or pq_zero:
            while pq_one:
                pq_one.sort(key=lambda a: (G[a], str(a)))
                alpha = pq_one.pop(0)
                if alpha in paired or alpha in critical:
                    continue
                uf = unpaired_facets(alpha)
                if not uf:
                    pq_zero.append(alpha)
                else:
                    V[uf[0]] = alpha
                    paired.update((uf[0], alpha))
                pq_one += [a for a in L if a not in paired and a not in critical
                           and a not in pq_one and len(unpaired_facets(a)) == 1]
            pq_zero = [a for a in pq_zero if a not in paired and a not in critical]
            if pq_zero:
                pq_zero.sort(key=lambda a: (G[a], str(a)))
                gamma = pq_zero.pop(0)
                critical.add(gamma)
                pq_one += [a for a in L if a not in paired and a not in critical
                           and len(unpaired_facets(a)) == 1]
    return V, F


def flooding_order(F: Dict[Cell, int]):
    """Sort key producing a flooding sequence (levels in increasing order)."""
    def key(item):
        cells = item if isinstance(item, tuple) and len(item) == 2 and item[0] in F and item[1] in F else (item,)
        try:
            return (max(F[c] for c in cells), str(cells))
        except (KeyError, TypeError):
            return (F[item], str(item))
    return key


def segment_image(image, connectivity: str = "cubical"):
    """Segment a 2-D or 3-D grayscale array; returns a :class:`Segmentation`.

    ``image`` is any nested sequence or NumPy array of integers. Plateaus are
    supported natively: no perturbation is applied to the data.

    >>> seg = segment_image([[0, 2, 1], [2, 2, 2], [1, 2, 0]])
    >>> seg.vertex_labels()[0][0] != seg.vertex_labels()[2][2]
    True
    """
    try:
        import numpy as np
        arr = np.asarray(image)
        shape = arr.shape
        get = lambda idx: int(arr[idx])
    except Exception:  # pragma: no cover - numpy is optional
        arr = image
        shape = (len(arr), len(arr[0])) if not isinstance(arr[0][0], (list, tuple)) \
            else (len(arr), len(arr[0]), len(arr[0][0]))

        def get(idx):
            cur = arr
            for i in idx:
                cur = cur[i]
            return int(cur)

    if connectivity != "cubical":
        raise ValueError("only the cubical model is supported")
    K = cubical_complex(shape)
    f = {}
    for c in K.cells_of_dim(0):
        f[c] = get(vertex_coords(c))
    V, F = gradient_from_vertex_map(K, f)
    seq = sequence_from_field(K, V, key=flooding_order(F))
    return Segmentation(K, seq, F, shape)


class Segmentation:
    """Result of :func:`segment_image`: complex, sequence, stack, labels."""

    def __init__(self, K: Complex, seq: MorseSequence, F: Dict[Cell, int],
                 shape: Sequence[int]) -> None:
        self.K = K
        self.sequence = seq
        self.F = F
        self.shape = tuple(shape)

    # -- delegation ------------------------------------------------------
    def labels(self):
        from .segmentation import basins
        return basins(self.sequence)

    def basin_sets(self):
        from .segmentation import basin_sets
        return basin_sets(self.sequence)

    def bridges(self):
        from .segmentation import bridges
        return bridges(self.sequence)

    def vertex_labels(self):
        """Label array over the original grid (list of lists / nested lists)."""
        from .segmentation import vertex_partition
        part = vertex_partition(self.sequence)
        roots = sorted({vertex_coords(m) for m in part.values()})
        index = {r: i for i, r in enumerate(roots)}

        def build(prefix):
            d = len(prefix)
            if d == len(self.shape):
                return index[vertex_coords(part[cube_vertex(prefix)])]
            return [build(prefix + (i,)) for i in range(self.shape[d])]

        return build(())

    def basins_at_level(self, level: int):
        """Basins of the cut at ``level`` (restriction of the global labels)."""
        C = cut(self.K, self.F, level)
        out = {}
        for m, cells in self.basin_sets().items():
            if self.F[m] <= level:
                out[m] = cells & C
        return out

    def summary(self) -> str:
        crit = self.sequence.critical_by_dim()
        return (f"complex: {len(self.K)} cells, shape {self.shape}\n"
                f"critical cells: "
                + ", ".join(f"dim {p}: {len(v)}" for p, v in sorted(crit.items()))
                + f"\nbasins: {len(self.basin_sets())}"
                f"\nbridges: {len(self.bridges())}"
                f"\nlevels: {min(self.F.values())}..{max(self.F.values())}")
