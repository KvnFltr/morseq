"""Cell complexes: abstract interface, cubical grids (any dimension), and
simplicial complexes.

A complex is described by its cells and the incidence (facet) relation. All
algorithms in :mod:`morseq` use nothing else, so they apply verbatim to
cubical and simplicial complexes, in any dimension.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

Cell = Tuple  # a cell is any hashable tuple


class Complex:
    """A finite cell complex given by cells and their facets.

    Parameters
    ----------
    cells:
        Iterable of hashable cells.
    facets:
        Mapping ``cell -> list of codimension-1 faces``.
    dims:
        Mapping ``cell -> dimension``.

    The constructor checks that the complex is closed under taking faces and
    that dimensions are consistent, so downstream algorithms may assume both.
    """

    def __init__(self, cells: Iterable[Cell], facets: Dict[Cell, Sequence[Cell]],
                 dims: Dict[Cell, int]) -> None:
        self.cells: List[Cell] = list(cells)
        self._cellset = set(self.cells)
        self.facets: Dict[Cell, Tuple[Cell, ...]] = {
            c: tuple(facets.get(c, ())) for c in self.cells
        }
        self.dim: Dict[Cell, int] = dict(dims)
        self._validate()
        self.cofacets: Dict[Cell, Tuple[Cell, ...]] = self._build_cofacets()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if len(self._cellset) != len(self.cells):
            raise ValueError("duplicate cells")
        for c in self.cells:
            if c not in self.dim:
                raise ValueError(f"missing dimension for cell {c!r}")
            for f in self.facets[c]:
                if f not in self._cellset:
                    raise ValueError(f"complex not closed under faces: {f!r} < {c!r}")
                if self.dim[f] != self.dim[c] - 1:
                    raise ValueError(f"facet {f!r} of {c!r} has wrong dimension")
            if len(set(self.facets[c])) != len(self.facets[c]):
                raise ValueError(f"repeated facet in boundary of {c!r}")

    def _build_cofacets(self) -> Dict[Cell, Tuple[Cell, ...]]:
        acc: Dict[Cell, List[Cell]] = {c: [] for c in self.cells}
        for c in self.cells:
            for f in self.facets[c]:
                acc[f].append(c)
        return {c: tuple(v) for c, v in acc.items()}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.cells)

    def __contains__(self, c: Cell) -> bool:
        return c in self._cellset

    def cells_of_dim(self, p: int) -> List[Cell]:
        """Cells of dimension ``p``."""
        return [c for c in self.cells if self.dim[c] == p]

    @property
    def dimension(self) -> int:
        """Top dimension of the complex (``-1`` if empty)."""
        return max(self.dim.values(), default=-1)

    def closure(self, cells: Iterable[Cell]) -> set:
        """Smallest subcomplex containing ``cells``."""
        out, stack = set(), list(cells)
        while stack:
            c = stack.pop()
            if c in out:
                continue
            out.add(c)
            stack.extend(self.facets[c])
        return out

    def is_subcomplex(self, cells: Iterable[Cell]) -> bool:
        """True when ``cells`` is closed under taking faces."""
        s = set(cells)
        return all(f in s for c in s for f in self.facets[c])

    def euler_characteristic(self) -> int:
        chi = 0
        for c in self.cells:
            chi += (-1) ** self.dim[c]
        return chi

    def betti_check_counts(self) -> Dict[int, int]:
        """Number of cells per dimension (useful in tests and reports)."""
        out: Dict[int, int] = {}
        for c in self.cells:
            out[self.dim[c]] = out.get(self.dim[c], 0) + 1
        return out


# ----------------------------------------------------------------------
# Cubical complexes
# ----------------------------------------------------------------------
def cubical_complex(shape: Sequence[int]) -> Complex:
    """Full cubical complex of a grid of ``shape`` *vertices* per axis.

    A cell is encoded as a tuple of intervals, one per axis: ``(k, k)`` for a
    degenerate factor and ``(k, k + 1)`` for a non-degenerate one. This is the
    standard elementary-cube encoding, valid in any dimension.

    >>> K = cubical_complex((3, 3))
    >>> K.betti_check_counts()
    {0: 9, 1: 12, 2: 4}
    >>> K.euler_characteristic()
    1
    """
    shape = tuple(int(n) for n in shape)
    if not shape or any(n < 1 for n in shape):
        raise ValueError("shape must be a non-empty tuple of positive ints")

    axis_intervals = []
    for n in shape:
        iv = [(k, k) for k in range(n)] + [(k, k + 1) for k in range(n - 1)]
        axis_intervals.append(iv)

    cells: List[Cell] = []

    def rec(prefix: Tuple) -> None:
        d = len(prefix)
        if d == len(shape):
            cells.append(prefix)
            return
        for iv in axis_intervals[d]:
            rec(prefix + (iv,))

    rec(())

    dims = {c: sum(1 for a, b in c if b > a) for c in cells}
    cellset = set(cells)
    facets: Dict[Cell, List[Cell]] = {}
    for c in cells:
        fs = []
        for i, (a, b) in enumerate(c):
            if b > a:
                for endpoint in (a, b):
                    f = c[:i] + ((endpoint, endpoint),) + c[i + 1:]
                    if f in cellset:
                        fs.append(f)
        facets[c] = fs
    return Complex(cells, facets, dims)


def cube_vertex(coords: Sequence[int]) -> Cell:
    """Encode grid coordinates as a vertex of :func:`cubical_complex`."""
    return tuple((int(k), int(k)) for k in coords)


def vertex_coords(cell: Cell) -> Tuple[int, ...]:
    """Grid coordinates of a cubical vertex (inverse of :func:`cube_vertex`)."""
    if any(b > a for a, b in cell):
        raise ValueError("not a vertex")
    return tuple(a for a, _ in cell)


# ----------------------------------------------------------------------
# Simplicial complexes
# ----------------------------------------------------------------------
def simplicial_complex(maximal_faces: Iterable[Iterable]) -> Complex:
    """Simplicial complex generated by ``maximal_faces``.

    Each simplex is stored as a sorted tuple of vertices; the complex is
    closed automatically.

    >>> K = simplicial_complex([(1, 2, 3)])
    >>> K.betti_check_counts()
    {0: 3, 1: 3, 2: 1}
    """
    cells = set()
    for face in maximal_faces:
        verts = tuple(sorted(set(face)))
        if not verts:
            raise ValueError("a simplex must be non-empty")
        for k in range(1, len(verts) + 1):
            cells.update(combinations(verts, k))
    cells = sorted(cells, key=lambda c: (len(c), c))
    dims = {c: len(c) - 1 for c in cells}
    facets = {
        c: [c[:i] + c[i + 1:] for i in range(len(c))] if len(c) > 1 else []
        for c in cells
    }
    return Complex(cells, facets, dims)
