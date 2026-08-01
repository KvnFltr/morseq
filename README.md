# morseq

Discrete Morse sequences and topological image segmentation, in pure Python.

`morseq` implements the segmentation layer of discrete Morse theory — stable
and unstable sets, the partition of vertices by minima, hermetic restrictions,
and **basins** — in the framework of *Morse sequences*, where acyclicity of the
gradient holds by construction.

The library is generic: every algorithm uses only the incidence relation, so it
runs on cubical complexes of **any dimension** and on simplicial complexes
alike. Grayscale data with **plateaus** are handled without perturbing the
values.

```python
from morseq import segment_image

seg = segment_image([[0, 3, 3, 3, 1],
                     [3, 3, 5, 3, 3],
                     [3, 5, 5, 5, 3],
                     [1, 3, 3, 3, 0]])

print(seg.summary())
for row in seg.vertex_labels():
    print(" ".join(map(str, row)))
```

```
complex: 63 cells, shape (4, 5)
critical cells: dim 0: 4, dim 1: 4, dim 2: 1
basins: 4
bridges: 12
levels: 0..5

0 0 0 1 1
0 0 0 1 1
2 0 2 1 3
2 2 2 3 3
```

## Install

```bash
pip install morseq                 # core library, no dependencies
pip install "morseq[plot]"         # + matplotlib/numpy for figures
```

From source:

```bash
git clone https://github.com/KvnFltr/morseq
cd morseq && pip install -e ".[dev]" && pytest
```

## What it computes

| Object | Function | Reference |
|---|---|---|
| Cubical complex, any dimension | `cubical_complex(shape)` | — |
| Simplicial complex | `simplicial_complex(faces)` | — |
| Gradient field from a vertex map | `gradient_from_vertex_map(K, f)` | RWS 2011, Alg. 1 |
| Morse sequence inducing a field | `sequence_from_field(K, V)` | Bertrand 2025, Th. 51 |
| Reference / coreference maps | `reference_map`, `coreference_map` | Bertrand 2025, Def. 6 |
| Critical / cocritical complexes | `critical_boundary`, `cocritical_coboundary` | Bertrand 2025, Def. 16/21 |
| Extension / coextension | `extension`, `coextension` | Bertrand 2025, Def. 32 |
| Exact gradient-path count | `count_gradient_paths` | Bertrand 2025, Th. 12 |
| Stable / unstable sets | `stable_set`, `unstable_set` | DFRS 2015, Sec. 3.1 |
| Vertex partition by minima | `vertex_partition` | DFRS 2015, Lem. 5 |
| Hermetic test and restriction | `is_hermetic`, `restrict` | DFRS 2015, Lem. 2 |
| Morse skeleton | `morse_skeleton` | DFRS 2015, Sec. 3.2 |
| **Basins** (one-pass labeling) | `basins`, `basin_sets` | this work |
| Bridges | `bridges` | DFRS 2015, Sec. 3.3 |
| Stacks, cuts, F-sequences | `max_extension`, `cut`, `is_f_sequence` | Bertrand 2026 |
| Image segmentation, 2-D/3-D | `segment_image` | this work |

## The basin algorithm

A basin is computed as the **maximal closure of a minimum under regular
expansions**: repeatedly add any pair of the gradient field whose required
faces are already present. The terminal subcomplex does not depend on the order
of the additions, and coincides with the DFRS basin — the maximal subcomplex
that regularly collapses onto the minimum.

Because the required faces of a pair always precede it in a Morse sequence, all
basins are obtained in **a single left-to-right scan**, in time linear in the
incidence structure:

```python
from morseq import basins
labels = basins(seq)      # cell -> critical vertex, or None
```

Cells that stay unlabeled belong to no basin; in dimension one these are
exactly the *bridges* of DFRS.

## Levels, cuts, plateaus

For an `F`-sequence, every cut of the stack is hermetic, so the global labels
restrict to every grayscale level with no recomputation:

```python
seg.basins_at_level(3)     # basins of the cut at level 3
```

No injectivity assumption is made on the data: a plateau is processed as one
level, the tie-breaking order is used only *inside* a level and never changes
the stack itself.

## Verification

Every theorem is an executable test. The suite checks, on the guide example, on
2-D and 3-D cubical grids, on simplicial complexes and on randomly generated
acyclic fields:

* stable and unstable sets equal the **literal recursive definitions** of DFRS;
* the parity characterization (`count_gradient_paths` vs the reference map) —
  Theorem 12 — holds for every cell and every critical cell;
* path existence and path parity differ exactly where paths branch;
* every vertex has a unique maximal path, and classes partition the vertices;
* prefixes and stack cuts are hermetic, and stable sets restrict to them;
* the basin closure is independent of the expansion order;
* the one-pass algorithm returns exactly the closure;
* each basin is a hermetic subcomplex, contains a single critical cell,
  collapses regularly onto its minimum, admits no further available pair, and
  meets the vertices in the stable set;
* basins are pairwise disjoint and restrict to prefixes and cuts;
* the critical and cocritical complexes satisfy `∂∘∂ = 0` and `δ∘δ = 0`, and
  the Euler characteristic of the critical complex matches that of the complex.

```bash
pytest -q                    # 623 tests
morseq check --shape 5x4     # same statements, on the command line
morseq check --shape 3x3x3   # in dimension three
```

## Command line

```bash
morseq segment image.png --labels labels.json --plot out.png
morseq check --shape 6x5 --trials 50
```

## References

* V. Robins, P. J. Wood, A. P. Sheppard. *Theory and algorithms for constructing
  discrete Morse complexes from grayscale digital images.* IEEE TPAMI 33(8),
  1646–1658, 2011.
* O. Delgado-Friedrichs, V. Robins, A. P. Sheppard. *Skeletonization and
  partitioning of digital images using discrete Morse theory.* IEEE TPAMI 37(3),
  654–666, 2015.
* G. Bertrand. *Morse sequences: a simple approach to discrete Morse theory.*
  JMIV 67(16), 2025. arXiv:2410.14227.
* G. Bertrand. *Morse sequences on stacks and flooding sequences.*
  arXiv:2509.01384.
* R. Forman. *Morse theory for cell complexes.* Adv. Math. 134, 90–145, 1998.

## Citing

This library accompanies a research internship report on Morse sequences and
image segmentation (ESIEE Paris, Université Gustave Eiffel, LIGM, 2026),
supervised by Gilles Bertrand, Jean Cousty and Laurent Najman.

## License

MIT — see [LICENSE](LICENSE).
