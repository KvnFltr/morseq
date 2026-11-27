# morseq

Discrete Morse sequences and topological image segmentation, in pure Python.

`morseq` implements the segmentation layer of discrete Morse theory — stable
and unstable sets, the partition of vertices by minima, hermetic restrictions,
and **basins** — in the framework of *Morse sequences*, where acyclicity of the
gradient holds by construction.

The library is generic: every algorithm uses only the incidence relation, so it
runs on cubical complexes of **any dimension** and on simplicial complexes
alike. Grayscale data with **plateaus** are handled without modifying the
values: ties are broken by a fixed vertex order, used only to compare vertices
of equal value.

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

Python 3.9 or later. The core library has no dependencies.

```bash
pip install "morseq @ git+https://github.com/KvnFltr/morseq"
pip install "morseq[plot] @ git+https://github.com/KvnFltr/morseq"   # + matplotlib, numpy, pillow
```

From source, with the test suite:

```bash
git clone https://github.com/KvnFltr/morseq
cd morseq
python -m pip install -e ".[dev]"
python -m pytest -q
```

On Windows, if `pip` warns that the scripts directory is not on `PATH`, use
`python -m morseq` instead of `morseq`.

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
| Basins | `basin` | DFRS 2015, Sec. 3.3 |
| **Basins, one-pass labeling** | `basins`, `basin_sets` | this work |
| Bridges (= stable sets of critical edges) | `bridges` | DFRS 2015, Sec. 3.3; this work |
| Cancellation of a critical pair | `cancel` | Forman 1998, Sec. 11 |
| Stacks compatible with a cancellation | `raise_stack`, `lower_stack` | this work |
| Simplification of minima | `simplify_minima` | this work |
| Stacks, cuts, F-sequences | `max_extension`, `cut`, `is_f_sequence` | Bertrand, arXiv:2509.01384 |
| Flooding order | `flooding_order`, `is_flooding_sequence` | Bertrand, arXiv:2509.01384 |
| Random Morse sequences | `random_morse_sequence` | — |
| Image segmentation, 2-D/3-D | `segment_image` | this work |

## Gradient from grayscale data: relation to Robins, Wood, Sheppard

`gradient_from_vertex_map` is Algorithm 1 (`ProcessLowerStars`) of Robins, Wood
and Sheppard (RWS 2011), with four implementation changes that do not change
its output on cubical and simplicial complexes:

1. ties between equal values are broken by a fixed vertex order, which plays
   the role of the small ramp added to the data in RWS (Eq. 8);
2. a cell enters the queue `PQone` when exactly one of its **facets** is
   unclassified, where RWS count all its faces: since a classified cell always
   has all its faces classified and intervals of the face poset are Boolean
   lattices, both counts are simultaneously 0, 1 or at least 2;
3. after each classification the lower star is rescanned, instead of visiting
   only the cofaces of the cells just classified (only those can change);
4. classified cells are skipped when popped from `PQzero` instead of being
   removed from it.

`tests/test_rws_equivalence.py` contains a line-by-line transcription of RWS
Algorithm 1 and checks that both return the same field on 2-D and 3-D images
with plateaus and on a simplicial complex.

Every pair lies in a single lower star, where all cells share the value of the
star's vertex. The field is therefore an `F`-sequence field for the
max-extension `F`, and `segment_image` orders it as a **flooding sequence**:
each grayscale cut is a prefix of the sequence.

## The basin algorithm

A basin, in the sense of DFRS, is the maximal subcomplex that regularly
collapses onto a minimum; DFRS prove that it is unique (Lemma 6). Equivalently,
it is the **closure of the minimum under regular expansions**: repeatedly add
any pair of the gradient field whose required faces are already present. The
result does not depend on the order of the additions.

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

## Simplification

Cancelling a minimum with a saddle (Forman's cancellation) reverses a gradient
path, and the new pairs usually join cells of different values: the field is
no longer an `F`-sequence field. It becomes one again for an explicit stack,
`raise_stack`, which fills the region of the minimum below the saddle up to
the saddle level (dually, `lower_stack` cuts a maximum down to a saddle).
Cancelling a minimum with a saddle merges exactly two basins.

```python
from morseq import simplify_minima
V2, F2, n = simplify_minima(K, V, F, threshold=16)   # persistence < 16 removed
```

## Performance

The pipeline of `segment_image` (lower-star gradient, flooding sequence,
basin labeling) is linear in the number of cells, up to the logarithmic factor
of the scheduling heap. It is written in pure Python, so it is a reference
implementation rather than a fast one. `examples/benchmark.py` measures it on
random 8-bit images; on a laptop (Python 3.13):

| image | cells | time |
|---|---|---|
| 64 × 64 | 16 129 | 0.3 s |
| 128 × 128 | 65 025 | 1.5 s |
| 256 × 256 | 261 121 | 8.4 s |

## Verification

Every theorem is an executable test. The suite checks, on 2-D and 3-D cubical
grids, on simplicial complexes, on random acyclic fields and on random Morse
sequences:

* stable and unstable sets equal the **literal recursive definitions** of DFRS;
* the parity characterization (`count_gradient_paths` vs the reference map) —
  Theorem 12 — holds for every cell and every critical cell;
* the set given by path parity is contained in the set given by path
  existence, and the inclusion is strict on an explicit example;
* every vertex has a unique maximal path, and classes partition the vertices;
* prefixes and stack cuts are hermetic, and stable sets restrict to them;
* the basin closure is independent of the expansion order;
* the one-pass algorithm returns exactly the closure;
* each basin is a hermetic subcomplex, contains a single critical cell,
  collapses regularly onto its minimum, admits no further available pair, and
  meets the vertices in the stable set;
* basins are pairwise disjoint and restrict to prefixes and cuts;
* the critical and cocritical complexes satisfy `∂∘∂ = 0` and `δ∘δ = 0`, and
  the Euler characteristic of the critical complex matches that of the complex;
* `gradient_from_vertex_map` returns the same field as RWS Algorithm 1;
* `segment_image` returns a flooding sequence whose prefixes include every
  grayscale cut.

```bash
python -m pytest -q                  # unit tests and doctests
python -m morseq check --shape 5x4   # same statements, on the command line
python -m morseq check --shape 3x3x3 # in dimension three
```

## Command line

```bash
python -m morseq segment image.png --labels labels.json --plot out.png
python -m morseq check --shape 6x5 --trials 50
```

(The `morseq` command is also installed as an executable; `python -m morseq`
avoids any `PATH` configuration.)

## Paper

The statements implemented here are proved in the paper in [`paper/`](paper/)
(LNCS format, source and PDF), whose experiments are reproducible with the
scripts of [`paper/experiments/`](paper/experiments/). To cite the software,
see [`CITATION.cff`](CITATION.cff).

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

## License

MIT — see [LICENSE](LICENSE).
