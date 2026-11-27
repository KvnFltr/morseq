# Paper: Basins, Bridges and Simplification with Morse Sequences

LNCS source of the paper that accompanies `morseq`, and the scripts that
produce every number and figure in it.

```
paper/
  main.tex                 the paper (Springer LNCS class llncs)
  references.bib
  numbers.tex              generated: numbers quoted in the text
  figures/                 running_example.pdf, simplification.pdf (generated)
  experiments/
    verification.py        large-scale checks of the statements
    runtime.py             running time of the pipeline
    simplification.py      simplification example and its figure
    make_numbers.py        writes numbers.tex from results/*.json
    results/               JSON outputs and the runtime table
```

## Reproduce

From the repository root, with `morseq[dev]` installed:

```bash
python paper/experiments/verification.py     # ~10 min
python paper/experiments/runtime.py          # ~5 min, run on an idle machine
python paper/experiments/simplification.py   # ~1 min
python paper/experiments/make_numbers.py     # fails if any check failed
```

`make_numbers.py` refuses to write `numbers.tex` if one of the checks reported
in the paper did not pass.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`llncs.cls` and `splncs04.bst` are part of TeX Live and MiKTeX (package
`llncs`).
