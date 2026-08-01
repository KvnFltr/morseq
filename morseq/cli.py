"""Command-line interface: ``morseq segment image.png`` and friends."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List


def _load_image(path: str):
    if path.endswith((".json", ".txt")):
        with open(path) as fh:
            return json.load(fh)
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=int)
    except ImportError:  # pragma: no cover
        raise SystemExit("reading images requires pillow and numpy: "
                         "pip install 'morseq[plot]' pillow")


def cmd_segment(args: argparse.Namespace) -> int:
    from .stacks import segment_image

    image = _load_image(args.image)
    seg = segment_image(image)
    print(seg.summary())
    if args.labels:
        labels = seg.vertex_labels()
        with open(args.labels, "w") as fh:
            json.dump(labels, fh)
        print(f"vertex labels written to {args.labels}")
    if args.plot:
        from .plotting import plot_segmentation
        plot_segmentation(seg, args.plot)
        print(f"figure written to {args.plot}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run the statements of the paper on a complex given on the command line."""
    import random

    from .complex import cubical_complex
    from .sequence import sequence_from_field, is_acyclic
    from .segmentation import (basin, basin_sets, stable_set, is_hermetic,
                               regular_collapse_order, requirements,
                               vertex_partition)

    shape = tuple(int(x) for x in args.shape.split("x"))
    K = cubical_complex(shape)
    checked = 0
    for seed in range(args.trials):
        rng = random.Random(seed)
        cand = [(f, c) for c in K.cells for f in K.facets[c]]
        rng.shuffle(cand)
        used, V = set(), {}
        for s, t in cand:
            if s in used or t in used or rng.random() < 0.25:
                continue
            V[s] = t
            used.update((s, t))
        if not is_acyclic(K, V):
            continue
        seq = sequence_from_field(K, V)
        seq.validate()
        sets = basin_sets(seq)
        for m in [c for c in seq.critical if K.dim[c] == 0]:
            B = sets.get(m, {m})
            assert B == basin(seq, m)
            assert is_hermetic(seq, B)
            assert {c for c in B if K.dim[c] == 0} == stable_set(seq, m)
            regular_collapse_order(seq, B, m)
            for s, t in seq.V.items():
                if s not in B:
                    assert not requirements(K, s, t) <= B
        part = vertex_partition(seq)
        assert set(part) == set(K.cells_of_dim(0))
        checked += 1
    print(f"{checked} acyclic fields on {shape}: all statements verified")
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="morseq",
                                description="Morse sequences and topological segmentation")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("segment", help="segment a grayscale image")
    s.add_argument("image", help="PNG/JPG file, or JSON array of integers")
    s.add_argument("--labels", help="write the vertex labels to this JSON file")
    s.add_argument("--plot", help="write a figure to this PNG file")
    s.set_defaults(func=cmd_segment)

    c = sub.add_parser("check", help="verify the theorems on random fields")
    c.add_argument("--shape", default="5x4", help="grid shape, e.g. 5x4 or 3x3x3")
    c.add_argument("--trials", type=int, default=20)
    c.set_defaults(func=cmd_check)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
