"""Random Morse sequences, for tests and experiments.

A random field obtained by matching random facet pairs is rarely acyclic as
soon as the complex is three-dimensional. Building the sequence step by step
avoids the problem: each step is an elementary filling or an elementary
expansion of the current subcomplex, so the induced field is acyclic by
construction.
"""
from __future__ import annotations

import random as _random
from typing import List, Optional

from .complex import Cell, Complex
from .sequence import MorseSequence


def random_morse_sequence(K: Complex, seed: Optional[int] = None,
                          p_expand: float = 0.8) -> MorseSequence:
    """A random Morse sequence on ``K``.

    At each step, an available expansion is chosen with probability
    ``p_expand`` (when there is one), otherwise an available filling.
    Intended for small complexes: each step scans the complex.

    >>> from morseq import cubical_complex
    >>> K = cubical_complex((3, 3, 3))
    >>> seq = random_morse_sequence(K, seed=1)
    >>> seq.validate()
    """
    rng = _random.Random(seed)
    present: set = set()
    items: List[object] = []
    while len(present) < len(K):
        fillings: List[Cell] = []
        expansions: List[tuple] = []
        for c in K.cells:
            if c in present:
                continue
            if all(f in present for f in K.facets[c]):
                fillings.append(c)
                for t in K.cofacets[c]:
                    if t not in present and all(
                            f in present for f in K.facets[t] if f != c):
                        expansions.append((c, t))
        if expansions and (not fillings or rng.random() < p_expand):
            s, t = rng.choice(expansions)
            items.append((s, t))
            present.update((s, t))
        else:
            c = rng.choice(fillings)
            items.append(c)
            present.add(c)
    return MorseSequence(K, items)
