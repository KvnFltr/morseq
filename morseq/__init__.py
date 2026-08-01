"""morseq -- discrete Morse sequences and topological image segmentation.

Implements, with executable proofs-as-tests, the translation of the
segmentation layer of Delgado-Friedrichs, Robins and Sheppard into the Morse
sequence framework of Bertrand.

Quick start
-----------
>>> from morseq import segment_image
>>> seg = segment_image([[0, 3, 1],
...                      [3, 3, 3],
...                      [1, 3, 0]])
>>> len(seg.basin_sets())
2
"""
from .complex import (Complex, cubical_complex, simplicial_complex,
                      cube_vertex, vertex_coords)
from .sequence import (MorseSequence, sequence_from_field, is_acyclic,
                       is_valid_field)
from .maps import (reference_map, coreference_map, critical_boundary,
                   cocritical_coboundary, extension, coextension,
                   count_gradient_paths)
from .segmentation import (stable_set, unstable_set, unstable_complex,
                           morse_skeleton, vertex_flow, vertex_partition,
                           is_hermetic, restrict, requirements, basin,
                           basins, basin_sets, bridges,
                           regular_collapse_order)
from .stacks import (max_extension, is_stack, cut, is_f_sequence,
                     is_flooding_sequence, gradient_from_vertex_map,
                     flooding_order, segment_image, Segmentation)

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
