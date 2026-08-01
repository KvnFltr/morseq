"""Produce the two-panel figure of a segmentation (requires matplotlib)."""
from morseq import segment_image
from morseq.plotting import plot_segmentation

seg = segment_image([[0, 4, 4, 4, 1],
                     [4, 4, 6, 4, 4],
                     [4, 6, 6, 6, 4],
                     [1, 4, 4, 4, 0]])
plot_segmentation(seg, "segmentation.png")
print("figure written to segmentation.png")
