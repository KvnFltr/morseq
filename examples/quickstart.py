"""Minimal example: segment a small grayscale image and print the labels."""
from morseq import segment_image

image = [[0, 3, 3, 3, 1],
         [3, 3, 5, 3, 3],
         [3, 5, 5, 5, 3],
         [1, 3, 3, 3, 0]]

seg = segment_image(image)
print(seg.summary())
print()
for row in seg.vertex_labels():
    print(" ".join(str(x) for x in row))
