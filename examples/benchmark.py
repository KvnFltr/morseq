"""Running time of segment_image on random images of increasing size.

    python examples/benchmark.py            # 2-D, up to 256 x 256
    python examples/benchmark.py --3d       # 3-D, up to 24^3

The time per cell should stay roughly constant: the pipeline is linear in the
number of cells, up to the logarithmic factor of the scheduling heap.
"""
import random
import sys
import time

from morseq import segment_image


def main() -> None:
    three_d = "--3d" in sys.argv
    sizes = (8, 12, 16, 20, 24) if three_d else (32, 64, 128, 256)
    print(f"{'size':>10} {'cells':>10} {'time (s)':>9} {'us/cell':>8} {'basins':>7}")
    for n in sizes:
        rng = random.Random(0)
        if three_d:
            img = [[[rng.randrange(256) for _ in range(n)] for _ in range(n)]
                   for _ in range(n)]
            label = f"{n}^3"
        else:
            img = [[rng.randrange(256) for _ in range(n)] for _ in range(n)]
            label = f"{n}x{n}"
        t = time.perf_counter()
        seg = segment_image(img)
        dt = time.perf_counter() - t
        cells = len(seg.K)
        print(f"{label:>10} {cells:>10} {dt:>9.2f} {1e6 * dt / cells:>8.1f} "
              f"{len(seg.basin_sets()):>7}", flush=True)


if __name__ == "__main__":
    main()
