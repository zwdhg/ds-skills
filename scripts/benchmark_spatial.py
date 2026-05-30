"""空间索引加速基准:网格最近邻 vs 暴力遍历。

    python scripts/benchmark_spatial.py [N] [Q]

演示 :class:`c2sim.spatial.SpatialGrid` 在大规模目标下相对暴力遍历的加速,
佐证体系在保持接口不变的前提下可扩展到蜂群规模。
"""

from __future__ import annotations

import random
import sys
import time

from c2sim.geometry import Vec3
from c2sim.spatial import SpatialGrid


def _brute_nearest(pts, c):
    best, best_d = None, float("inf")
    for p in pts:
        d = c.distance_to(p)
        if d < best_d:
            best, best_d = p, d
    return best


def main(argv: list[str]) -> int:
    n = int(argv[0]) if len(argv) > 0 else 4000
    q = int(argv[1]) if len(argv) > 1 else 2000
    rng = random.Random(0)

    def pt():
        return Vec3(rng.uniform(-15000, 15000), rng.uniform(-15000, 15000),
                    rng.uniform(0, 3000))

    pts = [pt() for _ in range(n)]
    queries = [pt() for _ in range(q)]

    grid = SpatialGrid(cell=1000.0)
    for p in pts:
        grid.insert(p, p)

    t = time.perf_counter()
    for c in queries:
        grid.nearest(c)
    t_grid = time.perf_counter() - t

    t = time.perf_counter()
    for c in queries:
        _brute_nearest(pts, c)
    t_brute = time.perf_counter() - t

    print(f"N={n} 目标, Q={q} 次最近邻查询")
    print(f"  网格索引:{t_grid * 1000:7.0f} ms")
    print(f"  暴力遍历:{t_brute * 1000:7.0f} ms")
    print(f"  加速比  :{t_brute / t_grid:6.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
