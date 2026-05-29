"""空间索引:与暴力遍历的等价性(无损加速)。"""

import random
import unittest

from c2sim.geometry import Vec3
from c2sim.spatial import SpatialGrid


def _brute_nearest(pts, c):
    best, best_d = None, float("inf")
    for p in pts:
        d = c.distance_to(p)
        if d < best_d:
            best, best_d = p, d
    return best


class TestSpatialGrid(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(SpatialGrid().nearest(Vec3(0, 0, 0)))

    def test_nearest_matches_brute(self):
        rng = random.Random(7)
        pts = [Vec3(rng.uniform(-12000, 12000), rng.uniform(-12000, 12000),
                    rng.uniform(0, 3000)) for _ in range(300)]
        grid = SpatialGrid(cell=1000.0)
        for p in pts:
            grid.insert(p, p)
        for _ in range(60):
            c = Vec3(rng.uniform(-12000, 12000), rng.uniform(-12000, 12000),
                     rng.uniform(0, 3000))
            got = grid.nearest(c)
            exp = _brute_nearest(pts, c)
            self.assertAlmostEqual(c.distance_to(got), c.distance_to(exp), places=6)

    def test_nearest_sparse_and_external_query(self):
        # 回归:稀疏点 + 查询点落在占用区之外时,仍须与暴力一致(不得返回 None/非最近)。
        rng = random.Random(99)
        for _ in range(3000):
            grid = SpatialGrid(cell=1000.0)
            pts = [Vec3(rng.uniform(-12000, 12000), rng.uniform(-12000, 12000),
                        rng.uniform(0, 3000)) for _ in range(rng.randint(1, 4))]
            for p in pts:
                grid.insert(p, p)
            c = Vec3(rng.uniform(-20000, 20000), rng.uniform(-20000, 20000),
                     rng.uniform(0, 3000))
            got = grid.nearest(c)
            exp = _brute_nearest(pts, c)
            self.assertIsNotNone(got)
            self.assertAlmostEqual(c.distance_to(got), c.distance_to(exp), places=6)

    def test_query_radius_matches_brute(self):
        rng = random.Random(11)
        pts = [Vec3(rng.uniform(-8000, 8000), rng.uniform(-8000, 8000),
                    rng.uniform(0, 2000)) for _ in range(400)]
        grid = SpatialGrid(cell=800.0)
        for p in pts:
            grid.insert(p, p)
        c, r = Vec3(0, 0, 500), 2500.0
        got = {id(p) for p in grid.query_radius(c, r)}
        exp = {id(p) for p in pts if c.distance_to(p) <= r}
        self.assertEqual(got, exp)


if __name__ == "__main__":
    unittest.main()
