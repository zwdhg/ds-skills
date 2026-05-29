"""几何与运动学工具测试。"""

import unittest

from c2sim.geometry import (
    Vec3,
    closest_point_of_approach,
    lead_intercept_time,
)


class TestVec3(unittest.TestCase):
    def test_arithmetic(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        self.assertEqual((a + b).as_tuple(), (5, 7, 9))
        self.assertEqual((b - a).as_tuple(), (3, 3, 3))
        self.assertEqual((a * 2).as_tuple(), (2, 4, 6))
        self.assertEqual((2 * a).as_tuple(), (2, 4, 6))

    def test_norm_and_unit(self):
        v = Vec3(3, 4, 0)
        self.assertAlmostEqual(v.norm(), 5.0)
        self.assertAlmostEqual(v.unit().norm(), 1.0)

    def test_zero_unit_is_safe(self):
        self.assertEqual(Vec3().unit().as_tuple(), (0, 0, 0))


class TestCPA(unittest.TestCase):
    def test_head_on_closes_to_zero(self):
        # 两点相向而行,必在中途相遇,最近距离≈0。
        t, d = closest_point_of_approach(
            Vec3(-100, 0, 0), Vec3(10, 0, 0), Vec3(100, 0, 0), Vec3(-10, 0, 0)
        )
        self.assertAlmostEqual(d, 0.0, places=6)
        self.assertAlmostEqual(t, 10.0, places=6)

    def test_parallel_constant_distance(self):
        t, d = closest_point_of_approach(
            Vec3(0, 0, 0), Vec3(5, 0, 0), Vec3(0, 50, 0), Vec3(5, 0, 0)
        )
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(d, 50.0)

    def test_receding_clamped_to_now(self):
        # 已经在分离:最近时刻为现在(t=0)。
        t, d = closest_point_of_approach(
            Vec3(0, 0, 0), Vec3(10, 0, 0), Vec3(-100, 0, 0), Vec3(-10, 0, 0)
        )
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(d, 100.0)


class TestLeadIntercept(unittest.TestCase):
    def test_stationary_target(self):
        # 静止目标在 1000m 外,弹速 100 → 10s 命中。
        t = lead_intercept_time(
            Vec3(0, 0, 0), Vec3(1000, 0, 0), Vec3(0, 0, 0), 100.0
        )
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 10.0, places=6)

    def test_intercept_point_matches(self):
        shooter = Vec3(0, 0, 0)
        tpos = Vec3(1000, 0, 0)
        tvel = Vec3(0, 50, 0)
        speed = 200.0
        t = lead_intercept_time(shooter, tpos, tvel, speed)
        self.assertIsNotNone(t)
        # 命中点处:弹飞行距离 == speed*t。
        aim = tpos + tvel * t
        self.assertAlmostEqual(shooter.distance_to(aim), speed * t, places=4)

    def test_unreachable_returns_none(self):
        # 目标背向远离且比弹快:追不上。
        t = lead_intercept_time(
            Vec3(0, 0, 0), Vec3(1000, 0, 0), Vec3(500, 0, 0), 100.0
        )
        self.assertIsNone(t)


if __name__ == "__main__":
    unittest.main()
