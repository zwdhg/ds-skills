"""几何与运动学工具测试。"""

import random
import unittest

from c2sim.geometry import (
    Vec3,
    angular_measurement_noise,
    closest_point_of_approach,
    deg2rad,
    lead_intercept_time,
    segment_cpa,
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
        t, d = closest_point_of_approach(
            Vec3(0, 0, 0), Vec3(10, 0, 0), Vec3(-100, 0, 0), Vec3(-10, 0, 0)
        )
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(d, 100.0)


class TestSegmentCPA(unittest.TestCase):
    def test_pass_within_window(self):
        # 相对位置 100m 前方,相对速度 -200 m/s,1s 内掠过到 0。
        t, d = segment_cpa(Vec3(100, 5, 0), Vec3(-200, 0, 0), 1.0)
        self.assertAlmostEqual(d, 5.0, places=6)
        self.assertGreater(t, 0.0)

    def test_clamped_to_window(self):
        # 最近时刻在 window 之外(将来),被裁剪到 dt。
        t, d = segment_cpa(Vec3(1000, 0, 0), Vec3(-10, 0, 0), 0.5)
        self.assertEqual(t, 0.5)


class TestLeadIntercept(unittest.TestCase):
    def test_stationary_target(self):
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
        aim = tpos + tvel * t
        self.assertAlmostEqual(shooter.distance_to(aim), speed * t, places=4)

    def test_unreachable_returns_none(self):
        t = lead_intercept_time(
            Vec3(0, 0, 0), Vec3(1000, 0, 0), Vec3(500, 0, 0), 100.0
        )
        self.assertIsNone(t)


class TestAngularNoise(unittest.TestCase):
    def test_zero_sigma_is_exact(self):
        rng = random.Random(0)
        target = Vec3(5000, 1000, 800)
        out = angular_measurement_noise(Vec3(), target, 0.0, 0.0, 0.0, rng)
        self.assertAlmostEqual(out.distance_to(target), 0.0, places=6)

    def test_elevation_error_dominates_altitude(self):
        # 俯仰角误差远大于方位/测距时,高度(z)误差应明显大于水平误差。
        rng = random.Random(1)
        sensor = Vec3(0, 0, 0)
        target = Vec3(8000, 0, 50)  # 远、低仰角
        zs, hs = [], []
        for _ in range(400):
            m = angular_measurement_noise(
                sensor, target, 1.0, deg2rad(0.1), deg2rad(3.0), rng
            )
            zs.append(abs(m.z - target.z))
            hs.append(abs(m.y - target.y))
        self.assertGreater(sum(zs) / len(zs), sum(hs) / len(hs))


if __name__ == "__main__":
    unittest.main()
