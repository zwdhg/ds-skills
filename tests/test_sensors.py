"""Spotter Pro 多模态探测测试。"""

import random
import unittest

from c2sim.geometry import Vec3
from c2sim.models import SensorModality, TargetKind
from c2sim.sensors import SpotterPro


def _target(tid, pos, rcs=0.2, emits_rf=True, kind=TargetKind.FIXED_WING_UAV):
    from c2sim.models import Target

    return Target(
        target_id=tid,
        position=pos,
        aim=Vec3(0, 0, 0),
        cruise_speed=40.0,
        kind=kind,
        rcs=rcs,
        emits_rf=emits_rf,
    )


def _spotter(**kw):
    # 探测概率拉满,便于确定性断言。
    defaults = dict(rf_detect_prob=1.0, radar_detect_prob=1.0, eo_classify_prob=1.0)
    defaults.update(kw)
    return SpotterPro("SPT", Vec3(0, 0, 20.0), **defaults)


class TestRadarRange(unittest.TestCase):
    def test_range_calibration(self):
        s = _spotter()
        self.assertAlmostEqual(s.radar_range_for(0.2), 10_000.0, delta=50.0)
        self.assertAlmostEqual(s.radar_range_for(0.01), 5_000.0, delta=200.0)

    def test_blind_zone(self):
        s = _spotter()
        rng = random.Random(0)
        reps = s.observe([_target("T", Vec3(100, 0, 20))], 0.0, rng)
        self.assertEqual(reps, [])  # 落在近界盲区内

    def test_out_of_sector_not_detected(self):
        s = _spotter(azimuth_center_deg=0.0, azimuth_width_deg=180.0)
        rng = random.Random(0)
        # 目标在 -x 方向(扇区背后)。
        reps = s.observe([_target("T", Vec3(-4000, 0, 200))], 0.0, rng)
        self.assertEqual(reps, [])


class TestModalities(unittest.TestCase):
    def test_rf_silent_target_no_rf_but_radar_tracks(self):
        s = _spotter()
        rng = random.Random(1)
        reps = s.observe(
            [_target("T", Vec3(4000, 0, 200), emits_rf=False)], 0.0, rng
        )
        mods = {r.modality for r in reps}
        self.assertIn(SensorModality.RADAR, mods)  # 雷达仍能探测
        # RF 静默:不应有任何 RF 模态报告(本模型中 RF 仅作引导,不产报告)
        self.assertNotIn(SensorModality.RF, mods)

    def test_eo_provides_classification(self):
        s = _spotter()
        rng = random.Random(2)
        reps = s.observe(
            [_target("T", Vec3(4000, 0, 200), kind=TargetKind.LOITERING_MUNITION)],
            0.0, rng,
        )
        eo = [r for r in reps if r.modality == SensorModality.EO]
        self.assertEqual(len(eo), 1)
        self.assertEqual(eo[0].classification, TargetKind.LOITERING_MUNITION)


class TestCapacity(unittest.TestCase):
    def test_tas_capacity_limits_radar_tracks(self):
        s = _spotter(radar_tas_capacity=2)
        rng = random.Random(3)
        targets = [_target(f"T{i}", Vec3(3000 + 50 * i, 200 * i, 200)) for i in range(5)]
        reps = s.observe(targets, 0.0, rng)
        radar = [r for r in reps if r.modality == SensorModality.RADAR]
        self.assertEqual(len(radar), 2)

    def test_eo_single_turret(self):
        s = _spotter()
        rng = random.Random(4)
        targets = [_target(f"T{i}", Vec3(3000 + 50 * i, 300 * i, 200)) for i in range(4)]
        reps = s.observe(targets, 0.0, rng)
        eo = [r for r in reps if r.modality == SensorModality.EO]
        self.assertEqual(len(eo), 1)  # 转台同一时刻仅服务一个目标


class TestCoverage(unittest.TestCase):
    def test_point_defense_area(self):
        self.assertAlmostEqual(_spotter().coverage_area_km2(), 78.54, delta=0.1)

    def test_border_sector_area(self):
        s = _spotter(azimuth_width_deg=180.0)
        self.assertAlmostEqual(s.coverage_area_km2(), 39.27, delta=0.1)


class TestUnionCoverage(unittest.TestCase):
    def test_single_station_matches_disc(self):
        from c2sim.sensors import union_coverage_km2
        s = _spotter()
        self.assertAlmostEqual(union_coverage_km2([s]), s.coverage_area_km2(),
                               delta=1.0)

    def test_overlap_union_less_than_sum(self):
        # 两个重叠站:并集应明显小于直接相加(扣除重叠),且大于单站。
        from c2sim.sensors import union_coverage_km2
        a = _spotter()
        b = SpotterPro("B", Vec3(3000, 0, 20), rf_detect_prob=1.0,
                       radar_detect_prob=1.0, eo_classify_prob=1.0)
        union = union_coverage_km2([a, b])
        self.assertLess(union, a.coverage_area_km2() + b.coverage_area_km2())
        self.assertGreater(union, a.coverage_area_km2())


if __name__ == "__main__":
    unittest.main()
