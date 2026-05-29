"""航迹融合测试。"""

import unittest

from c2sim.fusion import TrackFusion, fuse_reports
from c2sim.geometry import Vec3
from c2sim.models import SensorReport


def _rep(sid, t, pos, sigma=25.0):
    return SensorReport(sensor_id=sid, timestamp=t, position=pos, position_sigma=sigma)


class TestMultiSensorFusion(unittest.TestCase):
    def test_colocated_reports_merge(self):
        # 两部传感器对同一目标的邻近报告 → 单个融合量测。
        reports = [
            _rep("A", 0.0, Vec3(1000, 0, 0)),
            _rep("B", 0.0, Vec3(1010, 5, 0)),
        ]
        fused = fuse_reports(reports)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].sensors, {"A", "B"})
        # 融合后精度优于任一单部(σ 下降)。
        self.assertLess(fused[0].sigma, 25.0)

    def test_distant_reports_separate(self):
        reports = [
            _rep("A", 0.0, Vec3(0, 0, 0)),
            _rep("A", 0.0, Vec3(50_000, 0, 0)),
        ]
        fused = fuse_reports(reports)
        self.assertEqual(len(fused), 2)


class TestTrackLifecycle(unittest.TestCase):
    def test_track_creation_and_velocity_estimate(self):
        fusion = TrackFusion()
        # 目标以 100 m/s 沿 x 轴运动,逐帧喂入(无噪)。
        for i in range(8):
            t = i * 1.0
            fusion.update([_rep("A", t, Vec3(100.0 * i, 0, 0))], t)
        tracks = list(fusion.tracks.values())
        self.assertEqual(len(tracks), 1)
        trk = tracks[0]
        # α-β 滤波应收敛到约 100 m/s。
        self.assertAlmostEqual(trk.velocity.x, 100.0, delta=5.0)
        self.assertGreater(trk.confidence, 0.5)

    def test_single_track_for_moving_target(self):
        # 匀速目标不应分裂出多条航迹。
        fusion = TrackFusion()
        for i in range(10):
            t = i * 0.5
            fusion.update([_rep("A", t, Vec3(200.0 * t, 50.0 * t, 0))], t)
        self.assertEqual(len(fusion.tracks), 1)

    def test_coasted_track_deleted(self):
        fusion = TrackFusion(max_coast=3.0)
        fusion.update([_rep("A", 0.0, Vec3(0, 0, 0))], 0.0)
        self.assertEqual(len(fusion.tracks), 1)
        # 之后长时间无量测 → 撤销。
        fusion.update([], 10.0)
        self.assertEqual(len(fusion.tracks), 0)


if __name__ == "__main__":
    unittest.main()
