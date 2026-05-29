"""航迹融合测试。"""

import unittest

from c2sim.fusion import TrackFusion, fuse_reports
from c2sim.geometry import Vec3
from c2sim.models import SensorModality, SensorReport, TargetKind


def _rep(sid, t, pos, sigma=25.0, modality=SensorModality.RADAR, cls=None):
    return SensorReport(
        sensor_id=sid,
        modality=modality,
        timestamp=t,
        position=pos,
        position_sigma=sigma,
        classification=cls,
    )


class TestMultiSensorFusion(unittest.TestCase):
    def test_colocated_reports_merge(self):
        reports = [
            _rep("A", 0.0, Vec3(1000, 0, 0)),
            _rep("B", 0.0, Vec3(1010, 5, 0)),
        ]
        fused = fuse_reports(reports)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].sensors, {"A", "B"})
        self.assertLess(fused[0].sigma, 25.0)

    def test_eo_pulls_fused_position(self):
        # 高精度光电量测(小 σ)应主导融合位置。
        radar = _rep("R", 0.0, Vec3(1100, 0, 0), sigma=200.0)
        eo = _rep("E", 0.0, Vec3(1000, 0, 0), sigma=5.0, modality=SensorModality.EO)
        fused = fuse_reports([radar, eo])
        self.assertEqual(len(fused), 1)
        self.assertLess(abs(fused[0].position.x - 1000.0), 10.0)

    def test_eo_classification_propagates(self):
        eo = _rep(
            "E", 0.0, Vec3(1000, 0, 0), sigma=5.0,
            modality=SensorModality.EO, cls=TargetKind.LOITERING_MUNITION,
        )
        fused = fuse_reports([eo])
        self.assertEqual(fused[0].classification, TargetKind.LOITERING_MUNITION)

    def test_distant_reports_separate(self):
        reports = [
            _rep("A", 0.0, Vec3(0, 0, 0)),
            _rep("A", 0.0, Vec3(50_000, 0, 0)),
        ]
        self.assertEqual(len(fuse_reports(reports)), 2)


class TestTrackLifecycle(unittest.TestCase):
    def test_track_creation_and_velocity_estimate(self):
        fusion = TrackFusion()
        for i in range(8):
            t = i * 1.0
            fusion.update([_rep("A", t, Vec3(50.0 * i, 0, 0))], t)
        tracks = list(fusion.tracks.values())
        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0].velocity.x, 50.0, delta=5.0)
        self.assertGreater(tracks[0].confidence, 0.5)

    def test_classification_recorded_on_track(self):
        fusion = TrackFusion()
        fusion.update(
            [_rep("E", 0.0, Vec3(0, 0, 0), modality=SensorModality.EO,
                  cls=TargetKind.FIXED_WING_UAV)],
            0.0,
        )
        trk = next(iter(fusion.tracks.values()))
        self.assertEqual(trk.classification, TargetKind.FIXED_WING_UAV)
        self.assertIn(SensorModality.EO, trk.modalities)

    def test_coasted_track_deleted(self):
        fusion = TrackFusion(max_coast=3.0)
        fusion.update([_rep("A", 0.0, Vec3(0, 0, 0))], 0.0)
        self.assertEqual(len(fusion.tracks), 1)
        fusion.update([], 10.0)
        self.assertEqual(len(fusion.tracks), 0)


if __name__ == "__main__":
    unittest.main()
