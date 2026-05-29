"""威胁研判测试。"""

import unittest

from c2sim.geometry import Vec3
from c2sim.models import TargetKind, ThreatLevel, Track
from c2sim.threat import ThreatPolicy, assess, assess_track, classify_track


def _track(pos, vel, hits=5):
    return Track(
        track_id="T",
        position=pos,
        velocity=vel,
        last_update=0.0,
        hits=hits,
        coast_time=0.0,
    )


class TestClassification(unittest.TestCase):
    def test_ballistic(self):
        self.assertEqual(
            classify_track(_track(Vec3(0, 0, 50000), Vec3(2000, 0, -500))),
            TargetKind.BALLISTIC,
        )

    def test_cruise_missile_low_fast(self):
        self.assertEqual(
            classify_track(_track(Vec3(0, 0, 100), Vec3(300, 0, 0))),
            TargetKind.CRUISE_MISSILE,
        )

    def test_drone_slow(self):
        self.assertEqual(
            classify_track(_track(Vec3(0, 0, 500), Vec3(40, 0, 0))),
            TargetKind.DRONE,
        )

    def test_aircraft_default(self):
        self.assertEqual(
            classify_track(_track(Vec3(0, 0, 8000), Vec3(200, 0, 0))),
            TargetKind.AIRCRAFT,
        )


class TestAssessment(unittest.TestCase):
    def setUp(self):
        self.asset = Vec3(0, 0, 0)
        self.policy = ThreatPolicy()

    def test_inbound_more_threatening_than_receding(self):
        inbound = _track(Vec3(30_000, 0, 5000), Vec3(-250, 0, 0))
        receding = _track(Vec3(30_000, 0, 5000), Vec3(250, 0, 0))
        a_in = assess_track(inbound, self.asset, self.policy)
        a_out = assess_track(receding, self.asset, self.policy)
        self.assertGreater(a_in.score, a_out.score)
        self.assertIsNotNone(a_in.time_to_impact)

    def test_direct_inbound_has_impact_time(self):
        # 正对要地飞来 → 会穿透防护半径,有抵达时间。
        trk = _track(Vec3(50_000, 0, 0), Vec3(-500, 0, 0))
        a = assess_track(trk, self.asset, self.policy)
        self.assertIsNotNone(a.time_to_impact)
        self.assertGreaterEqual(a.level, ThreatLevel.MEDIUM)

    def test_crossing_target_no_impact(self):
        # 远距离横穿、不指向要地 → 不穿透,无抵达时间。
        trk = _track(Vec3(0, 80_000, 8000), Vec3(300, 0, 0))
        a = assess_track(trk, self.asset, self.policy)
        self.assertIsNone(a.time_to_impact)

    def test_low_confidence_track_discounted(self):
        good = _track(Vec3(30_000, 0, 0), Vec3(-300, 0, 0), hits=8)
        weak = _track(Vec3(30_000, 0, 0), Vec3(-300, 0, 0), hits=1)
        a_good = assess_track(good, self.asset, self.policy)
        a_weak = assess_track(weak, self.asset, self.policy)
        self.assertGreater(a_good.score, a_weak.score)

    def test_assess_sorts_descending(self):
        tracks = [
            _track(Vec3(120_000, 60_000, 9000), Vec3(100, 0, 0)),  # 远、掠过
            _track(Vec3(20_000, 0, 0), Vec3(-400, 0, 0)),          # 近、直扑
        ]
        tracks[0].track_id = "FAR"
        tracks[1].track_id = "NEAR"
        results = assess(tracks, self.asset, self.policy)
        self.assertEqual(results[0].track_id, "NEAR")
        self.assertGreaterEqual(results[0].score, results[1].score)


if __name__ == "__main__":
    unittest.main()
