"""威胁研判测试。"""

import unittest

from c2sim.geometry import Vec3
from c2sim.models import TargetKind, ThreatLevel, Track
from c2sim.threat import ThreatPolicy, assess, assess_track, classify_track


def _track(pos, vel, hits=5, cls=None):
    return Track(
        track_id="T",
        position=pos,
        velocity=vel,
        last_update=0.0,
        hits=hits,
        coast_time=0.0,
        classification=cls,
    )


class TestClassification(unittest.TestCase):
    def test_eo_classification_preferred(self):
        # 航迹已有光电识别结果时,优先采用,而非运动学猜测。
        trk = _track(Vec3(0, 0, 500), Vec3(5, 0, 0), cls=TargetKind.LOITERING_MUNITION)
        self.assertEqual(classify_track(trk), TargetKind.LOITERING_MUNITION)

    def test_kinematic_fixed_wing(self):
        self.assertEqual(
            classify_track(_track(Vec3(0, 0, 500), Vec3(45, 0, 0))),
            TargetKind.FIXED_WING_UAV,
        )

    def test_kinematic_rotary(self):
        self.assertEqual(
            classify_track(_track(Vec3(0, 0, 200), Vec3(15, 0, 0))),
            TargetKind.ROTARY_UAV,
        )

    def test_slow_or_unestimated_defaults_to_rotary(self):
        # 低速/速度未估计(含新生航迹 velocity≈0)→ 保守按旋翼,不落最低杀伤 MICRO。
        self.assertEqual(
            classify_track(_track(Vec3(0, 0, 100), Vec3(4, 0, 0))),
            TargetKind.ROTARY_UAV,
        )
        self.assertEqual(
            classify_track(_track(Vec3(0, 0, 100), Vec3())),  # 零速新航迹
            TargetKind.ROTARY_UAV,
        )


class TestAssessment(unittest.TestCase):
    def setUp(self):
        self.asset = Vec3(0, 0, 0)
        self.policy = ThreatPolicy()

    def test_inbound_more_threatening_than_receding(self):
        inbound = _track(Vec3(6_000, 0, 500), Vec3(-45, 0, 0))
        receding = _track(Vec3(6_000, 0, 500), Vec3(45, 0, 0))
        a_in = assess_track(inbound, self.asset, self.policy)
        a_out = assess_track(receding, self.asset, self.policy)
        self.assertGreater(a_in.score, a_out.score)
        self.assertIsNotNone(a_in.time_to_impact)

    def test_direct_inbound_has_impact_time(self):
        trk = _track(Vec3(7_000, 0, 0), Vec3(-50, 0, 0))
        a = assess_track(trk, self.asset, self.policy)
        self.assertIsNotNone(a.time_to_impact)
        self.assertGreaterEqual(a.level, ThreatLevel.MEDIUM)

    def test_crossing_target_no_impact(self):
        trk = _track(Vec3(0, 8_000, 500), Vec3(50, 0, 0))
        a = assess_track(trk, self.asset, self.policy)
        self.assertIsNone(a.time_to_impact)

    def test_low_confidence_track_discounted(self):
        good = _track(Vec3(6_000, 0, 0), Vec3(-50, 0, 0), hits=8)
        weak = _track(Vec3(6_000, 0, 0), Vec3(-50, 0, 0), hits=1)
        self.assertGreater(
            assess_track(good, self.asset, self.policy).score,
            assess_track(weak, self.asset, self.policy).score,
        )

    def test_assess_sorts_descending(self):
        far = _track(Vec3(9_000, 8_000, 600), Vec3(20, 0, 0))
        near = _track(Vec3(3_000, 0, 0), Vec3(-55, 0, 0))
        far.track_id = "FAR"
        near.track_id = "NEAR"
        results = assess([far, near], self.asset, self.policy)
        self.assertEqual(results[0].track_id, "NEAR")
        self.assertGreaterEqual(results[0].score, results[1].score)


if __name__ == "__main__":
    unittest.main()
