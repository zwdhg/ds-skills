"""拦截指令生成(火力-目标分配)测试。"""

import unittest

from c2sim.geometry import Vec3
from c2sim.interception import EngagementPolicy, plan_and_fire
from c2sim.models import CommandKind, ThreatAssessment, ThreatLevel, Track
from c2sim.weapons import LaunchPad


def _track(tid, pos, vel):
    return Track(track_id=tid, position=pos, velocity=vel, last_update=0.0, hits=5)


def _assess(tid, score, level):
    return ThreatAssessment(
        track_id=tid,
        score=score,
        level=level,
        time_to_impact=30.0,
        closest_approach=0.0,
        rationale="",
    )


def _pad(pid, pos, inventory=4, operating_radius=5_000.0):
    return LaunchPad(pid, pos, inventory=inventory, operating_radius=operating_radius)


class TestEngagement(unittest.TestCase):
    def test_engage_fires_and_decrements_inventory(self):
        trk = _track("T1", Vec3(3_000, 0, 500), Vec3(-40, 0, 0))
        pad = _pad("P1", Vec3(0, 0, 0))
        cmds, itcs = plan_and_fire(
            [_assess("T1", 0.6, ThreatLevel.HIGH)],
            {"T1": trk}, [pad], EngagementPolicy(), now=0.0, engaged_counts={},
        )
        self.assertEqual(len(itcs), 1)
        self.assertEqual(cmds[0].kind, CommandKind.ENGAGE)
        self.assertEqual(pad.inventory, 3)
        self.assertEqual(cmds[0].pad_id, "P1")

    def test_below_threshold_not_engaged(self):
        trk = _track("T1", Vec3(3_000, 0, 500), Vec3(-40, 0, 0))
        pad = _pad("P1", Vec3(0, 0, 0))
        _, itcs = plan_and_fire(
            [_assess("T1", 0.2, ThreatLevel.LOW)],
            {"T1": trk}, [pad], EngagementPolicy(), now=0.0, engaged_counts={},
        )
        self.assertEqual(itcs, [])
        self.assertEqual(pad.inventory, 4)

    def test_already_engaged_not_reengaged(self):
        trk = _track("T1", Vec3(3_000, 0, 500), Vec3(-40, 0, 0))
        pad = _pad("P1", Vec3(0, 0, 0))
        _, itcs = plan_and_fire(
            [_assess("T1", 0.6, ThreatLevel.HIGH)],
            {"T1": trk}, [pad], EngagementPolicy(), now=0.0, engaged_counts={"T1": 1},
        )
        self.assertEqual(itcs, [])

    def test_critical_salvo_two(self):
        trk = _track("T1", Vec3(3_000, 0, 500), Vec3(-40, 0, 0))
        pad = _pad("P1", Vec3(0, 0, 0), inventory=8)
        ec = {}
        _, itcs = plan_and_fire(
            [_assess("T1", 0.9, ThreatLevel.CRITICAL)],
            {"T1": trk}, [pad], EngagementPolicy(), now=0.0, engaged_counts=ec,
        )
        self.assertEqual(len(itcs), 2)
        self.assertEqual(ec["T1"], 2)

    def test_beyond_operating_radius_yields_hold(self):
        # 拦截点远超作业半径 → 无法解算 → HOLD,库存不变。
        trk = _track("T1", Vec3(50_000, 0, 500), Vec3(-40, 0, 0))
        pad = _pad("P1", Vec3(0, 0, 0), operating_radius=5_000.0)
        cmds, itcs = plan_and_fire(
            [_assess("T1", 0.6, ThreatLevel.HIGH)],
            {"T1": trk}, [pad], EngagementPolicy(), now=0.0, engaged_counts={},
        )
        self.assertEqual(itcs, [])
        self.assertEqual(cmds[0].kind, CommandKind.HOLD)
        self.assertEqual(pad.inventory, 4)

    def test_prefers_pad_with_more_inventory(self):
        trk = _track("T1", Vec3(2_000, 0, 300), Vec3(-30, 0, 0))
        low = _pad("LOW", Vec3(500, 0, 0), inventory=1)
        high = _pad("HIGH", Vec3(-500, 0, 0), inventory=8)
        cmds, _ = plan_and_fire(
            [_assess("T1", 0.6, ThreatLevel.HIGH)],
            {"T1": trk}, [low, high], EngagementPolicy(), now=0.0, engaged_counts={},
        )
        self.assertEqual(cmds[0].pad_id, "HIGH")


if __name__ == "__main__":
    unittest.main()
