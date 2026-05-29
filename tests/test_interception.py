"""拦截指令生成(火力-目标分配)测试。"""

import unittest

from c2sim.geometry import Vec3
from c2sim.interception import EngagementPolicy, plan_and_fire
from c2sim.models import CommandKind, ThreatAssessment, ThreatLevel, Track
from c2sim.weapons import ThunderBattery


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


class TestEngagement(unittest.TestCase):
    def test_engage_fires_and_decrements_inventory(self):
        trk = _track("T1", Vec3(30_000, 0, 5000), Vec3(-300, 0, 0))
        bty = ThunderBattery("B1", Vec3(0, 0, 0), inventory=4)
        cmds, itcs = plan_and_fire(
            [_assess("T1", 0.6, ThreatLevel.HIGH)],
            {"T1": trk},
            [bty],
            EngagementPolicy(),
            now=0.0,
            engaged_counts={},
        )
        self.assertEqual(len(itcs), 1)
        self.assertEqual(cmds[0].kind, CommandKind.ENGAGE)
        self.assertEqual(bty.inventory, 3)
        self.assertEqual(cmds[0].battery_id, "B1")

    def test_below_threshold_not_engaged(self):
        trk = _track("T1", Vec3(30_000, 0, 5000), Vec3(-300, 0, 0))
        bty = ThunderBattery("B1", Vec3(0, 0, 0), inventory=4)
        cmds, itcs = plan_and_fire(
            [_assess("T1", 0.2, ThreatLevel.LOW)],
            {"T1": trk},
            [bty],
            EngagementPolicy(),
            now=0.0,
            engaged_counts={},
        )
        self.assertEqual(itcs, [])
        self.assertEqual(bty.inventory, 4)

    def test_already_engaged_not_reengaged(self):
        trk = _track("T1", Vec3(30_000, 0, 5000), Vec3(-300, 0, 0))
        bty = ThunderBattery("B1", Vec3(0, 0, 0), inventory=4)
        cmds, itcs = plan_and_fire(
            [_assess("T1", 0.6, ThreatLevel.HIGH)],
            {"T1": trk},
            [bty],
            EngagementPolicy(),
            now=0.0,
            engaged_counts={"T1": 1},  # 已分配一发
        )
        self.assertEqual(itcs, [])

    def test_critical_salvo_two(self):
        trk = _track("T1", Vec3(30_000, 0, 5000), Vec3(-300, 0, 0))
        bty = ThunderBattery("B1", Vec3(0, 0, 0), inventory=8)
        ec = {}
        _, itcs = plan_and_fire(
            [_assess("T1", 0.9, ThreatLevel.CRITICAL)],
            {"T1": trk},
            [bty],
            EngagementPolicy(),
            now=0.0,
            engaged_counts=ec,
        )
        self.assertEqual(len(itcs), 2)
        self.assertEqual(ec["T1"], 2)

    def test_unreachable_yields_hold(self):
        # 目标远超作用距离 → 无法解算 → HOLD。
        trk = _track("T1", Vec3(500_000, 0, 5000), Vec3(-300, 0, 0))
        bty = ThunderBattery("B1", Vec3(0, 0, 0), inventory=4, max_range=70_000)
        cmds, itcs = plan_and_fire(
            [_assess("T1", 0.6, ThreatLevel.HIGH)],
            {"T1": trk},
            [bty],
            EngagementPolicy(),
            now=0.0,
            engaged_counts={},
        )
        self.assertEqual(itcs, [])
        self.assertEqual(cmds[0].kind, CommandKind.HOLD)
        self.assertEqual(bty.inventory, 4)

    def test_prefers_battery_with_more_inventory(self):
        trk = _track("T1", Vec3(30_000, 0, 5000), Vec3(-300, 0, 0))
        low = ThunderBattery("LOW", Vec3(1000, 0, 0), inventory=1)
        high = ThunderBattery("HIGH", Vec3(-1000, 0, 0), inventory=8)
        cmds, _ = plan_and_fire(
            [_assess("T1", 0.6, ThreatLevel.HIGH)],
            {"T1": trk},
            [low, high],
            EngagementPolicy(),
            now=0.0,
            engaged_counts={},
        )
        self.assertEqual(cmds[0].battery_id, "HIGH")


if __name__ == "__main__":
    unittest.main()
