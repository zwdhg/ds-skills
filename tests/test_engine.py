"""端到端引擎测试。"""

import unittest

from c2sim.engine import Engine, Scenario
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.scenarios import (
    build_border_band_scenario,
    build_point_defense_scenario,
)
from c2sim.sensors import SpotterPro
from c2sim.weapons import LaunchPad


def _inbound(tid, start, aim, speed, kind=TargetKind.FIXED_WING_UAV, rcs=0.15):
    return Target(
        target_id=tid,
        position=start,
        aim=aim,
        cruise_speed=speed,
        kind=kind,
        rcs=rcs,
    )


class TestEngine(unittest.TestCase):
    def test_single_inbound_destroyed(self):
        asset = Vec3(0, 0, 0)
        tgt = _inbound("INB", Vec3(7_000, 0, 500), asset, 45.0)
        scenario = Scenario(
            asset=asset,
            targets=[tgt],
            spotters=[SpotterPro("S", Vec3(0, 0, 20))],
            pads=[LaunchPad("P", Vec3(2_500, 0, 15))],
            seed=1,
        )
        result = Engine(scenario).run()
        self.assertIn("INB", result.destroyed)
        self.assertNotIn("INB", result.leaked)
        self.assertGreater(result.thunders_launched, 0)

    def test_leak_when_no_defense(self):
        tgt = _inbound("LEAK", Vec3(6_000, 0, 300), Vec3(0, 0, 0), 50.0)
        scenario = Scenario(
            asset=Vec3(0, 0, 0),
            targets=[tgt],
            spotters=[SpotterPro("S", Vec3(0, 0, 20))],
            pads=[],
            max_time=300.0,
        )
        result = Engine(scenario).run()
        self.assertIn("LEAK", result.leaked)
        self.assertEqual(result.thunders_launched, 0)

    def test_inventory_never_negative(self):
        scenario = build_point_defense_scenario(seed=3)
        total = sum(p.inventory for p in scenario.pads)
        result = Engine(scenario).run()
        for pad in scenario.pads:
            self.assertGreaterEqual(pad.inventory, 0)
        self.assertLessEqual(result.thunders_launched, total)

    def test_truth_isolation(self):
        # 指控链路看到的航迹不应携带真值身份。
        scenario = build_point_defense_scenario(seed=5)
        engine = Engine(scenario)
        engine.step()
        for trk in engine.tracker.tracks.values():
            self.assertFalse(hasattr(trk, "truth_id"))

    def test_all_targets_accounted(self):
        scenario = build_point_defense_scenario(seed=2026)
        result = Engine(scenario).run()
        accounted = (
            len(result.destroyed)
            + len(result.soft_killed)
            + len(result.leaked)
            + len(result.unresolved)
        )
        self.assertEqual(accounted, result.total_targets)

    def test_point_defense_neutralizes_all(self):
        # 软杀伤(干扰)+ 硬杀伤(Thunder)协同,应零突防处置全部目标。
        scenario = build_point_defense_scenario(seed=2026)
        result = Engine(scenario).run()
        neutralized = len(result.destroyed) + len(result.soft_killed)
        self.assertGreaterEqual(neutralized, 4)
        self.assertEqual(len(result.leaked), 0)

    def test_rf_silent_never_soft_killed(self):
        # 不变量(对任意种子成立):RF 静默目标不可被干扰,绝不出现在软杀伤。
        for seed in range(8):
            result = Engine(build_point_defense_scenario(seed=seed)).run()
            self.assertNotIn("T2-LOITER", result.soft_killed)

    def test_border_band_intercepts_all(self):
        scenario = build_border_band_scenario(seed=2026)
        result = Engine(scenario).run()
        self.assertEqual(len(result.leaked), 0)

    def test_rf_silent_loiter_still_engaged(self):
        # 巡飞弹 RF 静默,仅靠雷达/光电,仍应被融合-研判-拦截链路处置。
        scenario = build_point_defense_scenario(seed=2026)
        result = Engine(scenario).run()
        self.assertNotIn("T2-LOITER", result.unresolved)


if __name__ == "__main__":
    unittest.main()
