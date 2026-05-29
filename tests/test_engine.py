"""端到端引擎测试。"""

import unittest

from c2sim.engine import Engine, Scenario
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.scenarios import build_demo_scenario
from c2sim.sensors import Radar
from c2sim.weapons import ThunderBattery


class TestEngine(unittest.TestCase):
    def test_single_inbound_destroyed(self):
        # 单个直扑要地的有人机,应被拦截摧毁。
        asset = Vec3(0, 0, 0)
        tgt = Target(
            "INB",
            position=Vec3(40_000, 0, 6000),
            velocity=Vec3(-250, 0, 0),
            kind=TargetKind.AIRCRAFT,
            rcs=5.0,
        )
        scenario = Scenario(
            asset=asset,
            targets=[tgt],
            radars=[Radar("R", Vec3(0, 5000, 30), max_range=120_000)],
            batteries=[
                ThunderBattery("B", Vec3(8000, 0, 20), interceptor_speed=1200)
            ],
            seed=1,
        )
        result = Engine(scenario).run()
        self.assertIn("INB", result.destroyed)
        self.assertNotIn("INB", result.leaked)
        self.assertGreater(result.interceptors_fired, 0)

    def test_leak_when_no_defense(self):
        # 无任何拦截单元 → 目标必然突防。
        tgt = Target("LEAK", Vec3(20_000, 0, 3000), Vec3(-300, 0, 0))
        scenario = Scenario(
            asset=Vec3(0, 0, 0),
            targets=[tgt],
            radars=[Radar("R", Vec3(0, 0, 30))],
            batteries=[],
            max_time=200.0,
        )
        result = Engine(scenario).run()
        self.assertIn("LEAK", result.leaked)
        self.assertEqual(result.interceptors_fired, 0)

    def test_inventory_never_negative(self):
        scenario = build_demo_scenario(seed=3)
        result = Engine(scenario).run()
        for bty in scenario.batteries:
            self.assertGreaterEqual(bty.inventory, 0)
        # 发射数不应超过总库存。
        self.assertLessEqual(result.interceptors_fired, 8 + 8)

    def test_truth_isolation(self):
        # 指控链路看到的航迹不应携带真值身份(truth_id 仅在报告层)。
        scenario = build_demo_scenario(seed=5)
        engine = Engine(scenario)
        engine.step()
        for trk in engine.fusion.tracks.values():
            self.assertFalse(hasattr(trk, "truth_id"))

    def test_demo_accounts_for_all_targets(self):
        scenario = build_demo_scenario(seed=2026)
        result = Engine(scenario).run()
        accounted = (
            len(result.destroyed)
            + len(result.leaked)
            + len(result.unresolved)
        )
        self.assertEqual(accounted, result.total_targets)

    def test_demo_intercepts_majority(self):
        # 演示想定下,空气动力目标应被可靠拦截(至少摧毁 2 个)。
        scenario = build_demo_scenario(seed=2026)
        result = Engine(scenario).run()
        self.assertGreaterEqual(len(result.destroyed), 2)


if __name__ == "__main__":
    unittest.main()
