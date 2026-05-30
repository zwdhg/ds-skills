"""保真度补强测试:转弯率约束、虚警/杂波鲁棒性。"""

import math
import random
import unittest

from c2sim.engine import Engine, Scenario
from c2sim.geometry import Vec3, turn_towards
from c2sim.models import Target, TargetKind
from c2sim.scenarios import build_point_defense_scenario
from c2sim.sensors import SpotterPro
from c2sim.weapons import LaunchPad


class TestTurnTowards(unittest.TestCase):
    def test_within_limit_reaches_desired(self):
        cur = Vec3(10, 0, 0)
        des = Vec3(0, 10, 0)  # 90°
        out = turn_towards(cur, des, math.radians(180))
        self.assertAlmostEqual(out.x, 0.0, places=6)
        self.assertAlmostEqual(out.y, 10.0, places=6)

    def test_clamped_when_exceeding(self):
        cur = Vec3(10, 0, 0)
        des = Vec3(0, 10, 0)  # 期望转 90°
        out = turn_towards(cur, des, math.radians(30))  # 只准转 30°
        # 模长保持(=期望速率),夹角被限制到 30°。
        self.assertAlmostEqual(out.norm(), 10.0, places=6)
        ang = math.degrees(math.acos(cur.unit().dot(out.unit())))
        self.assertAlmostEqual(ang, 30.0, places=4)

    def test_turn_rate_costs_more_interceptors(self):
        # 极小转弯率应降低拦截效率(需要更多 Thunder / 或更多突防),
        # 至少不优于敏捷弹。
        agile = build_point_defense_scenario(seed=2026)
        for p in agile.pads:
            p.max_turn_rate = 5.0
        sluggish = build_point_defense_scenario(seed=2026)
        for p in sluggish.pads:
            p.max_turn_rate = 0.2
        ra = Engine(agile).run()
        rs = Engine(sluggish).run()
        self.assertGreaterEqual(
            len(rs.leaked) + rs.thunders_launched,
            len(ra.leaked) + ra.thunders_launched,
        )


class TestClutter(unittest.TestCase):
    def test_clutter_disabled_by_default(self):
        s = SpotterPro("S", Vec3(0, 0, 20))
        self.assertEqual(s.clutter_rate, 0.0)

    def test_clutter_generates_false_reports(self):
        s = SpotterPro("S", Vec3(0, 0, 20), clutter_rate=5.0)
        rng = random.Random(0)
        reps = s.observe([], 0.0, rng)  # 无真实目标
        self.assertGreater(len(reps), 0)
        self.assertTrue(all(r.truth_id is None for r in reps))

    def test_defense_robust_to_clutter(self):
        # 有杂波时仍应处置真实目标(融合需经得起虚警)。
        scenario = build_point_defense_scenario(seed=2026)
        for sp in scenario.spotters:
            sp.clutter_rate = 2.0
        result = Engine(scenario).run()
        neutralized = len(result.destroyed) + len(result.soft_killed)
        self.assertGreaterEqual(neutralized, 3)


if __name__ == "__main__":
    unittest.main()
