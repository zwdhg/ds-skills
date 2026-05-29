"""策略协议:契约(LSP)、协议符合性、注入(OCP/DIP)测试。"""

import unittest

from c2sim.engine import Engine
from c2sim.fusion import TrackFusion
from c2sim.geometry import Vec3
from c2sim.guidance import LeadPursuitGuidance, PurePursuitGuidance
from c2sim.interception import GreedyAssigner
from c2sim.scenarios import build_point_defense_scenario
from c2sim.sensors import SpotterPro
from c2sim.strategies import (
    GuidanceLaw,
    SensorModel,
    ThreatModel,
    Tracker,
    WeaponTargetAssigner,
)
from c2sim.threat import WeightedThreatModel


class TestGuidanceContract(unittest.TestCase):
    """所有 GuidanceLaw 实现须满足同一契约(里氏替换)。"""

    LAWS = [LeadPursuitGuidance(), PurePursuitGuidance()]

    def test_self_consistent_and_nonnegative_time(self):
        shooter = Vec3(0, 0, 0)
        for law in self.LAWS:
            sol = law.aim(shooter, 100.0, Vec3(1000, 200, 0), Vec3(0, 30, 0))
            self.assertIsNotNone(sol, msg=type(law).__name__)
            aim, tgo = sol
            self.assertGreaterEqual(tgo, 0.0)
            # 瞄准点自洽:按 speed 飞行 tgo 所经距离 == 到瞄准点距离。
            self.assertAlmostEqual(
                shooter.distance_to(aim), 100.0 * tgo, places=3,
                msg=type(law).__name__,
            )

    def test_stationary_target(self):
        for law in self.LAWS:
            sol = law.aim(Vec3(0, 0, 0), 100.0, Vec3(1000, 0, 0), Vec3())
            self.assertIsNotNone(sol)
            aim, tgo = sol
            self.assertAlmostEqual(aim.x, 1000.0, places=3)
            self.assertAlmostEqual(tgo, 10.0, places=3)

    def test_zero_speed_unreachable(self):
        for law in self.LAWS:
            self.assertIsNone(law.aim(Vec3(), 0.0, Vec3(100, 0, 0), Vec3()))


class TestProtocolConformance(unittest.TestCase):
    def test_default_implementations_conform(self):
        self.assertIsInstance(SpotterPro("S", Vec3()), SensorModel)
        self.assertIsInstance(TrackFusion(), Tracker)
        self.assertIsInstance(WeightedThreatModel(), ThreatModel)
        self.assertIsInstance(GreedyAssigner(), WeaponTargetAssigner)
        self.assertIsInstance(LeadPursuitGuidance(), GuidanceLaw)
        self.assertIsInstance(PurePursuitGuidance(), GuidanceLaw)


class _NullAssigner:
    """从不交战的分配器(测试注入用)。"""

    def plan(self, assessments, tracks, pads, now, engaged_counts, skip_tracks=None):
        return [], []


class TestInjection(unittest.TestCase):
    """OCP/DIP:不改引擎,注入替代策略即可改变行为。"""

    def test_inject_alternative_guidance_runs(self):
        engine = Engine(
            build_point_defense_scenario(seed=2026),
            guidance=PurePursuitGuidance(),
        )
        result = engine.run()
        accounted = (
            len(result.destroyed) + len(result.soft_killed)
            + len(result.leaked) + len(result.unresolved)
        )
        self.assertEqual(accounted, result.total_targets)

    def test_inject_null_assigner_disables_hard_kill(self):
        engine = Engine(
            build_point_defense_scenario(seed=2026),
            assigner=_NullAssigner(),
        )
        result = engine.run()
        self.assertEqual(result.thunders_launched, 0)
        # RF 静默巡飞弹不可干扰,无 Thunder 必然突防。
        self.assertIn("T2-LOITER", result.leaked)


if __name__ == "__main__":
    unittest.main()
