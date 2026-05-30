"""参数敏感性扫描测试。"""

import unittest

from c2sim.scenarios import build_swarm_scenario
from c2sim.sensitivity import (
    SweepResult,
    default_sweep,
    sweep_param,
)


class TestSweep(unittest.TestCase):
    def test_reliability_monotone_reduces_leakage(self):
        # 战斗部可靠性↑ → 突防率应不升(单调改善);摆幅 = max-min。
        r = sweep_param(
            build_swarm_scenario, "warhead_reliability",
            [0.7, 0.85, 1.0],
            lambda scn, v: setattr(scn, "warhead_reliability", v),
            seeds=range(6), metric=lambda m: m.leakage_rate,
        )
        leaks = [m for _, m in r.points]
        self.assertGreaterEqual(leaks[0] + 1e-9, leaks[-1])  # 高可靠 ≤ 低可靠
        self.assertGreaterEqual(r.swing, 0.0)

    def test_default_sweep_ranks_by_swing(self):
        results = default_sweep(build_swarm_scenario, seeds=range(4))
        self.assertTrue(results)
        swings = [r.swing for r in results]
        self.assertEqual(swings, sorted(swings, reverse=True))  # 降序
        for r in results:
            self.assertIsInstance(r, SweepResult)
            self.assertGreaterEqual(len(r.points), 2)


if __name__ == "__main__":
    unittest.main()
