"""规模/蜂群想定测试。"""

import unittest

from c2sim.engine import Engine
from c2sim.scenarios import build_swarm_scenario


class TestSwarm(unittest.TestCase):
    def test_runs_and_accounts_all(self):
        scenario = build_swarm_scenario(seed=2026, n=40)
        result = Engine(scenario).run()
        accounted = (
            len(result.destroyed) + len(result.soft_killed)
            + len(result.leaked) + len(result.unresolved)
        )
        self.assertEqual(accounted, 40)

    def test_neutralizes_majority(self):
        # 兵力可观,应处置过半;但因 TAS 容量与库存饱和,通常非零突防。
        result = Engine(build_swarm_scenario(seed=2026, n=40)).run()
        neutralized = len(result.destroyed) + len(result.soft_killed)
        self.assertGreaterEqual(neutralized, 20)

    def test_inventory_never_negative(self):
        scenario = build_swarm_scenario(seed=2026, n=40)
        total = sum(p.inventory for p in scenario.pads)
        result = Engine(scenario).run()
        for pad in scenario.pads:
            self.assertGreaterEqual(pad.inventory, 0)
        self.assertLessEqual(result.thunders_launched, total)


if __name__ == "__main__":
    unittest.main()
