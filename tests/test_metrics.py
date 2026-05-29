"""效能度量与蒙特卡洛聚合测试。"""

import math
import unittest

from c2sim.engine import SimResult
from c2sim.metrics import BatchMoe, Moe, run_batch
from c2sim.scenarios import build_point_defense_scenario


def _result(total, destroyed, soft, leaked, unresolved, thunders, dur):
    return SimResult(
        destroyed=[f"d{i}" for i in range(destroyed)],
        soft_killed=[f"s{i}" for i in range(soft)],
        leaked=[f"l{i}" for i in range(leaked)],
        unresolved=[f"u{i}" for i in range(unresolved)],
        total_targets=total,
        thunders_launched=thunders,
        duration=dur,
    )


class TestMoe(unittest.TestCase):
    def test_rates_and_cost(self):
        m = Moe.from_result(_result(4, 1, 2, 1, 0, 6, 150.0))
        self.assertEqual(m.neutralized, 3)
        self.assertAlmostEqual(m.leakage_rate, 0.25)
        self.assertAlmostEqual(m.neutralization_rate, 0.75)
        self.assertAlmostEqual(m.cost_per_kill, 6.0)  # 6 架 / 1 硬杀伤

    def test_cost_per_kill_infinite_without_hard_kill(self):
        m = Moe.from_result(_result(2, 0, 2, 0, 0, 0, 100.0))
        self.assertTrue(math.isinf(m.cost_per_kill))


class TestBatchMoe(unittest.TestCase):
    def test_aggregate_prob_zero_leak_and_means(self):
        runs = [
            Moe.from_result(_result(4, 4, 0, 0, 0, 8, 100.0)),
            Moe.from_result(_result(4, 3, 0, 1, 0, 7, 120.0)),
            Moe.from_result(_result(4, 4, 0, 0, 0, 9, 110.0)),
        ]
        b = BatchMoe.aggregate(runs)
        self.assertEqual(b.n_runs, 3)
        self.assertAlmostEqual(b.prob_zero_leak, 2 / 3)
        self.assertAlmostEqual(b.thunders.mean, 8.0)
        self.assertGreater(b.leakage_rate.mean, 0.0)

    def test_aggregate_empty_raises(self):
        with self.assertRaises(ValueError):
            BatchMoe.aggregate([])


class TestRunBatch(unittest.TestCase):
    def test_run_batch_over_seeds(self):
        runs = run_batch(build_point_defense_scenario, range(3))
        self.assertEqual(len(runs), 3)
        for m in runs:
            self.assertEqual(m.total, 4)
            self.assertLessEqual(m.neutralized + m.leaked + m.unresolved, m.total)


if __name__ == "__main__":
    unittest.main()
