"""外部真值交叉验证框架(骨架)测试。

仅验证**管线本身**跑通与契约正确;按 model-card/AGENTS 约束,合成轨迹不作
oracle,故这里只断言管线行为与相对性质,不断言绝对精度结论。
"""

import os
import tempfile
import unittest

from c2sim.fusion import TrackFusion
from c2sim.geometry import Vec3
from c2sim.sensors import SpotterPro
from c2sim.tracking import CovarianceTracker
from c2sim.validation import (
    TruthTrack,
    Validator,
    load_truth_csv,
    synthetic_track,
)


def _spotter():
    return SpotterPro("VAL", Vec3(0, 0, 20), rf_detect_prob=1.0,
                      radar_detect_prob=1.0, eo_classify_prob=1.0)


class TestTruthContract(unittest.TestCase):
    def test_add_and_query(self):
        tr = TruthTrack()
        tr.add(0.0, "A", 100, 0, 0)
        tr.add(0.5, "A", 150, 0, 0)
        self.assertEqual(tr.times(), [0.0, 0.5])
        self.assertEqual(tr.at(0.5)["A"].as_tuple(), (150, 0, 0))

    def test_csv_roundtrip(self):
        tr = synthetic_track(n=5)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.csv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("t,target_id,x,y,z\n")
                for s in tr.samples:
                    f.write(f"{s.t},{s.target_id},"
                            f"{s.position.x},{s.position.y},{s.position.z}\n")
            loaded = load_truth_csv(p)
        self.assertEqual(len(loaded.samples), len(tr.samples))
        self.assertEqual(loaded.at(0.0).keys(), tr.at(0.0).keys())


class TestValidatorPipeline(unittest.TestCase):
    def test_runs_and_reports_metrics(self):
        truth = synthetic_track(n=40)
        val = Validator(_spotter(), lambda: TrackFusion(gate_distance=600.0))
        err = val.run(truth)
        self.assertEqual(err.n, 40)
        self.assertGreater(err.matched_fraction, 0.5)   # 多数样本被关联
        self.assertGreater(err.rmse, 0.0)               # 带噪量测 → 非零误差
        self.assertGreaterEqual(err.max_error, err.rmse)

    def test_reproducible(self):
        truth = synthetic_track(n=30)
        a = Validator(_spotter(), lambda: CovarianceTracker(), seed=1).run(truth)
        b = Validator(_spotter(), lambda: CovarianceTracker(), seed=1).run(truth)
        self.assertAlmostEqual(a.rmse, b.rmse, places=9)

    def test_works_for_any_tracker_via_protocol(self):
        # 验证器只依赖 Tracker 协议:两种跟踪器都能跑(DI 接缝复用)。
        truth = synthetic_track(n=30)
        for fac in (lambda: TrackFusion(gate_distance=600.0),
                    lambda: CovarianceTracker(gate_distance=600.0)):
            err = Validator(_spotter(), fac).run(truth)
            self.assertGreater(err.matched_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
