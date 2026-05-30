"""协方差跟踪器:协议符合、可注入、各向异性高度精度增益。"""

import unittest

from c2sim.engine import Engine, Scenario
from c2sim.fusion import TrackFusion
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.sensors import SpotterPro
from c2sim.strategies import Tracker
from c2sim.tracking import CovarianceTracker


def _radar_only_scenario():
    # 单目标、关光电(纯雷达)、远距:俯仰误差主导高度噪声。
    tgt = Target("T", position=Vec3(9000, 3000, 1500), aim=Vec3(0, 3000, 1500),
                 cruise_speed=50.0, kind=TargetKind.FIXED_WING_UAV, rcs=0.2)
    return tgt, Scenario(
        asset=Vec3(0, 0, 0), targets=[tgt],
        spotters=[SpotterPro("S", Vec3(0, 0, 20), eo_range=0.0)],
        pads=[], max_time=120.0, seed=1,
    )


def _mean_altitude_error(tracker):
    tgt, scn = _radar_only_scenario()
    eng = Engine(scn, tracker=tracker)
    errs = []
    for _ in range(200):
        eng.step()
        trks = list(eng.tracker.tracks.values())
        if trks and tgt.alive:
            errs.append(abs(trks[0].position.z - tgt.position.z))
    return sum(errs) / len(errs)


class TestCovarianceTracker(unittest.TestCase):
    def test_conforms_to_protocol(self):
        self.assertIsInstance(CovarianceTracker(), Tracker)

    def test_injectable_into_engine(self):
        from c2sim.scenarios import build_point_defense_scenario
        result = Engine(build_point_defense_scenario(seed=2026),
                        tracker=CovarianceTracker(gate_distance=600.0)).run()
        accounted = (len(result.destroyed) + len(result.soft_killed)
                     + len(result.leaked) + len(result.unresolved))
        self.assertEqual(accounted, result.total_targets)

    def test_improves_altitude_accuracy_vs_scalar(self):
        # 雷达-only 远距:逐轴协方差融合应明显优于标量 α-β 的高度估计。
        scalar = _mean_altitude_error(
            TrackFusion(gate_distance=600.0, max_coast=6.0))
        cov = _mean_altitude_error(CovarianceTracker(gate_distance=600.0))
        self.assertLess(cov, scalar * 0.9)

    def test_reproducible(self):
        a = _mean_altitude_error(CovarianceTracker(gate_distance=600.0))
        b = _mean_altitude_error(CovarianceTracker(gate_distance=600.0))
        self.assertAlmostEqual(a, b, places=9)


if __name__ == "__main__":
    unittest.main()
