"""M-of-N 航迹确认:杂波鲁棒性。"""

import unittest

from c2sim.engine import Engine
from c2sim.fusion import TrackFusion
from c2sim.scenarios import build_point_defense_scenario


def _peak_confirmed_tracks(confirm_threshold, clutter, seed):
    scn = build_point_defense_scenario(seed=seed)
    for sp in scn.spotters:
        sp.clutter_rate = clutter
    eng = Engine(scn, tracker=TrackFusion(
        gate_distance=600.0, max_coast=6.0, confirm_threshold=confirm_threshold))
    peak = 0
    for _ in range(120):
        eng.step()
        peak = max(peak, sum(1 for t in eng.tracker.tracks.values() if t.confirmed))
    return peak


class TestConfirmation(unittest.TestCase):
    def test_default_threshold_one_confirms_immediately(self):
        # 阈值=1:首次命中即确认(向后兼容,不改默认行为)。
        t = TrackFusion(confirm_threshold=1)
        from c2sim.geometry import Vec3
        from c2sim.models import SensorModality, SensorReport
        t.update([SensorReport("S", SensorModality.RADAR, 0.0, Vec3(1000, 0, 0), 20.0)],
                 0.0)
        self.assertTrue(next(iter(t.tracks.values())).confirmed)

    def test_confirmation_suppresses_clutter_tracks(self):
        # 重杂波下,确认(N≥3)应将"确认航迹"峰值大幅压低(逼近真实目标数),
        # 而无确认(N=1)会被杂波灌爆。
        no_confirm = _peak_confirmed_tracks(1, clutter=4.0, seed=2026)
        confirmed = _peak_confirmed_tracks(4, clutter=4.0, seed=2026)
        self.assertGreater(no_confirm, 20)          # 杂波灌爆
        self.assertLess(confirmed, no_confirm / 3)  # 确认大幅净化画面

    def test_no_clutter_confirmation_keeps_real_tracks(self):
        # 无杂波时,确认阈值不应压制真实目标(仍能确认 4 个真实航迹之多)。
        peak = _peak_confirmed_tracks(4, clutter=0.0, seed=2026)
        self.assertGreaterEqual(peak, 3)


if __name__ == "__main__":
    unittest.main()
