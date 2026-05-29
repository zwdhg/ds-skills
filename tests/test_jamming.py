"""Hunter Max 软杀伤(无线电干扰)测试。"""

import unittest

from c2sim.engine import Engine, Scenario
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.sensors import SpotterPro
from c2sim.weapons import HunterMax, LaunchPad


def _rf_target(tid, start, emits_rf=True):
    return Target(
        target_id=tid,
        position=start,
        aim=Vec3(0, 0, 0),
        cruise_speed=45.0,
        kind=TargetKind.FIXED_WING_UAV,
        rcs=0.15,
        emits_rf=emits_rf,
    )


def _scenario(target, jammers, pads=None):
    return Scenario(
        asset=Vec3(0, 0, 0),
        targets=[target],
        spotters=[SpotterPro("S", Vec3(0, 0, 20))],
        pads=pads or [],
        jammers=jammers,
        max_time=200.0,
    )


class TestHunterMax(unittest.TestCase):
    def test_covers(self):
        hm = HunterMax("H", Vec3(0, 0, 0), jam_range=4000.0)
        self.assertTrue(hm.covers(Vec3(3000, 0, 0)))
        self.assertFalse(hm.covers(Vec3(5000, 0, 0)))


class TestSoftKill(unittest.TestCase):
    def test_rf_target_soft_killed_without_thunder(self):
        tgt = _rf_target("RF", Vec3(6000, 0, 400))
        hm = HunterMax("HM", Vec3(0, 0, 0), jam_range=7000.0, hold_time=3.0)
        result = Engine(_scenario(tgt, [hm])).run()
        self.assertIn("RF", result.soft_killed)
        self.assertNotIn("RF", result.leaked)
        self.assertEqual(result.thunders_launched, 0)  # 未消耗硬杀伤资源

    def test_rf_silent_target_immune_to_jamming(self):
        # RF 静默(自主)目标不可被干扰;无 Thunder 时应突防而非软杀伤。
        tgt = _rf_target("SILENT", Vec3(6000, 0, 400), emits_rf=False)
        hm = HunterMax("HM", Vec3(0, 0, 0), jam_range=7000.0, hold_time=3.0)
        result = Engine(_scenario(tgt, [hm])).run()
        self.assertNotIn("SILENT", result.soft_killed)
        self.assertIn("SILENT", result.leaked)

    def test_jammed_target_hovers(self):
        tgt = _rf_target("RF", Vec3(6000, 0, 400))
        tgt.jammed = True
        self.assertEqual(tgt.velocity.norm(), 0.0)  # 链路中断 → 悬停


if __name__ == "__main__":
    unittest.main()
