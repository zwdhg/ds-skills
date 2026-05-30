"""末段建模细化:信杂比加权锁定、再捕获、诱饵误关联量化。"""

import random
import unittest

from c2sim.engine import Engine
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.scenarios import build_decoy_scenario, build_swarm_scenario
from c2sim.world import seeker_lock


def _tgt(tid, pos, rcs=0.15):
    return Target(tid, position=pos, aim=Vec3(0, 0, 0), cruise_speed=40.0,
                  kind=TargetKind.FIXED_WING_UAV, rcs=rcs)


class _NoRng:
    def random(self):
        raise AssertionError("单候选不应消耗随机数")


class TestSeekerLock(unittest.TestCase):
    def test_single_candidate_no_rng(self):
        only = _tgt("A", Vec3(1000, 0, 0))
        got = seeker_lock([only], Vec3(1000, 0, 0), Vec3(), _NoRng(), 1200.0)
        self.assertEqual(got.target_id, "A")

    def test_strong_decoy_steals_lock_majority(self):
        # 真目标在线索处但弱回波;诱饵紧邻、强回波 → 多数被锁诱饵(信杂比)。
        real = _tgt("REAL", Vec3(1000, 0, 0), rcs=0.15)
        decoy = _tgt("DECOY", Vec3(1000, 150, 0), rcs=0.6)
        cue = Vec3(1000, 0, 0)
        rng = random.Random(0)
        wins = sum(1 for _ in range(2000)
                   if seeker_lock([real, decoy], cue, Vec3(), rng, 1200.0).target_id
                   == "DECOY")
        self.assertGreater(wins, 1000)  # 诱饵夺锁占多数

    def test_weaker_far_decoy_rarely_wins(self):
        real = _tgt("REAL", Vec3(1000, 0, 0), rcs=0.3)
        decoy = _tgt("DECOY", Vec3(1000, 900, 0), rcs=0.2)  # 远且弱
        rng = random.Random(1)
        wins = sum(1 for _ in range(2000)
                   if seeker_lock([real, decoy], Vec3(1000, 0, 0), Vec3(), rng,
                                  1200.0).target_id == "DECOY")
        self.assertLess(wins, 600)


class TestReacquisition(unittest.TestCase):
    def test_lost_lock_returns_to_search(self):
        # 已锁目标被置死 → 末段制导丢锁、回到搜索态(可再捕获)。
        from c2sim.scenarios import build_point_defense_scenario
        from c2sim.weapons import Phase

        scenario = build_point_defense_scenario(seed=1)
        engine = Engine(scenario)
        engine.step()
        # 构造一架已锁定的 Thunder,锁定某真实目标。
        itc = scenario.pads[0].fire("TRK-x", scenario.asset, engine.now + 1.0,
                                    engine.now, ids=engine.ids)
        target = scenario.targets[0]
        itc.acquired = True
        itc.locked_target_id = target.target_id
        itc.phase = Phase.TERMINAL
        engine.world.thunders.append(itc)
        # 所锁目标被击杀 → 丢锁回搜索。
        target.alive = False
        engine._guide_terminal()
        self.assertFalse(itc.acquired)
        self.assertIsNone(itc.locked_target_id)
        self.assertEqual(itc.phase, Phase.SEARCH)


class TestDecoyMisassociation(unittest.TestCase):
    def _real_leak_rate(self, with_decoys, seeds):
        leaked = total = 0
        for s in seeds:
            r = Engine(build_decoy_scenario(seed=s, with_decoys=with_decoys)).run()
            total += 4
            leaked += sum(1 for t in r.leaked if t.startswith("REAL"))
        return leaked / total

    def test_decoys_raise_real_leakage(self):
        # 亚视场强回波诱饵应通过误关联显著抬高真目标突防率(相对无诱饵对照)。
        ctrl = self._real_leak_rate(False, range(12))
        test = self._real_leak_rate(True, range(12))
        self.assertGreater(test, ctrl + 0.15)


if __name__ == "__main__":
    unittest.main()
