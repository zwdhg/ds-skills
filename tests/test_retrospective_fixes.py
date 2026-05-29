"""复盘优化的回归测试:ID 生成隔离、干扰占用回收。"""

import unittest

from c2sim.engine import Engine
from c2sim.models import IdGenerator
from c2sim.scenarios import build_point_defense_scenario


class TestIdGenerator(unittest.TestCase):
    def test_sequence_and_isolation(self):
        g1, g2 = IdGenerator(), IdGenerator()
        self.assertEqual(g1.next("T"), "T-0001")
        self.assertEqual(g1.next("T"), "T-0002")
        # 另一实例独立计数(不共享全局状态)。
        self.assertEqual(g2.next("T"), "T-0001")

    def test_engines_have_independent_ids(self):
        e1 = Engine(build_point_defense_scenario(seed=1))
        e2 = Engine(build_point_defense_scenario(seed=1))
        self.assertEqual(e1.ids.next("Z"), e2.ids.next("Z"))  # 各自从头

    def test_run_ids_reproducible_across_runs(self):
        # 同种子两次独立运行,产出的指令 ID 序列应一致(不再受全局计数器漂移)。
        r1 = Engine(build_point_defense_scenario(seed=7)).run()
        r2 = Engine(build_point_defense_scenario(seed=7)).run()
        ids1 = [c.command_id for c in r1.commands]
        ids2 = [c.command_id for c in r2.commands]
        self.assertEqual(ids1, ids2)   # 按次完全复现(含 CMD/THDR/TRK 同源计数)
        self.assertTrue(ids1)

    def test_track_ids_reproducible_across_runs(self):
        # 回归:航迹 ID(TRK)也须经引擎注入的 IdGenerator 生成,跨次按种子复现,
        # 不再受全局计数器漂移(此前 TrackFusion 漏用全局 next_id)。
        def trk_ids(seed):
            eng = Engine(build_point_defense_scenario(seed=seed))
            for _ in range(30):
                eng.step()
            return sorted(eng.tracker.tracks)
        self.assertEqual(trk_ids(7), trk_ids(7))
        self.assertTrue(all(t.startswith("TRK-") for t in trk_ids(7)))


class TestJammingLedgerInvariant(unittest.TestCase):
    def test_jammed_tracks_subset_of_currently_jammed(self):
        # 干扰台账中的航迹必须对应"当前确处于被干扰状态"的真实目标;
        # 一旦目标脱离被干扰状态即应被回收(可重新交由 Thunder)。
        engine = Engine(build_point_defense_scenario(seed=2026))
        for _ in range(40):
            engine.step()
            for _tid, target_id in engine.jammed_tracks.items():
                victim = engine.world.target_by_id(target_id)
                self.assertIsNotNone(victim)
                self.assertTrue(victim.jammed)


if __name__ == "__main__":
    unittest.main()
