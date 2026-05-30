"""末段锁定:线索偏置(软硬绑定)与误关联(seeker_lock)。"""

import random
import unittest

from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.world import seeker_lock


def _tgt(tid, pos, rcs=0.15):
    return Target(tid, position=pos, aim=Vec3(0, 0, 0), cruise_speed=40.0,
                  kind=TargetKind.FIXED_WING_UAV, rcs=rcs)


class TestSeekerLock(unittest.TestCase):
    def test_cue_bias_beats_interceptor_proximity(self):
        # A 离弹体近但远离线索;B 贴近线索。锁定应被**线索**牵引 → B 占多数
        # (若按"离弹体最近"则 A 必胜——以此证明锁定随分配航迹而非全局最近)。
        interceptor = Vec3(0, 0, 0)
        a = _tgt("A", Vec3(250, 0, 0))      # 近弹体
        b = _tgt("B", Vec3(1000, 0, 0))     # 近线索
        cue = Vec3(1000, 0, 0)
        rng = random.Random(0)
        b_wins = sum(1 for _ in range(2000)
                     if seeker_lock([a, b], cue, interceptor, rng, 1200.0).target_id
                     == "B")
        self.assertGreater(b_wins, 1000)

    def test_fallback_uses_interceptor_when_cue_lost(self):
        # 线索丢失 → 以弹体为基准;近弹体的 A 占多数。
        interceptor = Vec3(0, 0, 0)
        a = _tgt("A", Vec3(250, 0, 0))
        b = _tgt("B", Vec3(1000, 0, 0))
        rng = random.Random(1)
        a_wins = sum(1 for _ in range(2000)
                     if seeker_lock([a, b], None, interceptor, rng, 1200.0).target_id
                     == "A")
        self.assertGreater(a_wins, 1000)

    def test_empty_basket(self):
        self.assertIsNone(seeker_lock([], Vec3(0, 0, 0), Vec3(), random.Random(0),
                                      1200.0))


if __name__ == "__main__":
    unittest.main()
