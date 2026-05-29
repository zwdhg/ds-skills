"""末段锁定:硬绑定与误关联(select_seeker_lock)。"""

import unittest

from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.world import select_seeker_lock


def _tgt(tid, pos):
    return Target(tid, position=pos, aim=Vec3(0, 0, 0), cruise_speed=40.0,
                  kind=TargetKind.FIXED_WING_UAV)


class TestSeekerLock(unittest.TestCase):
    def test_hard_binding_follows_cue_not_interceptor(self):
        # A 离弹体更近,但上行线索 cue 指向 B → 应锁 B(随分配航迹,而非全局最近)。
        interceptor = Vec3(0, 0, 0)
        a = _tgt("A", Vec3(300, 0, 0))     # 离弹体近
        b = _tgt("B", Vec3(900, 0, 0))     # 离线索近
        cue = Vec3(920, 0, 0)
        got = select_seeker_lock([a, b], cue, interceptor)
        self.assertEqual(got.target_id, "B")

    def test_misassociation_locks_decoy_nearer_cue(self):
        # 预定目标 B,但诱饵 D 比 B 更接近(带噪)线索 → 锁错(误关联自然涌现)。
        interceptor = Vec3(0, 0, 0)
        b = _tgt("B", Vec3(1000, 0, 0))
        decoy = _tgt("D", Vec3(1000, 120, 0))
        cue = Vec3(1000, 90, 0)  # 噪声使线索偏向诱饵
        got = select_seeker_lock([b, decoy], cue, interceptor)
        self.assertEqual(got.target_id, "D")

    def test_fallback_to_nearest_when_cue_lost(self):
        interceptor = Vec3(0, 0, 0)
        a = _tgt("A", Vec3(300, 0, 0))
        b = _tgt("B", Vec3(900, 0, 0))
        got = select_seeker_lock([a, b], None, interceptor)
        self.assertEqual(got.target_id, "A")  # 航迹丢失 → 自主就近

    def test_empty_basket(self):
        self.assertIsNone(select_seeker_lock([], Vec3(0, 0, 0), Vec3(0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
