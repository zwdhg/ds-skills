"""被控对象(World / Plant)—— 仿真真值与物理。

持有目标与 Thunder 的**真值**状态,负责运动学推进、近炸引信毁伤判定、
干扰软杀伤物理与突防判定。它**不知道**指控链路、不持有交战台账、不做
任何记录/呈现——只产出**结果事件**(:class:`Kill` 等),由编排者
(:class:`c2sim.engine.Engine`)翻译为统计与历史。

与指控层的唯一耦合是:传感器从这里读取目标真值产生量测,制导从这里读取
所锁目标真值做末段寻的(经 :meth:`seeker_fix` 注入导引头噪声)。这条
"真值隔离"边界由本模块的接口形态强制,而非仅靠约定。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from c2sim.geometry import Vec3, segment_cpa, turn_towards
from c2sim.models import Target
from c2sim.spatial import SpatialGrid
from c2sim.weapons import HunterMax, Thunder


# --- 结果事件(World → 编排者)-------------------------------------------


@dataclass
class Kill:
    interceptor_id: str
    track_id: str
    target_id: str
    miss: float


@dataclass
class Miss:
    interceptor_id: str
    track_id: str
    target_id: str | None
    miss: float


@dataclass
class Leak:
    target_id: str
    x: float
    y: float


@dataclass
class SoftKill:
    target_id: str
    x: float
    y: float


@dataclass
class SelfDestruct:
    interceptor_id: str
    track_id: str


class World:
    """仿真真值与物理引擎。"""

    def __init__(
        self,
        targets: list[Target],
        jammers: list[HunterMax],
        asset: Vec3,
        defended_radius: float,
        single_shot_pk: float,
        rng: random.Random,
    ) -> None:
        self.targets = targets
        self.jammers = jammers
        self.asset = asset
        self.defended_radius = defended_radius
        self.single_shot_pk = single_shot_pk
        self.rng = rng
        self.thunders: list[Thunder] = []
        self._by_id = {t.target_id: t for t in targets}  # O(1) 身份查找
        self._grid: SpatialGrid | None = None             # 就近查询加速(每步重建)

    # -- 推进 -------------------------------------------------------------

    def advance_targets(self, dt: float) -> None:
        """匀速寻的推进(被干扰目标悬停)。"""
        for tgt in self.targets:
            if tgt.alive:
                tgt.advance(dt)

    def integrate_thunders(self, dt: float) -> list[SelfDestruct]:
        """推进 Thunder 弹道:末段近炸引信引爆;飞出作业半径则自毁。

        制导矢量由控制器在调用本方法**之前**设定(零延迟末段寻的)。
        本方法只做积分与引信/边界物理,不消耗随机数。
        """
        events: list[SelfDestruct] = []
        for itc in self.thunders:
            if not itc.alive or itc.detonated:
                continue
            # 转弯率约束:实际速度朝制导指令方向至多旋转 max_turn_rate·dt。
            itc.velocity = turn_towards(
                itc.velocity, itc.desired_velocity, itc.max_turn_rate * dt
            )
            if itc.acquired:
                victim = self.target_by_id(itc.locked_target_id)
                if victim is not None and victim.alive:
                    rel_p = itc.position - victim.position
                    rel_v = itc.velocity - victim.velocity
                    t_cpa, dmin = segment_cpa(rel_p, rel_v, dt)
                    if dmin <= itc.lethal_radius:
                        itc.position = itc.position + itc.velocity * t_cpa
                        itc.miss_distance = dmin
                        itc.detonated = True
                        continue
            itc.position = itc.position + itc.velocity * dt
            if itc.out_of_range():
                itc.alive = False
                events.append(SelfDestruct(itc.interceptor_id, itc.target_track_id))
        return events

    # -- 结算 -------------------------------------------------------------

    def resolve_detonations(self) -> list[Kill | Miss]:
        """对本步引爆的 Thunder 按单发杀伤概率判定毁伤,并清理弹体。"""
        events: list[Kill | Miss] = []
        for itc in self.thunders:
            if not itc.detonated or not itc.alive:
                continue
            itc.alive = False
            victim = self.target_by_id(itc.locked_target_id)
            if victim is None or not victim.alive:
                events.append(Miss(itc.interceptor_id, itc.target_track_id, None,
                                   itc.miss_distance))
                continue
            if (itc.miss_distance <= itc.lethal_radius
                    and self.rng.random() <= self.single_shot_pk):
                victim.alive = False
                events.append(Kill(itc.interceptor_id, itc.target_track_id,
                                   victim.target_id, itc.miss_distance))
            else:
                events.append(Miss(itc.interceptor_id, itc.target_track_id,
                                   victim.target_id, itc.miss_distance))
        self.thunders = [i for i in self.thunders if i.alive]
        return events

    def resolve_jamming(self, dt: float) -> list[SoftKill]:
        """累计被干扰时长;达阈值判软杀伤(迫降/返航)。脱离干扰圈则解除。"""
        events: list[SoftKill] = []
        for tgt in self.targets:
            if not tgt.alive or not tgt.jammed:
                continue
            covering = [j for j in self.jammers if j.covers(tgt.position)]
            if not covering:
                tgt.jammed = False
                tgt.jam_elapsed = 0.0
                continue
            tgt.jam_elapsed += dt
            if tgt.jam_elapsed >= min(j.hold_time for j in covering):
                tgt.alive = False
                events.append(SoftKill(tgt.target_id, tgt.position.x, tgt.position.y))
        return events

    def check_leaks(self) -> list[Leak]:
        """进入"安全穹顶"防护半径仍存活的目标判为突防。"""
        events: list[Leak] = []
        for tgt in self.targets:
            if tgt.alive and tgt.position.distance_to(self.asset) <= self.defended_radius:
                tgt.alive = False
                events.append(Leak(tgt.target_id, tgt.position.x, tgt.position.y))
        return events

    # -- 查询 / 注入 ------------------------------------------------------

    def reindex(self) -> None:
        """重建存活目标的空间索引(每步物理结算后调用)。"""
        grid = SpatialGrid(cell=1000.0)
        for t in self.targets:
            if t.alive:
                grid.insert(t.position, t)
        self._grid = grid

    def target_by_id(self, tid: str | None) -> Target | None:
        return self._by_id.get(tid) if tid is not None else None

    def nearest_alive_target(self, point: Vec3) -> Target | None:
        """距 ``point`` 最近的存活目标。

        有空间索引时走网格(与暴力遍历结果一致),否则回退暴力遍历。
        """
        if self._grid is not None:
            return self._grid.nearest(point)
        best, best_d = None, float("inf")
        for tgt in self.targets:
            if not tgt.alive:
                continue
            d = point.distance_to(tgt.position)
            if d < best_d:
                best, best_d = tgt, d
        return best

    def nearest_jammable(self, point: Vec3, jammer: HunterMax) -> Target | None:
        """干扰圈内、依赖 RF 链路的最近真实目标。"""
        best, best_d = None, float("inf")
        for t in self.targets:
            if not t.alive or not t.emits_rf or not jammer.covers(t.position):
                continue
            d = point.distance_to(t.position)
            if d < best_d:
                best, best_d = t, d
        return best

    def seeker_fix(self, victim: Target, sigma: float) -> Vec3:
        """弹载导引头对真实目标的带噪量测。"""
        return Vec3(
            victim.position.x + self.rng.gauss(0.0, sigma),
            victim.position.y + self.rng.gauss(0.0, sigma),
            victim.position.z + self.rng.gauss(0.0, sigma),
        )

    def add_thunders(self, thunders: list[Thunder]) -> None:
        self.thunders.extend(thunders)

    def active_target_count(self) -> int:
        return sum(1 for t in self.targets if t.alive)

    def alive_target_ids(self) -> list[str]:
        return [t.target_id for t in self.targets if t.alive]
