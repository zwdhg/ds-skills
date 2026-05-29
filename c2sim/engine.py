"""仿真引擎:Skyshield Nexus 指控闭环按时间步推进。

每个仿真步(tick)依次执行:

1. 推进来袭目标(真值,含末段加速寻的)与在飞 Thunder;
2. 处理 Thunder 引爆,按其与目标真值位置之差与杀伤半径判定毁伤;
3. 检查目标是否突防(进入"安全穹顶"防护半径);
4. Spotter Pro 多模态探测 → 航迹融合 → 威胁研判 → 拦截指令生成 → 发射;
5. Thunder 分阶段制导(起飞抵近 / 目标搜索 / 末段拦截)。

引擎掌握"真值"(目标真实位置、毁伤判定),指控链路只能看到带噪航迹——
这一信息隔离是整套仿真可信度的关键。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from c2sim.fusion import TrackFusion
from c2sim.geometry import Vec3, lead_intercept_time, segment_cpa
from c2sim.interception import EngagementPolicy, plan_and_fire
from c2sim.models import Command, Target
from c2sim.sensors import SpotterPro
from c2sim.threat import ThreatPolicy, assess
from c2sim.weapons import LaunchPad, Phase, Thunder


@dataclass
class Scenario:
    """一次演练想定。"""

    asset: Vec3                       # 被掩护要地("安全穹顶"中心)
    targets: list[Target]
    spotters: list[SpotterPro]
    pads: list[LaunchPad]
    dt: float = 0.5                   # 仿真步长(秒)
    max_time: float = 600.0           # 最长仿真时长(秒)
    seed: int = 1234
    threat_policy: ThreatPolicy = field(default_factory=ThreatPolicy)
    engagement_policy: EngagementPolicy = field(default_factory=EngagementPolicy)
    single_shot_pk: float = 0.92      # 单架 Thunder 命中波门内时的毁伤概率


@dataclass
class Event:
    """带时间戳的仿真事件,供复盘。"""

    time: float
    kind: str
    detail: str


@dataclass
class SimResult:
    """一次仿真的结果汇总。"""

    destroyed: list[str] = field(default_factory=list)
    leaked: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    total_targets: int = 0
    thunders_launched: int = 0
    commands: list[Command] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    duration: float = 0.0

    def summary(self) -> str:
        tail = f" | 在途未决 {len(self.unresolved)}" if self.unresolved else ""
        return (
            f"用时 {self.duration:.1f}s | 来袭目标 {self.total_targets} | "
            f"摧毁 {len(self.destroyed)} | 突防 {len(self.leaked)}{tail} | "
            f"发射 Thunder {self.thunders_launched} 架"
        )


class Engine:
    """Skyshield Nexus 指控仿真引擎。"""

    def __init__(self, scenario: Scenario) -> None:
        self.s = scenario
        self.rng = random.Random(scenario.seed)
        self.fusion = TrackFusion(gate_distance=600.0, max_coast=6.0)
        self.thunders: list[Thunder] = []
        self.engaged_counts: dict[str, int] = {}
        self.now = 0.0
        self.result = SimResult(total_targets=len(scenario.targets))

    # -- 单步 -------------------------------------------------------------

    def step(self) -> None:
        s = self.s
        dt = s.dt
        step_start = self.now
        self.now += dt

        # 1) 推进真值。
        for tgt in s.targets:
            if tgt.alive:
                tgt.advance(dt)
        # 推进 Thunder(末段以近炸引信判定引爆;飞出作业半径则自毁)。
        for itc in self.thunders:
            self._fly(itc, dt)

        # 2) 处理引爆与毁伤。
        self._resolve_detonations()

        # 3) 突防判定。
        self._check_leaks()

        # 4) Spotter Pro 多模态探测。
        reports = []
        for spotter in s.spotters:
            reports.extend(spotter.observe(s.targets, self.now, self.rng))

        # 5) 航迹融合。
        tracks = self.fusion.update(reports, self.now)
        track_map = {t.track_id: t for t in tracks}

        # 已消失的航迹释放其交战占用,允许对存活目标重新交战。
        for tid in list(self.engaged_counts):
            if tid not in track_map:
                del self.engaged_counts[tid]

        # 6) 威胁研判。
        assessments = assess(tracks, s.asset, s.threat_policy)

        # 7) Thunder 分阶段制导。
        self._guide(track_map)

        # 8) 拦截指令生成与发射。
        commands, new_thunders = plan_and_fire(
            assessments,
            track_map,
            s.pads,
            s.engagement_policy,
            self.now,
            self.engaged_counts,
        )
        for cmd in commands:
            self.result.commands.append(cmd)
            if cmd.kind.value == "engage":
                self._log("交战", cmd.note)
        self.thunders.extend(new_thunders)
        self.result.thunders_launched += len(new_thunders)

    # -- Thunder 飞行 -----------------------------------------------------

    def _target_by_id(self, tid: str | None) -> Target | None:
        if tid is None:
            return None
        for t in self.s.targets:
            if t.target_id == tid:
                return t
        return None

    def _fly(self, itc: Thunder, dt: float) -> None:
        """推进单架 Thunder 一步。

        末段(已锁定)在本步起始**以所锁目标的当前真值零延迟重解寻的**
        (弹载图像 AI),再用近炸引信判定:若本步内最近距离进入杀伤半径,
        则在最近接近点引爆。未锁定者按制导矢量飞行;飞出作业半径即自毁
        (不计毁伤),避免在空域中无意义滞留。
        """
        if not itc.alive or itc.detonated:
            return

        if itc.acquired:
            victim = self._target_by_id(itc.locked_target_id)
            if victim is not None and victim.alive:
                # 末段寻的:以(带导引头噪声的)真实目标为基准重解拦截矢量。
                seen = self._seeker_fix(victim, itc.seeker_sigma)
                t = lead_intercept_time(
                    itc.position, seen, victim.velocity, itc.max_speed
                )
                if t is not None:
                    itc.steer_to(seen + victim.velocity * t, self.now + t)
                # 近炸引信:本步内掠过杀伤半径即引爆。
                rel_p = itc.position - victim.position
                rel_v = itc.velocity - victim.velocity
                t_cpa, dmin = segment_cpa(rel_p, rel_v, dt)
                if dmin <= itc.lethal_radius:
                    itc.position = itc.position + itc.velocity * t_cpa
                    itc.miss_distance = dmin
                    itc.detonated = True
                    return

        itc.position = itc.position + itc.velocity * dt
        if itc.out_of_range():
            itc.alive = False
            self._release_engagement(itc)
            self._log("自毁", f"{itc.interceptor_id} 飞出作业半径,任务终止")

    def _seeker_fix(self, victim: Target, sigma: float) -> Vec3:
        return Vec3(
            victim.position.x + self.rng.gauss(0.0, sigma),
            victim.position.y + self.rng.gauss(0.0, sigma),
            victim.position.z + self.rng.gauss(0.0, sigma),
        )

    def _release_engagement(self, itc: Thunder) -> None:
        """释放该 Thunder 对其航迹的一次交战占用。"""
        tid = itc.target_track_id
        if tid in self.engaged_counts:
            self.engaged_counts[tid] = max(0, self.engaged_counts[tid] - 1)

    # -- 制导 -------------------------------------------------------------

    def _guide(self, track_map: dict[str, "object"]) -> None:
        """Thunder 起飞抵近段制导与目标搜索/锁定。

        * **起飞抵近**:依指控上行航迹做指令制导,飞向预测拦截点;
        * **目标搜索**:进入弹载传感器截获距离后按概率锁定真实目标。

        末段寻的与近炸引信在 :meth:`_fly` 中以零延迟真值解算。
        """
        from c2sim.models import Track

        for itc in self.thunders:
            if itc.detonated or not itc.alive:
                continue

            # 阶段推进:进入截获距离 → 搜索 → 按概率锁定 → 末段。
            if not itc.acquired:
                victim = self._nearest_alive_target(itc.position)
                if (
                    victim is not None
                    and itc.position.distance_to(victim.position)
                    <= itc.acquisition_range
                ):
                    itc.phase = Phase.SEARCH
                    if self.rng.random() <= itc.acquisition_prob:
                        itc.acquired = True
                        itc.locked_target_id = victim.target_id
                        itc.phase = Phase.TERMINAL
                        self._log(
                            "锁定",
                            f"{itc.interceptor_id} 末段锁定 {victim.target_id}",
                        )

            if itc.acquired:
                continue  # 末段制导由 _fly 接管

            # 起飞抵近 / 搜索阶段:指令制导,跟随上行航迹。
            trk = track_map.get(itc.target_track_id)
            if not isinstance(trk, Track):
                continue
            t = lead_intercept_time(
                itc.position, trk.position, trk.velocity, itc.max_speed
            )
            if t is None:
                continue
            itc.steer_to(trk.position + trk.velocity * t, self.now + t)

    # -- 子过程 -----------------------------------------------------------

    def _resolve_detonations(self) -> None:
        """对本步引爆的 Thunder 判定毁伤,并清理弹体。"""
        s = self.s
        for itc in self.thunders:
            if not itc.detonated or not itc.alive:
                continue
            itc.alive = False
            self._release_engagement(itc)

            # 毁伤判定基于近炸引信记录的真实脱靶量。
            victim = self._target_by_id(itc.locked_target_id)
            if victim is None or not victim.alive:
                self._log("脱靶", f"{itc.interceptor_id} 引爆时目标已失")
                continue
            miss = itc.miss_distance
            if miss <= itc.lethal_radius and self.rng.random() <= s.single_shot_pk:
                victim.alive = False
                self.result.destroyed.append(victim.target_id)
                self._log(
                    "摧毁",
                    f"{itc.interceptor_id} 摧毁 {victim.target_id} "
                    f"(脱靶量 {miss:.1f}m)",
                )
            else:
                self._log(
                    "脱靶",
                    f"{itc.interceptor_id} 未命中 {victim.target_id} "
                    f"(脱靶量 {miss:.1f}m)",
                )

        self.thunders = [i for i in self.thunders if i.alive]

    def _check_leaks(self) -> None:
        """进入"安全穹顶"防护半径仍存活的目标判为突防。"""
        s = self.s
        r = s.threat_policy.defended_radius
        for tgt in s.targets:
            if tgt.alive and tgt.position.distance_to(s.asset) <= r:
                tgt.alive = False
                self.result.leaked.append(tgt.target_id)
                self._log("突防", f"{tgt.target_id} 突入安全穹顶")

    def _nearest_alive_target(self, point: Vec3) -> Target | None:
        best: Target | None = None
        best_d = float("inf")
        for tgt in self.s.targets:
            if not tgt.alive:
                continue
            d = point.distance_to(tgt.position)
            if d < best_d:
                best_d = d
                best = tgt
        return best

    def _log(self, kind: str, detail: str) -> None:
        self.result.events.append(Event(self.now, kind, detail))

    # -- 主循环 -----------------------------------------------------------

    def _active_targets(self) -> int:
        return sum(1 for t in self.s.targets if t.alive)

    def run(self) -> SimResult:
        """运行至所有目标被处置或达到最长时长。"""
        while self.now < self.s.max_time:
            self.step()
            if self._active_targets() == 0 and not self.thunders:
                break
        self.result.duration = self.now
        self.result.unresolved = [t.target_id for t in self.s.targets if t.alive]
        return self.result
