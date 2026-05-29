"""仿真引擎:把全链路按时间步推进。

每个仿真步(tick)依次执行:

1. 推进来袭目标(真值)与在飞拦截弹;
2. 处理拦截弹引爆,按其与目标真值位置之差与杀伤半径判定毁伤;
3. 传感器对目标产生带噪量测;
4. 航迹融合 → 威胁研判 → 拦截指令生成 → Thunder 发射;
5. 检查目标是否突防(进入要地防护半径)及战斗结束条件。

引擎掌握"真值"(目标真实位置、毁伤判定),而指控链路只能看到航迹——
这一信息隔离是整套仿真可信度的关键。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from c2sim.fusion import TrackFusion
from c2sim.geometry import Vec3, lead_intercept_time
from c2sim.interception import EngagementPolicy, plan_and_fire
from c2sim.models import Command, Target
from c2sim.sensors import Radar
from c2sim.threat import ThreatPolicy, assess
from c2sim.weapons import Interceptor, ThunderBattery


@dataclass
class Scenario:
    """一次演练想定。"""

    asset: Vec3                       # 被掩护要地位置
    targets: list[Target]
    radars: list[Radar]
    batteries: list[ThunderBattery]
    dt: float = 0.5                   # 仿真步长(秒)
    max_time: float = 600.0           # 最长仿真时长(秒)
    seed: int = 1234
    threat_policy: ThreatPolicy = field(default_factory=ThreatPolicy)
    engagement_policy: EngagementPolicy = field(default_factory=EngagementPolicy)
    single_shot_pk: float = 0.85      # 单发杀伤概率(命中波门内时)


@dataclass
class Event:
    """带时间戳的仿真事件,供复盘。"""

    time: float
    kind: str
    detail: str


@dataclass
class SimResult:
    """一次仿真的结果汇总。"""

    destroyed: list[str] = field(default_factory=list)   # 被摧毁目标编号
    leaked: list[str] = field(default_factory=list)      # 突防目标编号
    unresolved: list[str] = field(default_factory=list)  # 仿真结束时仍在途的目标
    total_targets: int = 0
    interceptors_fired: int = 0
    commands: list[Command] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    duration: float = 0.0

    def summary(self) -> str:
        tail = f" | 在途未决 {len(self.unresolved)}" if self.unresolved else ""
        return (
            f"用时 {self.duration:.1f}s | 来袭目标 {self.total_targets} | "
            f"摧毁 {len(self.destroyed)} | 突防 {len(self.leaked)}{tail} | "
            f"发射 Thunder {self.interceptors_fired} 发"
        )


class Engine:
    """指挥控制仿真引擎。"""

    def __init__(self, scenario: Scenario) -> None:
        self.s = scenario
        self.rng = random.Random(scenario.seed)
        self.fusion = TrackFusion()
        self.interceptors: list[Interceptor] = []
        self.engaged_counts: dict[str, int] = {}
        self.now = 0.0
        self.result = SimResult(total_targets=len(scenario.targets))

    # -- 单步 -------------------------------------------------------------

    def step(self) -> None:
        s = self.s
        dt = s.dt
        step_start = self.now
        self.now += dt

        # 1) 推进真值与拦截弹(拦截弹按上一步设定的制导矢量飞行)。
        for tgt in s.targets:
            if tgt.alive:
                tgt.advance(dt)
        for itc in self.interceptors:
            itc.advance(dt, step_start)

        # 2) 处理引爆与毁伤。
        self._resolve_detonations()

        # 3) 突防判定。
        self._check_leaks()

        # 4) 传感器量测。
        reports = []
        for radar in s.radars:
            reports.extend(radar.observe(s.targets, self.now, self.rng))

        # 5) 航迹融合。
        tracks = self.fusion.update(reports, self.now)
        track_map = {t.track_id: t for t in tracks}

        # 已消失的航迹释放其交战占用,允许对存活目标重新交战。
        for tid in list(self.engaged_counts):
            if tid not in track_map:
                del self.engaged_counts[tid]

        # 6) 威胁研判。
        assessments = assess(tracks, s.asset, s.threat_policy)

        # 7) 中段制导:用最新航迹重新解算在飞拦截弹的拦截诸元。
        self._guide(track_map)

        # 8) 拦截指令生成与发射。
        commands, new_interceptors = plan_and_fire(
            assessments,
            track_map,
            s.batteries,
            s.engagement_policy,
            self.now,
            self.engaged_counts,
        )
        for cmd in commands:
            self.result.commands.append(cmd)
            if cmd.kind.value == "engage":
                self._log("交战", cmd.note)
        self.interceptors.extend(new_interceptors)
        self.result.interceptors_fired += len(new_interceptors)

    # -- 子过程 -----------------------------------------------------------

    def _resolve_detonations(self) -> None:
        """对本步引爆的拦截弹判定毁伤,并清理弹体。"""
        s = self.s
        for itc in self.interceptors:
            if not itc.detonated or not itc.alive:
                continue
            itc.alive = False
            # 释放该航迹的一次交战占用。
            if itc.target_track_id in self.engaged_counts:
                self.engaged_counts[itc.target_track_id] = max(
                    0, self.engaged_counts[itc.target_track_id] - 1
                )

            # 引爆点附近的存活目标即可能受毁伤(以真值判定)。
            victim = self._nearest_alive_target(itc.position)
            if victim is None:
                self._log("脱靶", f"{itc.interceptor_id} 附近无目标")
                continue
            miss = itc.position.distance_to(victim.position)
            if miss <= itc.lethal_radius and self.rng.random() <= s.single_shot_pk:
                victim.alive = False
                self.result.destroyed.append(victim.target_id)
                self._log(
                    "摧毁",
                    f"{itc.interceptor_id} 摧毁 {victim.target_id} "
                    f"(脱靶量 {miss:.0f}m)",
                )
            else:
                self._log(
                    "脱靶",
                    f"{itc.interceptor_id} 未命中 {victim.target_id} "
                    f"(脱靶量 {miss:.0f}m)",
                )

        self.interceptors = [i for i in self.interceptors if i.alive]

    def _guide(self, track_map: dict[str, "object"]) -> None:
        """拦截弹制导。

        分两段:**末段**——若进入导引头截获距离,弹上导引头直接锁定附近
        真实目标(带导引头噪声)精确寻的;**中段**——否则依据指控链路上行
        的最新航迹做指令制导。两段均无解或航迹丢失时,保持惯性飞行至既定
        引爆时刻。
        """
        from c2sim.models import Track  # 局部导入避免循环依赖噪声

        for itc in self.interceptors:
            if itc.detonated or not itc.alive:
                continue

            # 末段寻的:导引头锁定基准内最近的真实目标。
            victim = self._nearest_alive_target(itc.position)
            if (
                victim is not None
                and itc.position.distance_to(victim.position) <= itc.terminal_range
            ):
                sigma = itc.seeker_sigma
                seen = Vec3(
                    victim.position.x + self.rng.gauss(0.0, sigma),
                    victim.position.y + self.rng.gauss(0.0, sigma),
                    victim.position.z + self.rng.gauss(0.0, sigma),
                )
                t = lead_intercept_time(itc.position, seen, victim.velocity, itc.speed)
                if t is not None:
                    itc.steer_to(seen + victim.velocity * t, self.now + t)
                    continue

            # 中段指令制导:跟随上行航迹。
            trk = track_map.get(itc.target_track_id)
            if not isinstance(trk, Track):
                continue
            t = lead_intercept_time(
                itc.position, trk.position, trk.velocity, itc.speed
            )
            if t is None:
                continue
            aim = trk.position + trk.velocity * t
            itc.steer_to(aim, self.now + t)

    def _check_leaks(self) -> None:
        """进入要地防护半径仍存活的目标判为突防。"""
        s = self.s
        r = s.threat_policy.defended_radius
        for tgt in s.targets:
            if tgt.alive and tgt.position.distance_to(s.asset) <= r:
                tgt.alive = False
                self.result.leaked.append(tgt.target_id)
                self._log("突防", f"{tgt.target_id} 突入要地防护圈")

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
            if self._active_targets() == 0 and not self.interceptors:
                break
        self.result.duration = self.now
        self.result.unresolved = [
            t.target_id for t in self.s.targets if t.alive
        ]
        return self.result
