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
from c2sim.geometry import Vec3
from c2sim.guidance import LeadPursuitGuidance
from c2sim.interception import EngagementPolicy, GreedyAssigner
from c2sim.models import Command, CommandKind, Target, next_id
from c2sim.sensors import SpotterPro
from c2sim.strategies import (
    GuidanceLaw,
    ThreatModel,
    Tracker,
    WeaponTargetAssigner,
)
from c2sim.threat import ThreatPolicy, WeightedThreatModel
from c2sim.weapons import HunterMax, LaunchPad, Phase, Thunder
from c2sim.world import World


@dataclass
class Scenario:
    """一次演练想定。"""

    asset: Vec3                       # 被掩护要地("安全穹顶"中心)
    targets: list[Target]
    spotters: list[SpotterPro]
    pads: list[LaunchPad]
    jammers: list[HunterMax] = field(default_factory=list)  # Hunter Max 干扰单元
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
class History:
    """轻量航迹历史,供态势可视化。"""

    target_paths: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    thunder_paths: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    # 事件标记:(x, y, kind),kind ∈ {摧毁, 软杀伤, 突防}
    markers: list[tuple[float, float, str]] = field(default_factory=list)


@dataclass
class SimResult:
    """一次仿真的结果汇总。"""

    destroyed: list[str] = field(default_factory=list)      # Thunder 硬杀伤
    soft_killed: list[str] = field(default_factory=list)    # Hunter Max 软杀伤
    leaked: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    total_targets: int = 0
    thunders_launched: int = 0
    commands: list[Command] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    duration: float = 0.0

    def summary(self) -> str:
        soft = f" | 软杀伤 {len(self.soft_killed)}" if self.soft_killed else ""
        tail = f" | 在途未决 {len(self.unresolved)}" if self.unresolved else ""
        return (
            f"用时 {self.duration:.1f}s | 来袭目标 {self.total_targets} | "
            f"摧毁 {len(self.destroyed)}{soft} | 突防 {len(self.leaked)}{tail} | "
            f"发射 Thunder {self.thunders_launched} 架"
        )


@dataclass
class Trace:
    """仿真记录(结果 + 历史),供 CLI、可视化等消费。

    可视化/分析只需依赖本聚合的只读视图,无需依赖整个 :class:`Engine`(ISP)。
    """

    result: SimResult
    history: History


class Engine:
    """Skyshield Nexus 指控仿真:编排"被控对象(World)"与"控制器(策略)"。

    职责仅为**编排与控制**:按时间步驱动 World 物理、调用注入的传感/跟踪/
    研判/分配/制导策略、维护交战与干扰台账、把 World 产出的结果事件翻译为
    统计与历史。真值物理全部在 :class:`c2sim.world.World`。
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        tracker: Tracker | None = None,
        threat_model: ThreatModel | None = None,
        assigner: WeaponTargetAssigner | None = None,
        guidance: GuidanceLaw | None = None,
    ) -> None:
        self.s = scenario
        self.rng = random.Random(scenario.seed)
        self.world = World(
            targets=scenario.targets,
            jammers=scenario.jammers,
            asset=scenario.asset,
            defended_radius=scenario.threat_policy.defended_radius,
            single_shot_pk=scenario.single_shot_pk,
            rng=self.rng,
        )
        # 可替换策略(依赖倒置):默认即现有实现,可在组装处注入其它算法。
        self.tracker: Tracker = tracker or TrackFusion(
            gate_distance=600.0, max_coast=6.0
        )
        self.threat_model: ThreatModel = threat_model or WeightedThreatModel(
            scenario.threat_policy
        )
        self.assigner: WeaponTargetAssigner = assigner or GreedyAssigner(
            scenario.engagement_policy
        )
        self.guidance: GuidanceLaw = guidance or LeadPursuitGuidance()

        self.engaged_counts: dict[str, int] = {}
        self.jammed_tracks: dict[str, str] = {}  # track_id → 被干扰的真实目标 id
        self.now = 0.0
        self.trace = Trace(
            result=SimResult(total_targets=len(scenario.targets)),
            history=History(),
        )

    # 便捷只读视图。
    @property
    def result(self) -> SimResult:
        return self.trace.result

    @property
    def history(self) -> History:
        return self.trace.history

    # -- 单步 -------------------------------------------------------------

    def step(self) -> None:
        s = self.s
        dt = s.dt
        self.now += dt

        # 1) 末段制导(零延迟,先于积分;消耗导引头随机数)。
        self._guide_terminal()

        # 2) World 物理:推进真值与弹道,结算引信/软杀伤/突防。
        self.world.advance_targets(dt)
        self._apply(self.world.integrate_thunders(dt))
        self._apply(self.world.resolve_jamming(dt))
        self._apply(self.world.resolve_detonations())
        self._apply(self.world.check_leaks())
        self.world.reindex()  # 重建就近查询索引(供制导/干扰决策)

        # 3) Spotter Pro 多模态探测 → 航迹融合。
        reports = []
        for spotter in s.spotters:
            reports.extend(spotter.observe(self.world.targets, self.now, self.rng))
        tracks = self.tracker.update(reports, self.now)
        track_map = {t.track_id: t for t in tracks}

        # 已消失的航迹释放其交战/干扰占用。
        for tid in list(self.engaged_counts):
            if tid not in track_map:
                del self.engaged_counts[tid]
        for tid in list(self.jammed_tracks):
            if tid not in track_map:
                del self.jammed_tracks[tid]

        # 4) 威胁研判。
        assessments = self.threat_model.assess(tracks, s.asset)

        # 5) 软杀伤决策(RF 辐射目标优先干扰,节省 Thunder)。
        skip = self._plan_jamming(assessments, track_map)

        # 6) 目标搜索/锁定 + 起飞抵近段指令制导(消耗截获随机数)。
        self._guide_search(track_map)

        # 7) 拦截指令生成与发射(跳过已交由软杀伤处置的航迹)。
        commands, new_thunders = self.assigner.plan(
            assessments, track_map, s.pads, self.now, self.engaged_counts,
            skip_tracks=skip,
        )
        for cmd in commands:
            self.result.commands.append(cmd)
            if cmd.kind == CommandKind.ENGAGE:
                self._log("交战", cmd.note)
        self.world.add_thunders(new_thunders)
        self.result.thunders_launched += len(new_thunders)

        # 8) 记录航迹历史(可视化)。
        self._record_history()

    # -- 控制器:制导 ----------------------------------------------------

    def _guide_terminal(self) -> None:
        """末段拦截:对已锁定 Thunder,以所锁真实目标(带导引头噪声)寻的。"""
        for itc in self.world.thunders:
            if itc.detonated or not itc.alive or not itc.acquired:
                continue
            victim = self.world.target_by_id(itc.locked_target_id)
            if victim is None or not victim.alive:
                continue
            seen = self.world.seeker_fix(victim, itc.seeker_sigma)
            sol = self.guidance.aim(
                itc.position, itc.max_speed, seen, victim.velocity
            )
            if sol is not None:
                itc.steer_to(sol[0], self.now + sol[1])

    def _guide_search(self, track_map: dict[str, object]) -> None:
        """起飞抵近(指令制导)+ 目标搜索(按概率锁定真实目标)。"""
        from c2sim.models import Track

        for itc in self.world.thunders:
            if itc.detonated or not itc.alive:
                continue
            if not itc.acquired:
                victim = self.world.nearest_alive_target(itc.position)
                if (victim is not None
                        and itc.position.distance_to(victim.position)
                        <= itc.acquisition_range):
                    itc.phase = Phase.SEARCH
                    if self.rng.random() <= itc.acquisition_prob:
                        itc.acquired = True
                        itc.locked_target_id = victim.target_id
                        itc.phase = Phase.TERMINAL
                        self._log("锁定",
                                  f"{itc.interceptor_id} 末段锁定 {victim.target_id}")
            if itc.acquired:
                continue  # 末段由 _guide_terminal 接管
            trk = track_map.get(itc.target_track_id)
            if not isinstance(trk, Track):
                continue
            sol = self.guidance.aim(
                itc.position, itc.max_speed, trk.position, trk.velocity
            )
            if sol is None:
                continue
            itc.steer_to(sol[0], self.now + sol[1])

    # -- 控制器:软杀伤决策 ----------------------------------------------

    def _plan_jamming(self, assessments, track_map: dict[str, object]) -> set[str]:
        """对 RF 辐射、落入 Hunter Max 干扰圈的威胁航迹调度软杀伤。"""
        s = self.s
        skip: set[str] = set(self.jammed_tracks)
        if not s.jammers or not s.engagement_policy.prefer_jamming:
            return skip
        for a in assessments:
            if a.level < s.engagement_policy.engage_level:
                continue
            if a.track_id in self.jammed_tracks:
                continue
            trk = track_map.get(a.track_id)
            if trk is None or not getattr(trk, "rf_emitter", False):
                continue
            jammer = next((j for j in s.jammers if j.covers(trk.position)), None)
            if jammer is None:
                continue
            victim = self.world.nearest_jammable(trk.position, jammer)
            if victim is None:
                continue
            victim.jammed = True
            self.jammed_tracks[a.track_id] = victim.target_id
            skip.add(a.track_id)
            self.result.commands.append(Command(
                command_id=next_id("CMD"),
                kind=CommandKind.JAM,
                timestamp=self.now,
                track_id=a.track_id,
                jammer_id=jammer.jammer_id,
                note=f"{a.level.label}威胁(RF 辐射)→ {jammer.jammer_id} 实施干扰软杀伤",
            ))
            self._log("干扰", f"{jammer.jammer_id} 干扰 {victim.target_id}")
        return skip

    # -- 结果事件 → 统计/历史/台账 --------------------------------------

    def _apply(self, events: list) -> None:
        from c2sim.world import Kill, Leak, Miss, SelfDestruct, SoftKill

        for ev in events:
            if isinstance(ev, Kill):
                self._release(ev.track_id)
                self.result.destroyed.append(ev.target_id)
                self.history.markers.append((
                    self._pos(ev.target_id)))
                self._log("摧毁",
                          f"{ev.interceptor_id} 摧毁 {ev.target_id} "
                          f"(脱靶量 {ev.miss:.1f}m)")
            elif isinstance(ev, Miss):
                self._release(ev.track_id)
                if ev.target_id is None:
                    self._log("脱靶", f"{ev.interceptor_id} 引爆时目标已失")
                else:
                    self._log("脱靶",
                              f"{ev.interceptor_id} 未命中 {ev.target_id} "
                              f"(脱靶量 {ev.miss:.1f}m)")
            elif isinstance(ev, SelfDestruct):
                self._release(ev.track_id)
                self._log("自毁", f"{ev.interceptor_id} 飞出作业半径,任务终止")
            elif isinstance(ev, SoftKill):
                self.result.soft_killed.append(ev.target_id)
                self.history.markers.append((ev.x, ev.y, "软杀伤"))
                self._log("软杀伤", f"{ev.target_id} 链路中断,迫降/返航")
            elif isinstance(ev, Leak):
                self.result.leaked.append(ev.target_id)
                self.history.markers.append((ev.x, ev.y, "突防"))
                self._log("突防", f"{ev.target_id} 突入安全穹顶")

    def _pos(self, target_id: str):
        t = self.world.target_by_id(target_id)
        return (t.position.x, t.position.y, "摧毁")

    def _release(self, track_id: str) -> None:
        if track_id in self.engaged_counts:
            self.engaged_counts[track_id] = max(0, self.engaged_counts[track_id] - 1)

    def _record_history(self) -> None:
        for tgt in self.world.targets:
            if tgt.alive:
                self.history.target_paths.setdefault(tgt.target_id, []).append(
                    (tgt.position.x, tgt.position.y))
        for itc in self.world.thunders:
            if itc.alive:
                self.history.thunder_paths.setdefault(
                    itc.interceptor_id, []).append((itc.position.x, itc.position.y))

    def _log(self, kind: str, detail: str) -> None:
        self.result.events.append(Event(self.now, kind, detail))

    # -- 主循环 -----------------------------------------------------------

    def run(self) -> SimResult:
        """运行至所有目标被处置或达到最长时长。"""
        while self.now < self.s.max_time:
            self.step()
            if self.world.active_target_count() == 0 and not self.world.thunders:
                break
        self.result.duration = self.now
        self.result.unresolved = self.world.alive_target_ids()
        return self.result
