"""拦截指令生成(火力-目标分配)。

依据威胁研判结果,把 Thunder 发射单元的有限拦截弹分配给威胁航迹,
为每次交战解算**预测拦截点**与命中时刻,并下发 :class:`Command`。

分配策略:按威胁分从高到低贪心;每条航迹据其威胁等级确定齐射弹数
(shoot-look 简化为按等级配弹);为每发拦截弹挑选能覆盖该拦截点、
且剩余库存最多的发射单元。无可用资源时下发 HOLD 指令。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from c2sim.geometry import Vec3, lead_intercept_time
from c2sim.models import (
    Command,
    CommandKind,
    ThreatAssessment,
    ThreatLevel,
    Track,
    next_id,
)
from c2sim.weapons import Interceptor, ThunderBattery


@dataclass
class EngagementPolicy:
    """交战规则参数。"""

    engage_level: ThreatLevel = ThreatLevel.MEDIUM  # 达到此等级方可交战
    salvo_by_level: dict[ThreatLevel, int] = field(
        default_factory=lambda: {
            ThreatLevel.MEDIUM: 1,
            ThreatLevel.HIGH: 1,
            ThreatLevel.CRITICAL: 2,  # 紧急目标双发齐射提高毁伤概率
        }
    )

    def salvo_size(self, level: ThreatLevel) -> int:
        return self.salvo_by_level.get(level, 0)


@dataclass
class _Solution:
    battery: ThunderBattery
    intercept_point: Vec3
    flight_time: float


def _solve(battery: ThunderBattery, track: Track, now: float) -> _Solution | None:
    """解算某发射单元对某航迹的拦截诸元;不可达返回 None。"""
    t = lead_intercept_time(
        battery.position, track.position, track.velocity, battery.interceptor_speed
    )
    if t is None:
        return None
    intercept_point = track.position + track.velocity * t
    if not battery.can_reach(intercept_point):
        return None
    return _Solution(battery=battery, intercept_point=intercept_point, flight_time=t)


def _best_battery(
    batteries: list[ThunderBattery], track: Track, now: float
) -> _Solution | None:
    """在所有可达发射单元中择优:优先库存多,其次飞行时间短。"""
    solutions = [s for b in batteries if (s := _solve(b, track, now)) is not None]
    if not solutions:
        return None
    solutions.sort(key=lambda s: (-s.battery.inventory, s.flight_time))
    return solutions[0]


def plan_and_fire(
    assessments: list[ThreatAssessment],
    tracks: dict[str, Track],
    batteries: list[ThunderBattery],
    policy: EngagementPolicy,
    now: float,
    engaged_counts: dict[str, int],
) -> tuple[list[Command], list[Interceptor]]:
    """生成拦截指令并实施发射。

    参数:
        assessments: 已按威胁分降序排列的研判结果。
        tracks: 航迹查找表(track_id → Track)。
        batteries: 可调度的 Thunder 发射单元(库存会被原地扣减)。
        policy: 交战规则。
        now: 当前仿真时间。
        engaged_counts: 各航迹已承诺的拦截弹数(跨帧累计),原地更新,
            避免对同一目标重复过度交战。

    返回:
        ``(commands, interceptors)`` —— 本帧下发的指令与新发射的拦截弹。
    """
    commands: list[Command] = []
    interceptors: list[Interceptor] = []

    for assessment in assessments:
        if assessment.level < policy.engage_level:
            continue
        track = tracks.get(assessment.track_id)
        if track is None:
            continue

        desired = policy.salvo_size(assessment.level)
        already = engaged_counts.get(track.track_id, 0)
        needed = desired - already
        if needed <= 0:
            continue

        for _ in range(needed):
            sol = _best_battery(batteries, track, now)
            if sol is None:
                commands.append(
                    Command(
                        command_id=next_id("CMD"),
                        kind=CommandKind.HOLD,
                        timestamp=now,
                        track_id=track.track_id,
                        battery_id=None,
                        intercept_point=None,
                        intercept_time=None,
                        note="无可用拦截资源或超出射界,暂缓交战",
                    )
                )
                break

            intercept_time = now + sol.flight_time
            interceptor = sol.battery.fire(
                track.track_id, sol.intercept_point, intercept_time, now
            )
            interceptors.append(interceptor)
            engaged_counts[track.track_id] = (
                engaged_counts.get(track.track_id, 0) + 1
            )
            commands.append(
                Command(
                    command_id=next_id("CMD"),
                    kind=CommandKind.ENGAGE,
                    timestamp=now,
                    track_id=track.track_id,
                    battery_id=sol.battery.battery_id,
                    intercept_point=sol.intercept_point,
                    intercept_time=intercept_time,
                    note=(
                        f"{assessment.level.label}威胁 → {sol.battery.battery_id} "
                        f"发射 {interceptor.interceptor_id}, "
                        f"飞行 {sol.flight_time:.1f}s"
                    ),
                )
            )

    return commands, interceptors
