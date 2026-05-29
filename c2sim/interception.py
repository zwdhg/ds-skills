"""拦截指令生成(火力-目标分配)。

对应 Skyshield Nexus 的拦截策略生成:依据威胁研判结果,结合各发射平台的
**分布式部署位置、作业半径与可用 Thunder 库存**,为威胁航迹自主优选最优
发射平台,解算预测拦截点与命中时刻,下发 :class:`Command`(ENGAGE/HOLD)。

分配策略:按威胁分从高到低贪心;每条航迹据其威胁等级确定齐射架数;为每架
Thunder 挑选**作业半径覆盖该拦截点、且剩余库存最多**的发射平台。无可用资源
或超出作业半径时下发 HOLD。
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
from c2sim.weapons import LaunchPad, Thunder


@dataclass
class EngagementPolicy:
    """交战规则参数。"""

    engage_level: ThreatLevel = ThreatLevel.MEDIUM
    prefer_jamming: bool = True  # RF 辐射目标优先软杀伤(干扰),节省 Thunder
    # 拦截点须落在作业半径该比例以内才开火,避免贴边远射追不上高速目标。
    engage_radius_frac: float = 0.8
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
    pad: LaunchPad
    intercept_point: Vec3
    flight_time: float


def _solve(pad: LaunchPad, track: Track, radius_frac: float) -> _Solution | None:
    """解算某发射平台对某航迹的拦截诸元;不可达或贴边远射返回 None。"""
    t = lead_intercept_time(
        pad.position, track.position, track.velocity, pad.thunder_max_speed
    )
    if t is None:
        return None
    intercept_point = track.position + track.velocity * t
    if pad.inventory <= 0:
        return None
    if pad.position.distance_to(intercept_point) > pad.operating_radius * radius_frac:
        return None
    return _Solution(pad=pad, intercept_point=intercept_point, flight_time=t)


def _best_pad(
    pads: list[LaunchPad], track: Track, radius_frac: float
) -> _Solution | None:
    """在所有可达发射平台中择优:优先库存多,其次飞行时间短。"""
    solutions = [s for p in pads if (s := _solve(p, track, radius_frac)) is not None]
    if not solutions:
        return None
    solutions.sort(key=lambda s: (-s.pad.inventory, s.flight_time))
    return solutions[0]


def plan_and_fire(
    assessments: list[ThreatAssessment],
    tracks: dict[str, Track],
    pads: list[LaunchPad],
    policy: EngagementPolicy,
    now: float,
    engaged_counts: dict[str, int],
    skip_tracks: set[str] | None = None,
) -> tuple[list[Command], list[Thunder]]:
    """生成拦截指令并实施发射(Thunder 硬杀伤)。

    参数:
        assessments: 已按威胁分降序排列的研判结果。
        tracks: 航迹查找表(track_id → Track)。
        pads: 可调度的发射平台(库存会被原地扣减)。
        policy: 交战规则。
        now: 当前仿真时间。
        engaged_counts: 各航迹已承诺的 Thunder 架数(跨帧累计),原地更新。
        skip_tracks: 已由软杀伤(干扰)处置、不再用 Thunder 交战的航迹集合。

    返回:
        ``(commands, thunders)`` —— 本帧下发的指令与新发射的 Thunder。
    """
    commands: list[Command] = []
    thunders: list[Thunder] = []
    skip = skip_tracks or set()

    for assessment in assessments:
        if assessment.level < policy.engage_level:
            continue
        if assessment.track_id in skip:
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
            sol = _best_pad(pads, track, policy.engage_radius_frac)
            if sol is None:
                commands.append(
                    Command(
                        command_id=next_id("CMD"),
                        kind=CommandKind.HOLD,
                        timestamp=now,
                        track_id=track.track_id,
                        pad_id=None,
                        intercept_point=None,
                        intercept_time=None,
                        note="无可用 Thunder 或超出作业半径,暂缓交战",
                    )
                )
                break

            intercept_time = now + sol.flight_time
            thunder = sol.pad.fire(
                track.track_id, sol.intercept_point, intercept_time, now
            )
            thunders.append(thunder)
            engaged_counts[track.track_id] = (
                engaged_counts.get(track.track_id, 0) + 1
            )
            commands.append(
                Command(
                    command_id=next_id("CMD"),
                    kind=CommandKind.ENGAGE,
                    timestamp=now,
                    track_id=track.track_id,
                    pad_id=sol.pad.pad_id,
                    intercept_point=sol.intercept_point,
                    intercept_time=intercept_time,
                    note=(
                        f"{assessment.level.label}威胁 → {sol.pad.pad_id} "
                        f"发射 {thunder.interceptor_id}, 飞行 {sol.flight_time:.1f}s"
                    ),
                )
            )

    return commands, thunders
