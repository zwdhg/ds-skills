"""策略协议(依赖倒置接缝)。

把指控链路中**可替换的算法**抽象为协议(``typing.Protocol``),使高层编排
(:class:`c2sim.engine.Engine`)依赖抽象而非具体实现,从而:

* **OCP**:新增跟踪器 / 研判模型 / 火力分配 / 制导律 / 传感器无需修改引擎,
  只需提供一个符合协议的实现并注入;
* **DIP**:引擎依赖这些协议,具体实现在组装处注入;
* **LSP**:各实现须满足协议契约(见 ``tests/test_strategies.py`` 的契约测试)。

协议刻意保持最小、贴近现有调用形态,默认实现见各自模块:

* :class:`Tracker`              → :class:`c2sim.fusion.TrackFusion`
* :class:`ThreatModel`          → :class:`c2sim.threat.WeightedThreatModel`
* :class:`WeaponTargetAssigner` → :class:`c2sim.interception.GreedyAssigner`
* :class:`GuidanceLaw`          → :class:`c2sim.guidance.LeadPursuitGuidance`
* :class:`SensorModel`          → :class:`c2sim.sensors.SpotterPro`
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from c2sim.geometry import Vec3
from c2sim.models import (
    Command,
    SensorReport,
    Target,
    ThreatAssessment,
    Track,
)


@runtime_checkable
class SensorModel(Protocol):
    """对一批目标真值产出带噪量测。"""

    def observe(
        self, targets: list[Target], now: float, rng: random.Random
    ) -> list[SensorReport]: ...


@runtime_checkable
class Tracker(Protocol):
    """跨帧维护航迹:吸收本帧量测,返回当前存活航迹。"""

    def update(self, reports: list[SensorReport], now: float) -> list[Track]: ...


@runtime_checkable
class ThreatModel(Protocol):
    """对航迹相对被掩护要地做威胁研判,按威胁降序返回。"""

    def assess(
        self, tracks: list[Track], asset: Vec3
    ) -> list[ThreatAssessment]: ...


@runtime_checkable
class GuidanceLaw(Protocol):
    """制导律:给定射手与目标估计,解算瞄准点与剩余飞行时间。

    契约(LSP):返回 ``(aim_point, time_to_go)`` 时,``time_to_go >= 0``,
    且按 ``speed`` 飞行 ``time_to_go`` 秒所经距离应等于到 ``aim_point`` 的距离
    (即瞄准点自洽);不可达时返回 ``None``。
    """

    def aim(
        self, shooter: Vec3, speed: float, target_pos: Vec3, target_vel: Vec3
    ) -> tuple[Vec3, float] | None: ...


@runtime_checkable
class WeaponTargetAssigner(Protocol):
    """火力-目标分配:据研判结果调度发射平台,生成指令并发射 Thunder。"""

    def plan(
        self,
        assessments: list[ThreatAssessment],
        tracks: dict[str, Track],
        pads: list,
        now: float,
        engaged_counts: dict[str, int],
        skip_tracks: set[str] | None = None,
    ) -> tuple[list[Command], list]: ...
