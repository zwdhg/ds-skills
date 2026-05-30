"""制导律实现(:class:`c2sim.strategies.GuidanceLaw`)。

提供两种可互换的制导律,演示开闭原则——引擎不变,注入不同制导律即可:

* :class:`LeadPursuitGuidance` —— 领先追踪(比例预测拦截点),系统默认;
* :class:`PurePursuitGuidance` —— 纯追踪(始终指向目标当前位置),对比基线。
"""

from __future__ import annotations

from c2sim.geometry import Vec3, lead_intercept_time


class LeadPursuitGuidance:
    """领先追踪:解算定速命中匀速目标的预测拦截点。"""

    def aim(
        self, shooter: Vec3, speed: float, target_pos: Vec3, target_vel: Vec3
    ) -> tuple[Vec3, float] | None:
        t = lead_intercept_time(shooter, target_pos, target_vel, speed)
        if t is None:
            return None
        return target_pos + target_vel * t, t


class PurePursuitGuidance:
    """纯追踪:始终瞄准目标当前位置(不做领先量),作为对比基线。

    对横向高速目标效率低于领先追踪,但实现简单、鲁棒——用于算法对比研究。
    """

    def aim(
        self, shooter: Vec3, speed: float, target_pos: Vec3, target_vel: Vec3
    ) -> tuple[Vec3, float] | None:
        if speed <= 0.0:
            return None
        d = shooter.distance_to(target_pos)
        return target_pos, d / speed
