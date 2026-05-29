"""预置演练想定。

提供一个有代表性的多目标来袭场景,便于一键演示与回归测试。所有数值
均为仿真参数,坐标以要地为原点。
"""

from __future__ import annotations

from c2sim.engine import Scenario
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.sensors import Radar
from c2sim.weapons import ThunderBattery


def _inbound(
    tid: str, start: Vec3, aim: Vec3, speed: float, kind: TargetKind, rcs: float
) -> Target:
    """构造一个朝 ``aim`` 点匀速飞来的目标。"""
    direction = (aim - start).unit()
    return Target(
        target_id=tid,
        position=start,
        velocity=direction * speed,
        kind=kind,
        rcs=rcs,
    )


def build_demo_scenario(seed: int = 2026) -> Scenario:
    """构建一个含 4 个异类来袭目标的混合突击想定。

    要地位于原点。两部雷达交叉布站形成重叠探测区(供航迹融合发挥),
    两个 Thunder 发射单元分别前出部署以扩展拦截纵深。
    """
    asset = Vec3(0.0, 0.0, 0.0)

    targets = [
        # 高空有人机,正面突入。
        _inbound(
            "T1-AIRCRAFT",
            start=Vec3(100_000.0, 5_000.0, 8_000.0),
            aim=asset,
            speed=240.0,
            kind=TargetKind.AIRCRAFT,
            rcs=5.0,
        ),
        # 低空巡航导弹,侧翼掠地突防。
        _inbound(
            "T2-CRUISE",
            start=Vec3(90_000.0, -40_000.0, 150.0),
            aim=asset,
            speed=280.0,
            kind=TargetKind.CRUISE_MISSILE,
            rcs=0.3,
        ),
        # 慢速小型无人机。
        _inbound(
            "T3-DRONE",
            start=Vec3(35_000.0, 22_000.0, 1_200.0),
            aim=asset,
            speed=120.0,
            kind=TargetKind.DRONE,
            rcs=0.05,
        ),
        # 高速俯冲弹道目标。
        _inbound(
            "T4-BALLISTIC",
            start=Vec3(70_000.0, 20_000.0, 60_000.0),
            aim=asset,
            speed=1800.0,
            kind=TargetKind.BALLISTIC,
            rcs=1.0,
        ),
    ]

    radars = [
        Radar("RDR-A", position=Vec3(8_000.0, 8_000.0, 30.0), max_range=130_000.0),
        Radar("RDR-B", position=Vec3(-8_000.0, -6_000.0, 30.0), max_range=130_000.0),
    ]

    batteries = [
        ThunderBattery(
            "BTY-1",
            position=Vec3(20_000.0, 10_000.0, 20.0),
            inventory=8,
            interceptor_speed=1200.0,
            max_range=70_000.0,
        ),
        ThunderBattery(
            "BTY-2",
            position=Vec3(15_000.0, -20_000.0, 20.0),
            inventory=8,
            interceptor_speed=1200.0,
            max_range=70_000.0,
        ),
    ]

    return Scenario(
        asset=asset,
        targets=targets,
        radars=radars,
        batteries=batteries,
        seed=seed,
    )
