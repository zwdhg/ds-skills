"""预置演练想定 —— 对应参考系统的两种典型部署。

* :func:`build_point_defense_scenario` —— 核心要域点状防护(360° 安全穹顶);
* :func:`build_border_band_scenario` —— 边境线带状防护(多站 180° 扇区)。

所有数值为仿真参数,量级对齐参考系统:探测 ~10km、安全穹顶 5km、
Thunder ≈67m/s / 作业半径 5km,目标为低空小型无人机。
"""

from __future__ import annotations

import math

from c2sim.engine import Scenario
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.sensors import SpotterPro
from c2sim.weapons import HunterMax, LaunchPad


def _inbound(
    tid: str,
    aim: Vec3,
    start: Vec3,
    cruise: float,
    kind: TargetKind,
    rcs: float,
    emits_rf: bool = True,
    terminal_speed: float | None = None,
) -> Target:
    return Target(
        target_id=tid,
        position=start,
        aim=aim,
        cruise_speed=cruise,
        kind=kind,
        rcs=rcs,
        emits_rf=emits_rf,
        terminal_speed=terminal_speed,
    )


def _polar(cx: float, cy: float, az_deg: float, r: float, z: float) -> Vec3:
    a = math.radians(az_deg)
    return Vec3(cx + r * math.cos(a), cy + r * math.sin(a), z)


def build_point_defense_scenario(seed: int = 2026) -> Scenario:
    """核心要域点状防护:中心单站 Spotter Pro 360°,Thunder 前置分布式部署。

    四个异类无人机从不同方位低空来袭,其中含一枚 RF 静默的巡飞弹
    (仅雷达/光电可探)与一架微型小 RCS 目标。
    """
    asset = Vec3(0.0, 0.0, 0.0)

    spotters = [
        SpotterPro("SPT-0", position=Vec3(0.0, 0.0, 20.0), azimuth_width_deg=360.0),
    ]

    # Thunder 发射平台前置分布式部署(距中心 3km,作业半径 5km → 可外推至
    # ~8km),四象限布置以覆盖各来袭轴向。
    pads = [
        LaunchPad("PAD-E", _polar(0, 0, 0, 3_000, 15)),
        LaunchPad("PAD-NW", _polar(0, 0, 135, 3_000, 15)),
        LaunchPad("PAD-SW", _polar(0, 0, 225, 3_000, 15)),
        LaunchPad("PAD-SE", _polar(0, 0, 315, 3_000, 15)),
    ]

    # Hunter Max 干扰单元前置部署,对 RF 辐射目标实施软杀伤(节省 Thunder);
    # 干扰圈外推至 ~7.5km,使 RF 制式目标在突入穹顶前即被迫降/返航。
    jammers = [
        HunterMax("HM-E", _polar(0, 0, 0, 3_500, 15), jam_range=4_000.0),
        HunterMax("HM-SW", _polar(0, 0, 215, 3_500, 15), jam_range=4_000.0),
        HunterMax("HM-SE", _polar(0, 0, 315, 3_500, 15), jam_range=4_000.0),
    ]

    targets = [
        # 固定翼无人机,正东来袭(辐射图传信号,可被频谱测向预警)。
        _inbound(
            "T1-FIXEDWING",
            aim=asset,
            start=_polar(0, 0, 0, 9_500, 600),
            cruise=48.0,
            kind=TargetKind.FIXED_WING_UAV,
            rcs=0.15,
        ),
        # 巡飞弹,西北来袭,RF 静默(GPS 自主),末段加速突击。
        _inbound(
            "T2-LOITER",
            aim=asset,
            start=_polar(0, 0, 140, 10_500, 400),
            cruise=52.0,
            kind=TargetKind.LOITERING_MUNITION,
            rcs=0.2,
            emits_rf=False,
            terminal_speed=58.0,
        ),
        # 旋翼无人机,西南来袭,慢速低空。
        _inbound(
            "T3-ROTARY",
            aim=asset,
            start=_polar(0, 0, 215, 7_000, 300),
            cruise=18.0,
            kind=TargetKind.ROTARY_UAV,
            rcs=0.08,
        ),
        # 微型无人机,东南来袭,极小 RCS(雷达探测距离缩短)。
        _inbound(
            "T4-MICRO",
            aim=asset,
            start=_polar(0, 0, 315, 6_500, 250),
            cruise=22.0,
            kind=TargetKind.MICRO_UAV,
            rcs=0.02,
        ),
    ]

    return Scenario(
        asset=asset, targets=targets, spotters=spotters, pads=pads,
        jammers=jammers, seed=seed,
    )


def build_border_band_scenario(seed: int = 2026) -> Scenario:
    """边境线带状防护:沿 y 轴布设多站 Spotter Pro(180° 扇区朝向境外 +x)。

    被掩护一侧在 x<0;多架无人机自境外(+x)低空渗透突防。
    """
    asset = Vec3(-3_000.0, 0.0, 0.0)  # 被掩护侧参考点

    spotters = [
        SpotterPro(
            f"SPT-{i}",
            position=Vec3(0.0, y, 20.0),
            azimuth_center_deg=0.0,
            azimuth_width_deg=180.0,
        )
        for i, y in enumerate((-6_000.0, 0.0, 6_000.0))
    ]

    pads = [
        LaunchPad("PAD-N", Vec3(-1_000.0, 4_000.0, 15.0)),
        LaunchPad("PAD-C", Vec3(-1_000.0, 0.0, 15.0)),
        LaunchPad("PAD-S", Vec3(-1_000.0, -4_000.0, 15.0)),
    ]

    targets = [
        _inbound(
            "B1-FIXEDWING",
            aim=Vec3(-3_000.0, 1_000.0, 0.0),
            start=Vec3(8_000.0, 2_000.0, 500.0),
            cruise=50.0,
            kind=TargetKind.FIXED_WING_UAV,
            rcs=0.15,
        ),
        _inbound(
            "B2-LOITER",
            aim=Vec3(-3_000.0, -2_000.0, 0.0),
            start=Vec3(9_000.0, -3_000.0, 400.0),
            cruise=54.0,
            kind=TargetKind.LOITERING_MUNITION,
            rcs=0.2,
            emits_rf=False,
            terminal_speed=60.0,
        ),
        _inbound(
            "B3-FIXEDWING",
            aim=Vec3(-3_000.0, 5_000.0, 0.0),
            start=Vec3(8_500.0, 6_500.0, 550.0),
            cruise=46.0,
            kind=TargetKind.FIXED_WING_UAV,
            rcs=0.12,
        ),
    ]

    jammers = [
        HunterMax("HM-C", Vec3(-500.0, 0.0, 15.0), jam_range=4_500.0),
    ]

    return Scenario(
        asset=asset, targets=targets, spotters=spotters, pads=pads,
        jammers=jammers, seed=seed,
    )


def build_swarm_scenario(seed: int = 2026, n: int = 40) -> Scenario:
    """蜂群突击:``n`` 架无人机自四面八方来袭,考验体系规模与饱和应对。

    目标布局确定(随种子只变传感/毁伤噪声,便于蒙特卡洛)。防御兵力可观但
    可被饱和——雷达 TAS 容量(每站 6)与 Thunder 库存共同构成饱和瓶颈,故
    通常无法零突防,正是蜂群对抗要研究的问题。也用于规模/性能验证。
    """
    asset = Vec3(0.0, 0.0, 0.0)

    # 多站 360° 探测以提升聚合 TAS 容量(每站 TAS≥6)。
    spotters = [
        SpotterPro("SPT-C", Vec3(0.0, 0.0, 20.0), azimuth_width_deg=360.0),
        SpotterPro("SPT-N", Vec3(0.0, 4_000.0, 20.0), azimuth_width_deg=360.0),
        SpotterPro("SPT-S", Vec3(0.0, -4_000.0, 20.0), azimuth_width_deg=360.0),
    ]
    # 8 个发射平台环形前置,合计库存匹配蜂群规模。
    pads = [
        LaunchPad(f"PAD-{a:03d}", _polar(0, 0, a, 3_000, 15), inventory=6)
        for a in range(0, 360, 45)
    ]
    # 4 部 Hunter Max 覆盖各象限,对 RF 制式目标软杀伤减压。
    jammers = [
        HunterMax(f"HM-{a:03d}", _polar(0, 0, a, 3_500, 15), jam_range=4_500.0)
        for a in range(45, 360, 90)
    ]

    targets = []
    for i in range(n):
        az = (i * 360.0 / n)
        rng_m = 8_000.0 + (i % 5) * 600.0
        kind = (TargetKind.LOITERING_MUNITION if i % 4 == 0
                else TargetKind.FIXED_WING_UAV)
        targets.append(_inbound(
            tid=f"SW{i:02d}",
            aim=asset,
            start=_polar(0, 0, az, rng_m, 200.0 + (i % 4) * 150.0),
            cruise=42.0 + (i % 7) * 2.0,
            kind=kind,
            rcs=0.10 + (i % 3) * 0.05,
            emits_rf=(i % 4 != 0),  # 巡飞弹(每 4 个)RF 静默
        ))

    return Scenario(
        asset=asset, targets=targets, spotters=spotters, pads=pads,
        jammers=jammers, seed=seed, max_time=400.0,
    )


def build_decoy_scenario(seed: int = 2026, with_decoys: bool = True) -> Scenario:
    """亚视场诱饵压力想定:量化末段**误关联**。

    四个突击波,每波 1 个高价值真目标(RF 静默巡飞弹,须 Thunder 硬杀伤),
    在 ``with_decoys=True`` 时各伴随 2 个**亚视场间距(~250m)内的强回波诱饵**
    (RCS 远大于真目标)。信杂比加权下,导引头常被诱饵夺锁 → 真目标漏防。
    ``with_decoys=False`` 为对照(同样的真目标、无诱饵),二者之差即诱饵经
    误关联取得的突防增益。无干扰单元(隔离硬杀伤,聚焦误关联)。
    """
    asset = Vec3(0.0, 0.0, 0.0)
    spotters = [SpotterPro("SPT-C", Vec3(0.0, 0.0, 20.0), azimuth_width_deg=360.0)]
    pads = [
        LaunchPad(f"PAD-{a:03d}", _polar(0, 0, a, 3_000, 15), inventory=6)
        for a in (45, 135, 225, 315)
    ]

    targets = []
    for k, az in enumerate((45, 135, 225, 315)):
        start = _polar(0, 0, az, 9_000, 500.0)
        targets.append(Target(
            target_id=f"REAL-{k}", position=start, aim=asset, cruise_speed=50.0,
            kind=TargetKind.LOITERING_MUNITION, rcs=0.15, emits_rf=False,
            terminal_speed=58.0,
        ))
        if with_decoys:
            perp = _polar(0, 0, az + 90.0, 1.0, 0.0)  # 单位横向矢量
            for j, off in enumerate((-250.0, 250.0)):
                targets.append(Target(
                    target_id=f"DECOY-{k}-{j}",
                    position=Vec3(start.x + perp.x * off, start.y + perp.y * off,
                                  start.z),
                    aim=asset, cruise_speed=50.0,
                    kind=TargetKind.FIXED_WING_UAV, rcs=0.5,  # 强回波诱饵
                    emits_rf=False,
                ))

    return Scenario(
        asset=asset, targets=targets, spotters=spotters, pads=pads,
        seed=seed, max_time=400.0,
    )


# 默认演示想定。
build_demo_scenario = build_point_defense_scenario
