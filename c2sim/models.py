"""仿真域内的核心数据模型。

这些类型在处理链路各环节之间传递:传感器产出 :class:`SensorReport`,
融合器维护 :class:`Track`,研判器输出 :class:`ThreatAssessment`,
拦截环节生成 :class:`Command`。:class:`Target` 是引擎掌握的"真值",
对指控链路不可见——链路只能看到带噪的航迹。
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

from c2sim.geometry import Vec3


class ThreatLevel(enum.IntEnum):
    """威胁等级,数值越大越紧急。"""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return {
            ThreatLevel.NONE: "无",
            ThreatLevel.LOW: "低",
            ThreatLevel.MEDIUM: "中",
            ThreatLevel.HIGH: "高",
            ThreatLevel.CRITICAL: "紧急",
        }[self]


class TargetKind(enum.Enum):
    """来袭目标类型,影响默认杀伤性权重。"""

    AIRCRAFT = "aircraft"
    CRUISE_MISSILE = "cruise_missile"
    DRONE = "drone"
    BALLISTIC = "ballistic"

    @property
    def lethality(self) -> float:
        """[0,1] 区间的相对杀伤性,用于威胁研判。"""
        return {
            TargetKind.DRONE: 0.3,
            TargetKind.AIRCRAFT: 0.6,
            TargetKind.CRUISE_MISSILE: 0.85,
            TargetKind.BALLISTIC: 1.0,
        }[self]


# ---------------------------------------------------------------------------
# 真值(引擎私有)
# ---------------------------------------------------------------------------


@dataclass
class Target:
    """来袭目标的"真值"状态,仅引擎与传感器可见。"""

    target_id: str
    position: Vec3
    velocity: Vec3
    kind: TargetKind = TargetKind.AIRCRAFT
    rcs: float = 1.0  # 雷达散射截面(平方米),影响探测概率
    alive: bool = True

    def advance(self, dt: float) -> None:
        """匀速推进 ``dt`` 秒。"""
        self.position = self.position + self.velocity * dt


# ---------------------------------------------------------------------------
# 探测与航迹
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorReport:
    """单部传感器在某一时刻对某一目标的一次带噪测量。

    报告**不含**目标真实身份——融合器须自行完成数据关联。
    """

    sensor_id: str
    timestamp: float
    position: Vec3  # 带噪的量测位置
    position_sigma: float  # 量测一倍标准差(米),反映精度
    # 真值目标编号,仅供仿真打分/调试,链路不得据此关联。
    truth_id: str | None = None


@dataclass
class Track:
    """由一个或多个传感器报告融合得到的、持续维护的目标航迹。"""

    track_id: str
    position: Vec3
    velocity: Vec3
    last_update: float
    # 该航迹本帧汇聚的传感器编号集合。
    contributing_sensors: set[str] = field(default_factory=set)
    hits: int = 0  # 累计关联到的报告数,衡量航迹质量/置信度
    coast_time: float = 0.0  # 距上次成功关联的时长(秒)

    @property
    def confidence(self) -> float:
        """[0,1] 航迹置信度:命中越多、惯性外推越短,置信越高。"""
        maturity = min(self.hits / 5.0, 1.0)
        freshness = max(0.0, 1.0 - self.coast_time / 6.0)
        return round(maturity * freshness, 3)


@dataclass
class ThreatAssessment:
    """对单条航迹的威胁研判结果。"""

    track_id: str
    score: float  # 归一化威胁分 [0,1]
    level: ThreatLevel
    time_to_impact: float | None  # 预计抵达被掩护要地的时间(秒),None 为不来袭
    closest_approach: float  # 对要地的最近接近距离(米)
    rationale: str  # 人类可读的研判依据


# ---------------------------------------------------------------------------
# 拦截指令
# ---------------------------------------------------------------------------


class CommandKind(enum.Enum):
    ENGAGE = "engage"  # 交战:发射拦截弹
    HOLD = "hold"      # 暂不交战(无可用拦截资源等)


@dataclass
class Command:
    """指控链路下发给某 Thunder 发射单元的拦截指令。"""

    command_id: str
    kind: CommandKind
    timestamp: float
    track_id: str | None  # 交战目标航迹
    battery_id: str | None  # 受令发射单元
    intercept_point: Vec3 | None  # 预测拦截点
    intercept_time: float | None  # 预计命中时刻(绝对仿真时间)
    note: str = ""


_counter = itertools.count(1)


def next_id(prefix: str) -> str:
    """生成带前缀的单调递增标识,便于日志阅读。"""
    return f"{prefix}-{next(_counter):04d}"
