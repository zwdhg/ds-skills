"""仿真域内的核心数据模型(反无人机 C-UAS 规格)。

这些类型在处理链路各环节之间传递:Spotter Pro 各模块产出
:class:`SensorReport`,航迹融合维护 :class:`Track`,Skyshield Nexus 研判
输出 :class:`ThreatAssessment`、拦截环节生成 :class:`Command`。
:class:`Target` 是引擎掌握的"真值",对指控链路不可见——链路只能看到
带噪航迹。
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
    """来袭目标类型(低空小目标),影响默认杀伤性权重。"""

    MICRO_UAV = "micro_uav"            # 微型多旋翼/穿越机
    ROTARY_UAV = "rotary_uav"          # 旋翼无人机
    FIXED_WING_UAV = "fixed_wing_uav"  # 固定翼无人机
    LOITERING_MUNITION = "loitering_munition"  # 巡飞弹(如 Shahed-136)

    @property
    def lethality(self) -> float:
        """[0,1] 区间的相对杀伤性,用于威胁研判。"""
        return {
            TargetKind.MICRO_UAV: 0.25,
            TargetKind.ROTARY_UAV: 0.45,
            TargetKind.FIXED_WING_UAV: 0.7,
            TargetKind.LOITERING_MUNITION: 1.0,
        }[self]


class SensorModality(enum.Enum):
    """Spotter Pro 的探测模态。"""

    RF = "rf"        # 频谱测向(二维定向,远程预警/引导)
    RADAR = "radar"  # X 波段 AESA(三维位置)
    EO = "eo"        # 光电(高精度角度 + 识别)


# ---------------------------------------------------------------------------
# 真值(引擎私有)
# ---------------------------------------------------------------------------


@dataclass
class Target:
    """来袭无人机目标的"真值"状态,仅引擎与传感器可见。

    目标朝攻击瞄准点 ``aim``(被掩护要地)自主寻的;进入 ``terminal_range``
    后切换到 ``terminal_speed`` 末段加速突击(对应"中段巡飞、末段加速")。
    """

    target_id: str
    position: Vec3
    aim: Vec3
    cruise_speed: float
    kind: TargetKind = TargetKind.FIXED_WING_UAV
    rcs: float = 0.1                    # 雷达散射截面(平方米)
    terminal_speed: float | None = None  # 末段突击速度(None 表示不加速)
    terminal_range: float = 1500.0      # 切入末段的距要地距离(米)
    emits_rf: bool = True               # 是否辐射可被频谱测向截获的信号
    alive: bool = True

    def current_speed(self) -> float:
        if (
            self.terminal_speed is not None
            and self.position.distance_to(self.aim) <= self.terminal_range
        ):
            return self.terminal_speed
        return self.cruise_speed

    @property
    def velocity(self) -> Vec3:
        return (self.aim - self.position).unit() * self.current_speed()

    def advance(self, dt: float) -> None:
        """朝瞄准点寻的推进 ``dt`` 秒。"""
        self.position = self.position + self.velocity * dt


# ---------------------------------------------------------------------------
# 探测与航迹
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorReport:
    """Spotter Pro 某模块在某时刻对某目标的一次量测。

    报告**不含**目标真实身份(``truth_id`` 仅供仿真打分/调试)。
    ``classification`` 仅由光电模块在完成识别后填写。
    """

    sensor_id: str
    modality: SensorModality
    timestamp: float
    position: Vec3          # 量测位置(RF 模态为粗略定向折算点)
    position_sigma: float   # 等效一倍标准差(米),用于融合加权与波门
    classification: TargetKind | None = None  # 光电识别结果
    truth_id: str | None = None


@dataclass
class Track:
    """由多模态量测融合得到的、持续维护的目标航迹。"""

    track_id: str
    position: Vec3
    velocity: Vec3
    last_update: float
    contributing_sensors: set[str] = field(default_factory=set)
    modalities: set[SensorModality] = field(default_factory=set)
    classification: TargetKind | None = None  # 光电确认的类型
    hits: int = 0
    coast_time: float = 0.0

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
    score: float
    level: ThreatLevel
    time_to_impact: float | None
    closest_approach: float
    rationale: str


# ---------------------------------------------------------------------------
# 拦截指令
# ---------------------------------------------------------------------------


class CommandKind(enum.Enum):
    ENGAGE = "engage"  # 交战:调度发射 Thunder
    HOLD = "hold"      # 暂不交战(无可用拦截资源/超出作业半径等)


@dataclass
class Command:
    """Skyshield Nexus 下发给某发射平台的拦截指令。"""

    command_id: str
    kind: CommandKind
    timestamp: float
    track_id: str | None
    pad_id: str | None             # 受令发射平台
    intercept_point: Vec3 | None   # 预测拦截点
    intercept_time: float | None   # 预计命中时刻(绝对仿真时间)
    note: str = ""


_counter = itertools.count(1)


def next_id(prefix: str) -> str:
    """生成带前缀的单调递增标识,便于日志阅读。"""
    return f"{prefix}-{next(_counter):04d}"
