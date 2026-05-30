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
    emits_rf: bool = True               # 是否辐射可被频谱测向截获/可被干扰的信号
    jammed: bool = False                # 是否处于被干扰(链路中断)状态
    jam_elapsed: float = 0.0            # 持续被干扰时长(秒)
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
        # 被干扰(控制/导航链路中断)→ 失去寻的能力,原地悬停/失速。
        if self.jammed:
            return Vec3()
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
    cov: Vec3 | None = None  # 各轴方差(var_x,var_y,var_z);供协方差跟踪器各向异性融合
    classification: TargetKind | None = None  # 光电识别结果
    rf_emitter: bool = False  # 该目标本帧被频谱测向截获(辐射 RF,可被干扰)
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
    rf_emitter: bool = False  # 是否为 RF 辐射源(可实施干扰软杀伤)
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
    ENGAGE = "engage"  # 硬杀伤:调度发射 Thunder
    JAM = "jam"        # 软杀伤:调度 Hunter Max 实施无线电干扰
    HOLD = "hold"      # 暂不交战(无可用拦截资源/超出作业半径等)


@dataclass
class Command:
    """Skyshield Nexus 下发给发射平台 / Hunter Max 的拦截指令。"""

    command_id: str
    kind: CommandKind
    timestamp: float
    track_id: str | None
    pad_id: str | None = None      # 受令发射平台(ENGAGE)
    jammer_id: str | None = None   # 受令干扰单元(JAM)
    intercept_point: Vec3 | None = None
    intercept_time: float | None = None
    note: str = ""


class IdGenerator:
    """带前缀的单调递增 ID 生成器。

    每个引擎应持有独立实例,避免跨引擎/跨蒙特卡洛运行共享全局计数器——
    使 ID 可按次复现、无界增长受控。
    """

    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def next(self, prefix: str) -> str:
        return f"{prefix}-{next(self._counter):04d}"


# 模块级默认生成器,仅为向后兼容(直接调用 next_id 的旧路径/测试);
# 引擎运行路径应注入独立的 :class:`IdGenerator`。
_default_ids = IdGenerator()


def next_id(prefix: str) -> str:
    """用模块级默认生成器生成 ID(向后兼容入口)。"""
    return _default_ids.next(prefix)
