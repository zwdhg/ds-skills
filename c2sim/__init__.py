"""c2sim —— 演练指挥控制系统(纯仿真)。

本包实现一个**离线、自洽的防空对抗仿真引擎**,用于训练、兵棋推演与
算法验证。其中所有目标、传感器与 Thunder 拦截弹均为软件模型,
**不接入任何真实装备、不读取真实世界目标数据**。

核心处理链路:

    传感器探测 ──► 航迹融合 ──► 威胁研判 ──► 拦截指令生成 ──► Thunder 调度
       sensors      fusion       threat      interception        weapons

顶层入口见 :mod:`c2sim.engine` 与 :mod:`c2sim.cli`。
"""

from c2sim.geometry import Vec3
from c2sim.models import (
    Command,
    SensorModality,
    SensorReport,
    Target,
    TargetKind,
    ThreatAssessment,
    ThreatLevel,
    Track,
)

__all__ = [
    "Vec3",
    "SensorReport",
    "SensorModality",
    "Track",
    "Target",
    "TargetKind",
    "ThreatAssessment",
    "ThreatLevel",
    "Command",
]

__version__ = "0.1.0"
