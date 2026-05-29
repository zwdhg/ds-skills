# c2sim — 演练指挥控制系统(反无人机 · 纯仿真)

一个**离线、自洽的反无人机(C-UAS)对抗仿真引擎**,用于训练、兵棋推演与
算法验证。系统对齐参考系统(塞防科技智能反无人机防御体系)的体系构成与
规格,复现指挥控制核心处理链路:

```
                                                       ┌─► Thunder 硬杀伤
Spotter Pro 多模态探测 ─► 航迹融合 ─► 威胁研判 ─► 拦截指令生成 ┤
      sensors            fusion      threat     interception └─► Hunter Max 软杀伤
            └───────────────── Skyshield Nexus 指控闭环 ─────────────────┘
```

> ⚠️ **范围声明**:本项目是**软件仿真**。其中目标、传感器与 Thunder 截击机
> **全部是数学模型**,在模拟环境内运行。它**不接入、也无法接入任何真实
> 装备**,**不读取真实世界目标数据**,不产生任何现实世界动作。其定位等同于
> 战术仿真 / 兵棋类软件。参考系统本身是**防御性反无人机系统**,而非火控系统。
> 规格摘要见 [`docs/reference-system.md`](docs/reference-system.md)。

## 体系映射

| 参考系统单元 | 本仿真模块 | 职责 |
| --- | --- | --- |
| **Spotter Pro** | `sensors.py` | 多模态探测:频谱测向 + X 波段 AESA 雷达 + 光电,级联引导 |
| **Skyshield Nexus** | `engine.py` | 指控中枢:航迹融合 + 威胁研判 + 拦截指令生成 + 调度 |
| **Thunder** | `weapons.py` | AI 自主截击机(硬杀伤):起飞抵近 → 目标搜索 → 末段拦截 |
| **发射平台** | `weapons.py` | Thunder 存放/部署/发射,受作业半径约束 |
| **Hunter Max** | `weapons.py` | 无线电干扰设备(软杀伤):对 RF 制式目标致迫降/返航 |

## 快速开始

无第三方依赖,仅需 Python 3.10+。

```bash
# 核心要域点状防护(360° 安全穹顶)
python -m c2sim.cli

# 边境线带状防护(多站 180° 扇区)
python -m c2sim.cli --scenario border

# 打印逐条事件时间线
python -m c2sim.cli --events

# 生成态势 SVG 图(无第三方依赖)
python -m c2sim.cli --plot situation.svg

# 指定随机种子复现
python -m c2sim.cli --seed 7

# 运行测试(58 项)
python -m unittest discover -s tests
```

示例输出:

```
================================================================
  演练指挥控制系统 · Skyshield Nexus 仿真复盘
  部署样式:核心要域点状防护
================================================================
探测站 1 | 发射平台 4 | 雷达覆盖合计 ≈ 78.5 km²
用时 151.0s | 来袭目标 4 | 摧毁 1 | 软杀伤 3 | 突防 0 | 发射 Thunder 5 架
```

> 上例体现软硬结合:3 个 RF 制式无人机由 Hunter Max **软杀伤**(迫降/返航),
> RF 静默的巡飞弹由 Thunder **硬杀伤**——仅消耗 5 架 Thunder 即零突防。

## 处理链路要点

| 环节 | 模块 | 说明 |
| --- | --- | --- |
| 多模态探测 | `sensors.py` | **频谱测向**(≥10km,二维定向,引导雷达)→ **AESA 雷达**(三维位置,RCS 相关作用距离,TAS 容量≥6,近界盲区)→ **光电**(0.15mrad 高精度 + 目标识别,单转台)。雷达**俯仰误差大于方位误差**(高度误差更大)。 |
| 航迹融合 | `fusion.py` | 多源量测逆方差加权融合 → 最近邻波门关联 → α-β 滤波 → 航迹生命周期;光电识别结果随融合并入航迹。 |
| 威胁研判 | `threat.py` | 企图(CPA)+ 紧迫(抵达时间)+ 逼近 + 杀伤(优先采用光电识别类型)四因子合成威胁分与等级。 |
| 拦截指令 | `interception.py` | 火力-目标分配:按威胁排序,结合各发射平台**作业半径/库存**优选最优单元,解算预测拦截点,下发 ENGAGE/HOLD。 |
| Thunder | `weapons.py` | 三阶段:起飞抵近(指令制导)→ 目标搜索(弹载截获)→ 末段拦截(图像寻的,近炸引信)。作业半径约束 + 末段余度。 |
| Hunter Max | `weapons.py` | 软杀伤:对 RF 制式目标在干扰圈内致链路中断,持续达阈值判迫降/返航;RF 静默目标免疫。 |
| 引擎 | `engine.py` | 时间步进、软/硬杀伤决策、真值/毁伤判定、突防判定。 |

### 关键设计

- **真值隔离**:引擎掌握真值(目标真实位置、毁伤判定),指控链路只能看到
  带噪航迹。传感器报告中的 `truth_id` 仅供仿真打分/调试,融合器**不据此
  关联**。
- **多模态融合的工程价值**:雷达俯仰误差导致高度精度差;光电的高精度角度
  量测经融合**收紧高度精度**——对应参考资料"改善传统雷达俯仰测量误差大、
  制约初-中制导交班"的论述。
- **频谱测向作为引导而非定位**:二维测向无距离信息,建模为**为雷达提供早期
  截获线索**;RF 静默(自主)目标(如巡飞弹)只能靠雷达/光电处置,更难。
- **Thunder 末段图像寻的**:末段以弹载传感器锁定真实目标精确寻的,这解释了
  规格中 ≥90% 的单发拦截成功率。
- **软硬结合**:Skyshield Nexus 对 RF 制式目标优先调度 Hunter Max 软杀伤
  (节省 Thunder),对 RF 静默/抗扰目标用 Thunder 硬杀伤——对应参考资料
  "侦测—识别—干扰"一体化防控体系。

## 态势可视化

`python -m c2sim.cli --plot out.svg` 生成俯视 SVG 态势图(纯文本矢量,浏览器
可直接查看,**无第三方依赖**):安全穹顶、Spotter Pro 覆盖扇区、发射平台
作业半径、Hunter Max 干扰圈、目标航迹(按结局着色:红=摧毁/青=软杀伤/
深红=突防)、Thunder 轨迹及事件标记。亦可在代码中调用:

```python
from c2sim.engine import Engine
from c2sim.scenarios import build_point_defense_scenario
from c2sim.viz import render_svg

engine = Engine(build_point_defense_scenario())
engine.run()
render_svg(engine, "situation.svg", title="态势图")
```

## 典型部署

| 想定 | 构成 | 覆盖 |
| --- | --- | --- |
| 核心要域点状防护 | 中心单站 360° + Thunder 四象限前置 | 半径 5km 安全穹顶,≈78.5 km² |
| 边境线带状防护 | 多站 180° 扇区沿线 + 平台后置 | 单组 ≈39.27 km² |

## 自定义想定

```python
from c2sim.engine import Engine, Scenario
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.sensors import SpotterPro
from c2sim.weapons import LaunchPad

scenario = Scenario(
    asset=Vec3(0, 0, 0),
    targets=[
        Target("T1", Vec3(7_000, 0, 500), aim=Vec3(0, 0, 0),
               cruise_speed=48.0, kind=TargetKind.FIXED_WING_UAV, rcs=0.15),
    ],
    spotters=[SpotterPro("SPT", Vec3(0, 0, 20))],
    pads=[LaunchPad("PAD", Vec3(2_500, 0, 15))],
)
print(Engine(scenario).run().summary())
```

可调参数:`ThreatPolicy`(研判权重/阈值)、`EngagementPolicy`(交战等级/齐射)、
`SpotterPro`(各模块精度/容量/扇区)、`LaunchPad`(作业半径/速度/库存)、
`Target`(类型/RCS/末段加速/是否辐射 RF)。

## 项目结构

```
c2sim/
  geometry.py      三维矢量、CPA、定速拦截解算、LOS 角误差噪声
  models.py        数据模型(目标/报告/航迹/研判/指令)
  sensors.py       Spotter Pro 多模态探测
  fusion.py        航迹融合
  threat.py        威胁研判
  interception.py  拦截指令生成(火力-目标分配,软/硬杀伤决策)
  weapons.py       Thunder 截击机、发射平台、Hunter Max 干扰设备
  engine.py        Skyshield Nexus 仿真引擎
  viz.py           态势 SVG 可视化(无依赖)
  scenarios.py     点状/带状两种部署想定
  cli.py           命令行入口
docs/
  reference-system.md  参考系统规格摘要
tests/             unittest 测试(58 项)
```
