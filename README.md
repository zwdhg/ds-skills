# c2sim — 演练指挥控制系统(纯仿真)

一个**离线、自洽的防空对抗仿真引擎**,用于训练、兵棋推演与算法验证。
系统完整复现指挥控制的核心处理链路:

```
传感器探测 ──► 航迹融合 ──► 威胁研判 ──► 拦截指令生成 ──► Thunder 调度毁伤
  sensors      fusion       threat      interception        weapons
```

> ⚠️ **范围声明**:本项目是**软件仿真**。其中的目标、传感器与 Thunder
> 拦截弹**全部是数学模型**,在模拟环境内运行。它**不接入、也无法接入任何
> 真实装备**,**不读取真实世界的目标数据**,不产生任何现实世界动作。其定位
> 等同于战术仿真 / 兵棋类软件,服务于训练与算法研究。

## 快速开始

无第三方依赖,仅需 Python 3.10+。

```bash
# 运行内置演示想定
python -m c2sim.cli

# 打印逐条事件时间线
python -m c2sim.cli --events

# 指定随机种子复现
python -m c2sim.cli --seed 7

# 运行测试
python -m unittest discover -s tests
```

示例输出:

```
============================================================
  演练指挥控制系统 · 仿真复盘
============================================================
用时 190.5s | 来袭目标 4 | 摧毁 3 | 突防 1 | 发射 Thunder 11 发
------------------------------------------------------------
已摧毁: T3-DRONE, T1-AIRCRAFT, T2-CRUISE
突防(未拦截): T4-BALLISTIC
指令总数: 45
============================================================
```

## 处理链路

| 环节 | 模块 | 职责 |
| --- | --- | --- |
| 传感器 | `c2sim/sensors.py` | 雷达按距离/RCS 对目标产生**带噪**量测;探测概率随作用距离衰减 |
| 航迹融合 | `c2sim/fusion.py` | 多源量测逆方差加权融合 → 最近邻波门关联 → α-β 滤波估计位置/速度 → 航迹生命周期管理 |
| 威胁研判 | `c2sim/threat.py` | 由企图(CPA)、紧迫(抵达时间)、逼近(距离)、杀伤(运动学反推类型)四因子合成威胁分与等级 |
| 拦截指令 | `c2sim/interception.py` | 火力-目标分配:按威胁排序贪心分配拦截弹,解算预测拦截点,生成 ENGAGE / HOLD 指令 |
| Thunder | `c2sim/weapons.py` | 发射单元(有限库存/射界)与在飞拦截弹;中段指令制导 + 末段导引头寻的 |
| 引擎 | `c2sim/engine.py` | 按时间步推进全链路,掌握真值并判定毁伤、突防 |

### 关键设计:真值隔离

引擎掌握**真值**(目标真实位置、毁伤判定),而指控链路只能看到**带噪航迹**。
传感器报告中的 `truth_id` 仅用于仿真打分/调试,融合器**不得**据此关联——
数据关联必须靠运动学完成。这一信息隔离是仿真可信度的核心。

### 关键设计:两段制导

- **中段**:拦截弹跟随指控链路上行的航迹做指令制导,持续重解拦截点;
- **末段**:进入导引头截获距离后,弹上导引头锁定真实目标(带导引头噪声)
  精确寻的。

这解释了为何空气动力目标能被可靠拦截,而 1800 m/s 的弹道目标在 1200 m/s
拦截弹面前常常突防——这是**真实的能力边界**,而非缺陷:拦截高速弹道目标
需要更高速的拦截弹。

## 自定义想定

```python
from c2sim.engine import Engine, Scenario
from c2sim.geometry import Vec3
from c2sim.models import Target, TargetKind
from c2sim.sensors import Radar
from c2sim.weapons import ThunderBattery

scenario = Scenario(
    asset=Vec3(0, 0, 0),
    targets=[
        Target("T1", Vec3(40_000, 0, 6000), Vec3(-250, 0, 0),
               kind=TargetKind.AIRCRAFT, rcs=5.0),
    ],
    radars=[Radar("R", Vec3(0, 5000, 30), max_range=120_000)],
    batteries=[ThunderBattery("B", Vec3(8000, 0, 20), interceptor_speed=1200)],
)
result = Engine(scenario).run()
print(result.summary())
```

可调参数集中在 `ThreatPolicy`(研判权重/阈值)、`EngagementPolicy`
(交战等级/齐射弹数)与各模型字段(雷达精度、拦截弹速度/杀伤半径等)。

## 项目结构

```
c2sim/
  geometry.py      三维矢量、CPA、定速拦截解算
  models.py        数据模型(目标/报告/航迹/研判/指令)
  sensors.py       雷达量测模型
  fusion.py        航迹融合
  threat.py        威胁研判
  interception.py  拦截指令生成(火力-目标分配)
  weapons.py       Thunder 发射单元与拦截弹
  engine.py        仿真引擎(时间步进 + 真值/毁伤判定)
  scenarios.py     预置演示想定
  cli.py           命令行入口
tests/             unittest 测试(35 项)
```
