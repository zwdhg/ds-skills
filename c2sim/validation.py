"""外部真值交叉验证框架(骨架)。

`c2sim` 当前**最大短板是无独立验证基准(oracle)**:一切数值仅在模型假设内
自洽(见 docs/model-card.md)。本模块提供一个**留好数据接口**的验证骨架——
一旦有**外部真值轨迹**(真实雷达/光电记录、或更高保真仿真的输出),即可:

    外部真值轨迹 ──► 本仓传感器模型产生量测 ──► 待验跟踪器估计 ──► 与真值对比

并输出跟踪误差指标(RMSE、最大误差、各轴 RMSE),作为**对估计算法的独立
检验**。本仓的 c2sim 引擎不参与——验证的是"我们的传感+跟踪能否复现外部真值"。

⚠️ **诚实约束(写入 AGENTS.md 硬规则)**:
- 真正的 oracle 必须是**外部、独立**数据。本模块自带的 :func:`synthetic_track`
  仅用于**自检管线本身是否跑通**,**严禁**用合成轨迹"验证"模型——那又是自证
  (用同一套运动学既造真值又评估),正是我们要避免的循环。
- 接入真实数据前,本框架只声明**接口与指标**,不产出任何可信结论。

数据契约见 :class:`TruthTrack`;接入示例见 :func:`load_truth_csv`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from c2sim.geometry import Vec3


# ---------------------------------------------------------------------------
# 数据契约:外部真值轨迹的最小 schema
# ---------------------------------------------------------------------------


@dataclass
class TruthSample:
    """某时刻一个目标的真值状态。"""

    t: float          # 时间(秒)
    target_id: str
    position: Vec3    # 真值位置(米)


@dataclass
class TruthTrack:
    """一条外部真值轨迹(时间有序的真值采样序列)。

    这是外部数据接入的**唯一契约**:任何来源(真实记录/高保真仿真)只要能产出
    ``[(t, target_id, x, y, z), ...]``,即可构造本对象并送入 :class:`Validator`。
    """

    samples: list[TruthSample] = field(default_factory=list)

    def add(self, t: float, target_id: str, x: float, y: float, z: float) -> None:
        self.samples.append(TruthSample(t, target_id, Vec3(x, y, z)))

    def times(self) -> list[float]:
        return sorted({s.t for s in self.samples})

    def at(self, t: float) -> dict[str, Vec3]:
        """时刻 ``t`` 各目标的真值位置(精确匹配时刻)。"""
        return {s.target_id: s.position for s in self.samples if s.t == t}


# ---------------------------------------------------------------------------
# 验证指标
# ---------------------------------------------------------------------------


@dataclass
class TrackingError:
    """跟踪估计相对真值的误差指标。"""

    n: int
    rmse: float          # 三维位置 RMSE(米)
    max_error: float     # 最大瞬时三维误差(米)
    rmse_horizontal: float
    rmse_vertical: float  # 高度(z)RMSE——多模态融合价值的关键观测量
    matched_fraction: float  # 真值样本被成功关联到航迹的比例

    def summary(self) -> str:
        return (
            f"样本 {self.n} | RMSE {self.rmse:.1f}m "
            f"(水平 {self.rmse_horizontal:.1f} / 高度 {self.rmse_vertical:.1f}) | "
            f"最大 {self.max_error:.1f}m | 关联率 {self.matched_fraction*100:.0f}%"
        )


# ---------------------------------------------------------------------------
# 验证器
# ---------------------------------------------------------------------------


class Validator:
    """用本仓传感器模型 + 待验跟踪器复跑外部真值轨迹,评估跟踪误差。

    参数:
        spotter: 传感器模型(:class:`c2sim.sensors.SpotterPro` 或符合 SensorModel)。
        tracker_factory: 无参工厂,返回待验 :class:`c2sim.strategies.Tracker` 实例。
        match_gate: 真值-航迹关联波门(米),用最近邻把真值匹配到航迹。
        seed: 量测噪声随机种子(验证可复现)。
    """

    def __init__(self, spotter, tracker_factory, match_gate: float = 300.0,
                 seed: int = 0) -> None:
        self.spotter = spotter
        self.tracker_factory = tracker_factory
        self.match_gate = match_gate
        self.seed = seed

    def run(self, truth: TruthTrack) -> TrackingError:
        import random

        from c2sim.models import Target, TargetKind

        rng = random.Random(self.seed)
        tracker = self.tracker_factory()

        sq_sum = h_sq = v_sq = 0.0
        max_err = 0.0
        n = 0
        matched = 0

        for t in truth.times():
            truth_now = truth.at(t)
            # 用真值位置构造瞬时目标,交给传感器模型产生本帧带噪量测。
            targets = [
                Target(tid, position=pos, aim=pos, cruise_speed=0.0,
                       kind=TargetKind.FIXED_WING_UAV)
                for tid, pos in truth_now.items()
            ]
            reports = self.spotter.observe(targets, t, rng)
            tracks = tracker.update(reports, t)

            # 每个真值目标用最近邻关联到航迹,累计误差。
            for pos in truth_now.values():
                n += 1
                best = None
                best_d = self.match_gate
                for trk in tracks:
                    d = pos.distance_to(trk.position)
                    if d < best_d:
                        best, best_d = trk, d
                if best is None:
                    continue
                matched += 1
                dx = best.position.x - pos.x
                dy = best.position.y - pos.y
                dz = best.position.z - pos.z
                sq = dx * dx + dy * dy + dz * dz
                sq_sum += sq
                h_sq += dx * dx + dy * dy
                v_sq += dz * dz
                max_err = max(max_err, math.sqrt(sq))

        m = max(matched, 1)
        return TrackingError(
            n=n,
            rmse=math.sqrt(sq_sum / m),
            max_error=max_err,
            rmse_horizontal=math.sqrt(h_sq / m),
            rmse_vertical=math.sqrt(v_sq / m),
            matched_fraction=matched / max(n, 1),
        )


# ---------------------------------------------------------------------------
# 数据接入示例(CSV)与管线自检用合成轨迹
# ---------------------------------------------------------------------------


def load_truth_csv(path: str) -> TruthTrack:
    """从 CSV 加载外部真值轨迹。

    期望表头:``t,target_id,x,y,z``(米/秒)。这是接入真实记录的样板入口——
    替换为任何能产出该五元组的来源即可。
    """
    import csv

    track = TruthTrack()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            track.add(float(row["t"]), row["target_id"],
                      float(row["x"]), float(row["y"]), float(row["z"]))
    return track


def synthetic_track(target_id: str = "GT", n: int = 60, dt: float = 0.5,
                    start: Vec3 | None = None, vel: Vec3 | None = None) -> TruthTrack:
    """生成一条匀速合成真值轨迹。

    ⚠️ **仅用于自检验证管线本身**(传感器→跟踪器→误差统计是否跑通),
    **不得用作 oracle**:用同一套运动学既造真值又评估即自证,无验证意义。
    """
    # 默认起点置于探测作用距离内(便于自检管线);仅自检用,非 oracle。
    start = start or Vec3(6000.0, 2000.0, 800.0)
    vel = vel or Vec3(-50.0, 0.0, 0.0)
    track = TruthTrack()
    for i in range(n):
        t = i * dt
        p = start + vel * t
        track.add(t, target_id, p.x, p.y, p.z)
    return track
