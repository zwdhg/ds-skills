"""航迹融合。

两级融合:

1. **多源量测融合**——把不同传感器对同一目标的并发报告按邻近度聚类,
   做逆方差加权,得到单个精度更高的"融合量测"。
2. **量测-航迹关联**——将融合量测以最近邻 + 波门(gating)方式关联到
   已有航迹,用 α-β 滤波更新位置与速度;未关联的量测起始新航迹,
   未被更新的航迹转入惯性外推(coast),超时则撤销。

实现刻意保持轻量(无外部依赖),但覆盖了真实航迹处理的关键环节:
数据关联、状态估计、航迹生命周期管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from c2sim.geometry import Vec3
from c2sim.models import SensorReport, Track, next_id


@dataclass
class _FusedMeasurement:
    """一组邻近传感器报告逆方差加权后的融合量测。"""

    position: Vec3
    sigma: float
    sensors: set[str]
    modalities: set = field(default_factory=set)
    classification: object | None = None
    rf_emitter: bool = False


def fuse_reports(
    reports: list[SensorReport], cluster_distance: float = 1500.0
) -> list[_FusedMeasurement]:
    """将相互邻近的传感器报告聚类并融合为高精度量测。

    采用简单的单趟贪心聚类:报告若落在某簇质心 ``cluster_distance``
    米内即并入该簇。对防空场景下稀疏目标已足够;簇内按 1/σ² 加权融合。
    """
    clusters: list[list[SensorReport]] = []
    for rep in reports:
        for cluster in clusters:
            centroid = _centroid(cluster)
            if centroid.distance_to(rep.position) <= cluster_distance:
                cluster.append(rep)
                break
        else:
            clusters.append([rep])
    return [_inverse_variance_fuse(c) for c in clusters]


def _centroid(reports: list[SensorReport]) -> Vec3:
    acc = Vec3()
    for r in reports:
        acc = acc + r.position
    return acc / len(reports)


def _inverse_variance_fuse(reports: list[SensorReport]) -> _FusedMeasurement:
    """逆方差(1/σ²)加权融合一簇报告。"""
    wsum = 0.0
    acc = Vec3()
    for r in reports:
        w = 1.0 / max(r.position_sigma, 1e-3) ** 2
        acc = acc + r.position * w
        wsum += w
    fused_pos = acc / wsum
    # 融合后等效标准差:独立量测合成精度提升。
    fused_sigma = (1.0 / wsum) ** 0.5
    # 光电识别结果(若有)并入融合量测。
    classification = next(
        (r.classification for r in reports if r.classification is not None), None
    )
    return _FusedMeasurement(
        position=fused_pos,
        sigma=fused_sigma,
        sensors={r.sensor_id for r in reports},
        modalities={r.modality for r in reports},
        classification=classification,
        rf_emitter=any(r.rf_emitter for r in reports),
    )


class TrackFusion:
    """跨帧维护航迹集合的航迹融合器。

    参数:
        gate_distance: 量测-航迹关联波门(米)。
        alpha, beta: α-β 滤波增益,分别作用于位置与速度修正。
        max_coast: 惯性外推容忍时长(秒),超过则撤销航迹。
        ids: 航迹 ID 生成器(引擎注入以保证按次复现);None 时用模块默认。
    """

    def __init__(
        self,
        gate_distance: float = 3000.0,
        alpha: float = 0.6,
        beta: float = 0.2,
        max_coast: float = 8.0,
        confirm_threshold: int = 1,
        ids=None,
    ) -> None:
        self.gate_distance = gate_distance
        self.alpha = alpha
        self.beta = beta
        self.max_coast = max_coast
        self.confirm_threshold = confirm_threshold
        self._mint = ids.next if ids is not None else next_id
        self.tracks: dict[str, Track] = {}

    def update(self, reports: list[SensorReport], now: float) -> list[Track]:
        """处理本帧报告,返回更新后仍存活的航迹列表。"""
        measurements = fuse_reports(reports)

        # 1) 预测各航迹到当前时刻,便于最近邻关联。
        predicted: dict[str, Vec3] = {}
        for tid, trk in self.tracks.items():
            dt = now - trk.last_update
            predicted[tid] = trk.position + trk.velocity * dt

        # 2) 全局最近邻关联(贪心,按距离从小到大)。
        pairs: list[tuple[float, int, str]] = []
        for mi, m in enumerate(measurements):
            for tid, pred_pos in predicted.items():
                d = pred_pos.distance_to(m.position)
                if d <= self.gate_distance:
                    pairs.append((d, mi, tid))
        pairs.sort(key=lambda p: p[0])

        used_meas: set[int] = set()
        used_track: set[str] = set()
        for _d, mi, tid in pairs:
            if mi in used_meas or tid in used_track:
                continue
            self._update_track(self.tracks[tid], measurements[mi], now)
            used_meas.add(mi)
            used_track.add(tid)

        # 3) 未关联的量测 → 起始新航迹。
        for mi, m in enumerate(measurements):
            if mi not in used_meas:
                self._spawn_track(m, now)

        # 4) 未更新的航迹 → 惯性外推,超时撤销。
        for tid, trk in list(self.tracks.items()):
            if tid in used_track:
                continue
            trk.coast_time = now - trk.last_update
            trk.contributing_sensors = set()
            if trk.coast_time > self.max_coast:
                del self.tracks[tid]

        return list(self.tracks.values())

    def _update_track(self, trk: Track, m: _FusedMeasurement, now: float) -> None:
        """以 α-β 滤波用融合量测更新航迹状态。"""
        dt = now - trk.last_update
        if dt <= 0.0:
            dt = 1e-3
        pred = trk.position + trk.velocity * dt
        residual = m.position - pred
        trk.position = pred + residual * self.alpha
        trk.velocity = trk.velocity + residual * (self.beta / dt)
        trk.last_update = now
        trk.coast_time = 0.0
        trk.hits += 1
        if trk.hits >= self.confirm_threshold:
            trk.confirmed = True  # 确认后latch,不再回退
        trk.contributing_sensors = set(m.sensors)
        trk.modalities = set(m.modalities)
        if m.classification is not None:
            trk.classification = m.classification
        if m.rf_emitter:
            trk.rf_emitter = True

    def _spawn_track(self, m: _FusedMeasurement, now: float) -> None:
        """由一个无主量测起始新航迹(初始速度未知,置零)。"""
        tid = self._mint("TRK")
        self.tracks[tid] = Track(
            track_id=tid,
            position=m.position,
            velocity=Vec3(),
            last_update=now,
            contributing_sensors=set(m.sensors),
            modalities=set(m.modalities),
            classification=m.classification,
            rf_emitter=m.rf_emitter,
            confirmed=self.confirm_threshold <= 1,
            hits=1,
            coast_time=0.0,
        )
