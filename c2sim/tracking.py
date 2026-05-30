"""协方差航迹跟踪器(:class:`c2sim.strategies.Tracker` 的可注入替代实现)。

相对默认的标量 α-β 融合(:class:`c2sim.fusion.TrackFusion`),本跟踪器维护
**各轴位置方差**并做**逐轴卡尔曼更新**,从而真正利用量测的各向异性误差:
雷达俯仰(高度 z)噪声大 → 对 z 的卡尔曼增益小、少更新;光电角精度高 →
对相应轴增益大、强收紧。以此兑现"多模态融合改善高度精度"的工程价值
(默认标量融合无法区分轴向)。

经 ``Engine(scenario, tracker=CovarianceTracker())`` 注入启用;默认不改变行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from c2sim.geometry import Vec3
from c2sim.models import SensorModality, SensorReport, Track


@dataclass
class _CovMeas:
    """一簇邻近报告按各轴逆方差融合后的量测(带各轴方差)。"""

    position: Vec3
    var: Vec3                # 各轴方差
    sensors: set
    modalities: set = field(default_factory=set)
    classification: object | None = None
    rf_emitter: bool = False


def _report_var(r: SensorReport) -> Vec3:
    if r.cov is not None:
        return r.cov
    s2 = max(r.position_sigma, 1e-3) ** 2
    return Vec3(s2, s2, s2)


def _fuse_cov(reports: list[SensorReport], cluster_distance: float) -> list[_CovMeas]:
    """邻近报告聚类 + 各轴逆方差融合(各轴独立)。"""
    clusters: list[list[SensorReport]] = []
    for rep in reports:
        for cluster in clusters:
            c = cluster[0].position
            if c.distance_to(rep.position) <= cluster_distance:
                cluster.append(rep)
                break
        else:
            clusters.append([rep])

    fused: list[_CovMeas] = []
    for cluster in clusters:
        wx = wy = wz = 0.0
        ax = ay = az = 0.0
        for r in cluster:
            v = _report_var(r)
            wxi, wyi, wzi = 1.0 / max(v.x, 1e-6), 1.0 / max(v.y, 1e-6), 1.0 / max(v.z, 1e-6)
            wx += wxi; wy += wyi; wz += wzi
            ax += r.position.x * wxi; ay += r.position.y * wyi; az += r.position.z * wzi
        pos = Vec3(ax / wx, ay / wy, az / wz)
        var = Vec3(1.0 / wx, 1.0 / wy, 1.0 / wz)
        cls = next((r.classification for r in cluster if r.classification is not None),
                   None)
        fused.append(_CovMeas(
            position=pos, var=var,
            sensors={r.sensor_id for r in cluster},
            modalities={r.modality for r in cluster},
            classification=cls,
            rf_emitter=any(r.rf_emitter for r in cluster),
        ))
    return fused


class CovarianceTracker:
    """逐轴卡尔曼航迹跟踪器(各向异性误差感知)。"""

    def __init__(
        self,
        gate_distance: float = 600.0,
        cluster_distance: float = 1500.0,
        beta: float = 0.3,
        process_var: float = 400.0,   # 每秒各轴过程噪声方差(机动不确定性)
        max_coast: float = 6.0,
        confirm_threshold: int = 1,
        ids=None,
    ) -> None:
        self.gate_distance = gate_distance
        self.cluster_distance = cluster_distance
        self.beta = beta
        self.process_var = process_var
        self.max_coast = max_coast
        self.confirm_threshold = confirm_threshold
        from c2sim.models import next_id
        self._mint = ids.next if ids is not None else next_id
        self.tracks: dict[str, Track] = {}
        self._var: dict[str, Vec3] = {}  # 各航迹的各轴位置方差

    def update(self, reports: list[SensorReport], now: float) -> list[Track]:
        meas = _fuse_cov(reports, self.cluster_distance)

        predicted: dict[str, Vec3] = {}
        for tid, trk in self.tracks.items():
            dt = now - trk.last_update
            predicted[tid] = trk.position + trk.velocity * dt

        pairs: list[tuple[float, int, str]] = []
        for mi, m in enumerate(meas):
            for tid, pp in predicted.items():
                d = pp.distance_to(m.position)
                if d <= self.gate_distance:
                    pairs.append((d, mi, tid))
        pairs.sort(key=lambda p: p[0])

        used_m: set[int] = set()
        used_t: set[str] = set()
        for _d, mi, tid in pairs:
            if mi in used_m or tid in used_t:
                continue
            self._kalman_update(self.tracks[tid], meas[mi], now)
            used_m.add(mi); used_t.add(tid)

        for mi, m in enumerate(meas):
            if mi not in used_m:
                self._spawn(m, now)

        for tid, trk in list(self.tracks.items()):
            if tid in used_t:
                continue
            trk.coast_time = now - trk.last_update
            trk.contributing_sensors = set()
            if trk.coast_time > self.max_coast:
                del self.tracks[tid]
                self._var.pop(tid, None)

        return list(self.tracks.values())

    def _kalman_update(self, trk: Track, m: _CovMeas, now: float) -> None:
        dt = now - trk.last_update
        if dt <= 0.0:
            dt = 1e-3
        pred = trk.position + trk.velocity * dt
        P = self._var[trk.track_id]
        # 预测协方差膨胀(过程噪声)。
        P = Vec3(P.x + self.process_var * dt, P.y + self.process_var * dt,
                 P.z + self.process_var * dt)
        res = m.position - pred
        new_pos = []
        new_vel = []
        new_P = []
        for p_pred, p_vel, res_a, P_a, R_a in (
            (pred.x, trk.velocity.x, res.x, P.x, m.var.x),
            (pred.y, trk.velocity.y, res.y, P.y, m.var.y),
            (pred.z, trk.velocity.z, res.z, P.z, m.var.z),
        ):
            K = P_a / (P_a + max(R_a, 1e-6))     # 逐轴卡尔曼增益
            new_pos.append(p_pred + K * res_a)
            new_vel.append(p_vel + (self.beta * K / dt) * res_a)
            new_P.append((1.0 - K) * P_a)
        trk.position = Vec3(*new_pos)
        trk.velocity = Vec3(*new_vel)
        self._var[trk.track_id] = Vec3(*new_P)
        trk.last_update = now
        trk.coast_time = 0.0
        trk.hits += 1
        if trk.hits >= self.confirm_threshold:
            trk.confirmed = True
        trk.contributing_sensors = set(m.sensors)
        trk.modalities = set(m.modalities)
        if m.classification is not None:
            trk.classification = m.classification
        if m.rf_emitter:
            trk.rf_emitter = True

    def _spawn(self, m: _CovMeas, now: float) -> None:
        tid = self._mint("TRK")
        self.tracks[tid] = Track(
            track_id=tid, position=m.position, velocity=Vec3(), last_update=now,
            contributing_sensors=set(m.sensors), modalities=set(m.modalities),
            classification=m.classification, rf_emitter=m.rf_emitter,
            confirmed=self.confirm_threshold <= 1, hits=1, coast_time=0.0,
        )
        self._var[tid] = m.var
