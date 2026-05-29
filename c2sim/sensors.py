"""Spotter Pro —— 地面多模态目标侦测系统(仿真)。

复现参考系统"侦测—识别—跟踪"一体化能力的三个模块及其级联引导关系:

1. **频谱测向(RF)**:二维定向、远程预警(≥10km),为雷达提供方位线索;
   仅能截获**辐射射频信号**的目标,且不提供距离——因此建模为"截获线索"
   (cue)而非三维位置量测。
2. **X 波段 AESA 雷达**:三维位置量测,作用距离随 RCS 变化;**俯仰角误差
   大于方位角误差**(高度误差更大);跟踪容量 TAS 受限(≥6);存在近界盲区。
3. **光电(EO)**:高精度角度量测(0.15 mrad)+ 目标识别;伺服转台同一时刻
   仅服务一个目标;在雷达引导下完成识别与精确锁定,显著收紧高度精度。

各模块产出统一的 :class:`SensorReport` 喂入航迹融合。Spotter Pro 自主调度
雷达 TAS 与光电转台(基于就近优先),无需指控干预。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from c2sim.geometry import Vec3, angular_measurement_noise, deg2rad
from c2sim.models import SensorModality, SensorReport, Target, TargetKind


@dataclass
class SpotterPro:
    """多模态探测站。

    坐标以站址为参考;``azimuth_center`` / ``azimuth_width`` 描述水平覆盖
    扇区(点状防护取 360°,边境带状取 180°)。
    """

    station_id: str
    position: Vec3
    azimuth_center_deg: float = 0.0
    azimuth_width_deg: float = 360.0

    # --- 频谱测向 ---
    rf_range: float = 10_000.0
    rf_detect_prob: float = 0.95

    # --- AESA 雷达 ---
    radar_ref_range: float = 10_000.0   # @ 参考 RCS
    radar_ref_rcs: float = 0.2
    radar_range_exp: float = 0.231      # 距离-RCS 指数(标定:0.01㎡→5km,0.2㎡→10km)
    coverage_radius: float = 5_000.0    # 对小型固定翼的有效探测半径(覆盖评估用)
    radar_blind_zone: float = 200.0
    radar_elevation_max_deg: float = 30.0
    radar_sigma_range: float = 10.0
    radar_sigma_az: float = field(default_factory=lambda: deg2rad(0.5))
    radar_sigma_el: float = field(default_factory=lambda: deg2rad(1.5))  # 俯仰更差
    radar_tas_capacity: int = 6
    radar_detect_prob: float = 0.95

    # --- 光电 ---
    eo_range: float = 12_000.0
    eo_capacity: int = 1
    eo_sigma_range: float = 15.0
    eo_sigma_ang: float = 0.15e-3       # 0.15 mrad
    eo_classify_prob: float = 0.9       # 识别正确概率

    # 内部状态:当前雷达 TAS 跟踪与光电锁定的目标(以真值索引,代表硬件波束指向)
    _tas: set[str] = field(default_factory=set)
    _eo_lock: str | None = None

    # -- 几何判定 ---------------------------------------------------------

    def _in_sector(self, target: Target) -> bool:
        if self.azimuth_width_deg >= 360.0:
            return True
        d = target.position - self.position
        az = math.degrees(math.atan2(d.y, d.x))
        diff = abs((az - self.azimuth_center_deg + 180.0) % 360.0 - 180.0)
        return diff <= self.azimuth_width_deg / 2.0

    def _elevation_deg(self, target: Target) -> float:
        d = target.position - self.position
        horiz = math.hypot(d.x, d.y)
        return math.degrees(math.atan2(d.z, max(horiz, 1e-6)))

    def radar_range_for(self, rcs: float) -> float:
        """按 RCS 折算雷达作用距离(幂律,经规格表两点标定)。"""
        return self.radar_ref_range * (rcs / self.radar_ref_rcs) ** self.radar_range_exp

    def _radar_detectable(self, target: Target) -> bool:
        if not self._in_sector(target):
            return False
        d = self.position.distance_to(target.position)
        if d < self.radar_blind_zone or d > self.radar_range_for(target.rcs):
            return False
        return self._elevation_deg(target) <= self.radar_elevation_max_deg

    def _rf_detectable(self, target: Target) -> bool:
        return (
            target.emits_rf
            and self._in_sector(target)
            and self.position.distance_to(target.position) <= self.rf_range
        )

    # -- 每帧探测 ---------------------------------------------------------

    def observe(
        self, targets: list[Target], now: float, rng: random.Random
    ) -> list[SensorReport]:
        alive = [t for t in targets if t.alive]
        reports: list[SensorReport] = []

        # 1) 频谱测向:产生方位截获线索(优先引导雷达)。
        cued: set[str] = set()
        for t in alive:
            if self._rf_detectable(t) and rng.random() <= self.rf_detect_prob:
                cued.add(t.target_id)

        # 2) 雷达 TAS:维持已跟踪目标,空余容量按"已被RF引导优先、其次就近"补充。
        detectable = [t for t in alive if self._radar_detectable(t)]
        det_ids = {t.target_id for t in detectable}
        self._tas &= det_ids  # 丢弃已不可探测的航迹
        if len(self._tas) < self.radar_tas_capacity:
            candidates = [t for t in detectable if t.target_id not in self._tas]
            candidates.sort(
                key=lambda t: (
                    t.target_id not in cued,  # 被RF引导者优先(False 排前)
                    self.position.distance_to(t.position),
                )
            )
            for t in candidates:
                if len(self._tas) >= self.radar_tas_capacity:
                    break
                self._tas.add(t.target_id)

        tracked = [t for t in detectable if t.target_id in self._tas]
        for t in tracked:
            if rng.random() > self.radar_detect_prob:
                continue
            noisy = angular_measurement_noise(
                self.position,
                t.position,
                self.radar_sigma_range,
                self.radar_sigma_az,
                self.radar_sigma_el,
                rng,
            )
            r = self.position.distance_to(t.position)
            sigma = math.sqrt(
                (
                    self.radar_sigma_range**2
                    + (r * self.radar_sigma_az) ** 2
                    + (r * self.radar_sigma_el) ** 2
                )
                / 3.0
            )
            reports.append(
                SensorReport(
                    sensor_id=f"{self.station_id}/RADAR",
                    modality=SensorModality.RADAR,
                    timestamp=now,
                    position=noisy,
                    position_sigma=sigma,
                    truth_id=t.target_id,
                )
            )

        # 3) 光电:转台就近优先锁定一个雷达在跟目标,识别并高精度定位。
        eo_candidates = [
            t
            for t in tracked
            if self.position.distance_to(t.position) <= self.eo_range
        ]
        eo_candidates.sort(key=lambda t: self.position.distance_to(t.position))
        for t in eo_candidates[: self.eo_capacity]:
            self._eo_lock = t.target_id
            noisy = angular_measurement_noise(
                self.position,
                t.position,
                self.eo_sigma_range,
                self.eo_sigma_ang,
                self.eo_sigma_ang,
                rng,
            )
            r = self.position.distance_to(t.position)
            sigma = math.sqrt(
                (
                    self.eo_sigma_range**2
                    + 2.0 * (r * self.eo_sigma_ang) ** 2
                )
                / 3.0
            )
            reports.append(
                SensorReport(
                    sensor_id=f"{self.station_id}/EO",
                    modality=SensorModality.EO,
                    timestamp=now,
                    position=noisy,
                    position_sigma=sigma,
                    classification=self._classify(t, rng),
                    truth_id=t.target_id,
                )
            )

        return reports

    def _classify(self, target: Target, rng: random.Random) -> TargetKind:
        """光电识别:高概率给出正确类型,否则误判为相近类型。"""
        if rng.random() <= self.eo_classify_prob:
            return target.kind
        others = [k for k in TargetKind if k != target.kind]
        return rng.choice(others)

    # -- 覆盖能力 ---------------------------------------------------------

    def coverage_area_km2(self) -> float:
        """该站对小型固定翼目标的有效覆盖面积(平方公里),用于部署评估。

        以 ``coverage_radius``(对 2.5m 翼展固定翼的有效探测距离 ≈5km)为半径,
        按扇区角度折算。360°→≈78.5 km²;180°→≈39.27 km²,与参考资料一致。
        """
        r_km = self.coverage_radius / 1000.0
        return math.pi * r_km**2 * (self.azimuth_width_deg / 360.0)
