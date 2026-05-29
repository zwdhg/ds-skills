"""仿真传感器(雷达)模型。

每部雷达固定布站,对作用距离内的目标按探测概率产生**带噪**量测。
噪声幅度随距离增大、随 RCS 减小而升高,以贴近真实精度衰减规律。
多部雷达对同一目标的并发量测,正是航迹融合需要处理的输入。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from c2sim.geometry import Vec3
from c2sim.models import SensorReport, Target


@dataclass
class Radar:
    """固定布站的监视雷达。"""

    sensor_id: str
    position: Vec3
    max_range: float = 120_000.0  # 作用距离(米)
    base_sigma: float = 25.0      # 近界基础量测误差(米)
    detect_prob: float = 0.95     # 作用距离内的单帧探测概率

    def measurement_sigma(self, target: Target) -> float:
        """该次量测的位置误差(米)。距离越远、RCS 越小,误差越大。"""
        dist = self.position.distance_to(target.position)
        range_factor = 1.0 + 4.0 * (dist / self.max_range)
        rcs_factor = 1.0 / max(0.1, target.rcs) ** 0.5
        return self.base_sigma * range_factor * rcs_factor

    def detection_probability(self, target: Target) -> float:
        """对给定目标的本帧探测概率,超出作用距离为 0。"""
        dist = self.position.distance_to(target.position)
        if dist > self.max_range:
            return 0.0
        # RCS 偏小则探测概率下降。
        rcs_penalty = min(1.0, target.rcs ** 0.25)
        return self.detect_prob * rcs_penalty

    def observe(
        self, targets: list[Target], now: float, rng: random.Random
    ) -> list[SensorReport]:
        """对一批目标产生本帧量测报告。"""
        reports: list[SensorReport] = []
        for tgt in targets:
            if not tgt.alive:
                continue
            if rng.random() > self.detection_probability(tgt):
                continue
            sigma = self.measurement_sigma(tgt)
            noisy = Vec3(
                tgt.position.x + rng.gauss(0.0, sigma),
                tgt.position.y + rng.gauss(0.0, sigma),
                tgt.position.z + rng.gauss(0.0, sigma),
            )
            reports.append(
                SensorReport(
                    sensor_id=self.sensor_id,
                    timestamp=now,
                    position=noisy,
                    position_sigma=sigma,
                    truth_id=tgt.target_id,
                )
            )
        return reports
