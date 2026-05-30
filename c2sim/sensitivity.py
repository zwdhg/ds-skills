"""参数敏感性分析。

围绕基线**逐参数**扰动,蒙特卡洛评估各参数对关键效能度量(默认突防率)的
影响幅度,从而回答"结论对哪些参数敏感"——使 MOE 不再是孤立点估计,而带上
对输入假设的依赖说明。仅依赖标准库。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from c2sim.engine import Engine, Scenario
from c2sim.metrics import Moe


@dataclass
class SweepResult:
    """单参数扫描结果。"""

    param: str
    points: list[tuple[float, float]]  # (参数值, MOE 均值)

    @property
    def swing(self) -> float:
        """该参数在扫描范围内引起的 MOE 摆幅(max-min),作为敏感度。"""
        ms = [m for _, m in self.points]
        return max(ms) - min(ms)


def sweep_param(
    builder: Callable[..., Scenario],
    name: str,
    values: list[float],
    apply: Callable[[Scenario, float], None],
    seeds,
    metric: Callable[[Moe], float],
) -> SweepResult:
    """对单个参数取若干值,各值跑多种子取 MOE 均值。"""
    points: list[tuple[float, float]] = []
    for v in values:
        ms = []
        for s in seeds:
            scn = builder(seed=s)
            apply(scn, v)
            ms.append(metric(Moe.from_result(Engine(scn).run())))
        points.append((v, statistics.fmean(ms)))
    return SweepResult(name, points)


def _set_reliability(scn: Scenario, v: float) -> None:
    scn.warhead_reliability = v


def _set_turn(scn: Scenario, v: float) -> None:
    for p in scn.pads:
        p.max_turn_rate = v


def _set_speed(scn: Scenario, v: float) -> None:
    for p in scn.pads:
        p.thunder_max_speed = v


def _set_radar(scn: Scenario, v: float) -> None:
    for sp in scn.spotters:
        sp.radar_ref_range = v


def _set_jam(scn: Scenario, v: float) -> None:
    for j in scn.jammers:
        j.jam_range = v


def default_sweep(builder, seeds, metric=lambda m: m.leakage_rate) -> list[SweepResult]:
    """对一组关键参数做默认敏感性扫描,按敏感度(摆幅)降序返回。"""
    specs = [
        ("warhead_reliability", [0.80, 0.90, 0.95, 1.00], _set_reliability),
        ("thunder_max_turn_rate", [0.5, 1.0, 2.0, 4.0], _set_turn),
        ("thunder_max_speed", [55.0, 66.7, 80.0], _set_speed),
        ("radar_ref_range", [7000.0, 10000.0, 13000.0], _set_radar),
    ]
    # 仅当想定含干扰单元时扫描干扰半径。
    if builder(seed=0).jammers:
        specs.append(("jam_range", [3000.0, 4000.0, 5000.0], _set_jam))

    results = [sweep_param(builder, n, vals, fn, seeds, metric)
               for n, vals, fn in specs]
    results.sort(key=lambda r: r.swing, reverse=True)
    return results


def format_sweep(results: list[SweepResult], metric_name: str = "突防率") -> str:
    lines = [f"参数敏感性({metric_name},按摆幅降序):"]
    for r in results:
        pts = "  ".join(f"{v:g}→{m*100:.1f}%" for v, m in r.points)
        lines.append(f"  {r.param:22s} 摆幅 {r.swing*100:4.1f}pp | {pts}")
    return "\n".join(lines)
