"""效能度量(MOE/MOP)与蒙特卡洛批量评估。

把单次仿真结果归约为**效能度量**(:class:`Moe`),并对多随机种子批量运行
做统计聚合(:class:`BatchMoe`:均值、标准差、95% 置信区间、零突防概率),
使 `c2sim` 从"单次演示"成为"统计分析平台"。仅依赖标准库 ``statistics``。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass
class Moe:
    """单次仿真的效能度量。"""

    total: int
    destroyed: int       # Thunder 硬杀伤
    soft_killed: int     # Hunter Max 软杀伤
    leaked: int
    unresolved: int
    thunders: int        # 发射的 Thunder 数
    duration: float

    @property
    def neutralized(self) -> int:
        return self.destroyed + self.soft_killed

    @property
    def leakage_rate(self) -> float:
        return self.leaked / self.total if self.total else 0.0

    @property
    def neutralization_rate(self) -> float:
        return self.neutralized / self.total if self.total else 0.0

    @property
    def cost_per_kill(self) -> float:
        """硬杀伤效费比:消耗 Thunder 数 / 硬杀伤数(无硬杀伤为 inf)。"""
        return self.thunders / self.destroyed if self.destroyed else math.inf

    @classmethod
    def from_result(cls, result) -> "Moe":
        return cls(
            total=result.total_targets,
            destroyed=len(result.destroyed),
            soft_killed=len(result.soft_killed),
            leaked=len(result.leaked),
            unresolved=len(result.unresolved),
            thunders=result.thunders_launched,
            duration=result.duration,
        )


@dataclass
class _Stat:
    """一个标量度量在批量上的统计量。"""

    mean: float
    stdev: float
    ci95: tuple[float, float]  # 均值的 95% 置信区间(正态近似)

    @classmethod
    def of(cls, xs: list[float]) -> "_Stat":
        n = len(xs)
        if n == 0:
            return cls(mean=0.0, stdev=0.0, ci95=(0.0, 0.0))
        mean = statistics.fmean(xs)
        # 样本标准差(/n-1)作为离散度与均值标准误(SEM)的基准;n=1 退化为 0。
        # CI 为正态近似(n 较小时偏窄,详见 docs/model-card.md)。
        sd = statistics.stdev(xs) if n > 1 else 0.0
        half = 1.96 * sd / math.sqrt(n)
        return cls(mean=mean, stdev=sd, ci95=(mean - half, mean + half))


@dataclass
class BatchMoe:
    """多次仿真的聚合效能度量。"""

    n_runs: int
    leakage_rate: _Stat
    neutralization_rate: _Stat
    cost_per_kill: _Stat       # 仅统计有硬杀伤的样本
    thunders: _Stat
    duration: _Stat
    prob_zero_leak: float      # 零突防的比例

    @classmethod
    def aggregate(cls, runs: list[Moe]) -> "BatchMoe":
        if not runs:
            raise ValueError("无样本可聚合")
        # 仅统计有硬杀伤的样本;若全程无硬杀伤,效费比为未定义(inf),
        # 不可用 0.0 哨兵(否则"零硬杀伤"会被误报为"完美效费比")。
        finite_cpk = [r.cost_per_kill for r in runs if math.isfinite(r.cost_per_kill)]
        inf = float("inf")
        cpk = (_Stat.of(finite_cpk) if finite_cpk
               else _Stat(mean=inf, stdev=0.0, ci95=(inf, inf)))
        return cls(
            n_runs=len(runs),
            leakage_rate=_Stat.of([r.leakage_rate for r in runs]),
            neutralization_rate=_Stat.of([r.neutralization_rate for r in runs]),
            cost_per_kill=cpk,
            thunders=_Stat.of([float(r.thunders) for r in runs]),
            duration=_Stat.of([r.duration for r in runs]),
            prob_zero_leak=sum(1 for r in runs if r.leaked == 0) / len(runs),
        )

    def table(self) -> str:
        def pct(s: _Stat) -> str:
            return (f"{s.mean*100:5.1f}% ±{s.stdev*100:4.1f} "
                    f"[{s.ci95[0]*100:.1f}, {s.ci95[1]*100:.1f}]")

        cpk = (f"{self.cost_per_kill.mean:5.2f} 架/杀 ±{self.cost_per_kill.stdev:.2f}"
               if math.isfinite(self.cost_per_kill.mean) else "N/A(无硬杀伤)")
        return "\n".join([
            f"蒙特卡洛 {self.n_runs} 次:",
            f"  突防率          {pct(self.leakage_rate)}",
            f"  处置率          {pct(self.neutralization_rate)}",
            f"  零突防概率      {self.prob_zero_leak*100:5.1f}%",
            f"  硬杀伤效费比    {cpk}",
            f"  Thunder 消耗    {self.thunders.mean:5.2f} 架 "
            f"±{self.thunders.stdev:.2f}",
            f"  用时            {self.duration.mean:5.1f}s "
            f"±{self.duration.stdev:.1f}",
        ])


def run_batch(builder, seeds, **engine_kwargs) -> list[Moe]:
    """对一组随机种子批量运行,返回各次的 :class:`Moe`。

    参数:
        builder: 想定工厂,签名 ``builder(seed=int) -> Scenario``。
        seeds: 随机种子可迭代对象。
        engine_kwargs: 透传给 :class:`c2sim.engine.Engine` 的策略注入参数。
    """
    from c2sim.engine import Engine

    runs: list[Moe] = []
    for seed in seeds:
        result = Engine(builder(seed=seed), **engine_kwargs).run()
        runs.append(Moe.from_result(result))
    return runs
