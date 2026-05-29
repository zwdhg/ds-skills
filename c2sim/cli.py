"""命令行入口:运行演练想定并打印复盘报告。

    python -m c2sim.cli                       # 点状防护演示想定
    python -m c2sim.cli --scenario border     # 边境带状防护
    python -m c2sim.cli --events              # 打印逐条事件时间线
    python -m c2sim.cli --seed 7
"""

from __future__ import annotations

import argparse

from c2sim.engine import Engine, Scenario, SimResult
from c2sim.scenarios import (
    build_border_band_scenario,
    build_point_defense_scenario,
)

_SCENARIOS = {
    "point": ("核心要域点状防护", build_point_defense_scenario),
    "border": ("边境线带状防护", build_border_band_scenario),
}


def _print_report(name: str, scenario: Scenario, result: SimResult, events: bool) -> None:
    print("=" * 64)
    print("  演练指挥控制系统 · Skyshield Nexus 仿真复盘")
    print(f"  部署样式:{name}")
    print("=" * 64)
    total_cov = sum(s.coverage_area_km2() for s in scenario.spotters)
    print(
        f"探测站 {len(scenario.spotters)} | 发射平台 {len(scenario.pads)} | "
        f"雷达覆盖合计 ≈ {total_cov:.1f} km²"
    )
    print(result.summary())
    print("-" * 64)
    if result.destroyed:
        print("已摧毁: " + ", ".join(result.destroyed))
    if result.leaked:
        print("突防(未拦截): " + ", ".join(result.leaked))
    if result.unresolved:
        print("在途未决: " + ", ".join(result.unresolved))
    print(f"指令总数: {len(result.commands)}")

    if events:
        print("-" * 64)
        print("事件时间线:")
        for ev in result.events:
            print(f"  [{ev.time:7.1f}s] {ev.kind:4s} | {ev.detail}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="演练指挥控制系统仿真(反无人机)")
    parser.add_argument(
        "--scenario", choices=sorted(_SCENARIOS), default="point", help="部署样式"
    )
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--events", action="store_true", help="打印逐条事件时间线")
    parser.add_argument(
        "--plot", metavar="PATH", help="生成态势 SVG 图并写入指定路径"
    )
    args = parser.parse_args(argv)

    name, builder = _SCENARIOS[args.scenario]
    scenario = builder(seed=args.seed)
    engine = Engine(scenario)
    result = engine.run()
    _print_report(name, scenario, result, events=args.events)

    if args.plot:
        from c2sim.viz import render_svg

        render_svg(scenario, engine.trace, args.plot, title=f"态势图 · {name}")
        print(f"态势图已写入: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
