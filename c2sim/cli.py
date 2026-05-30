"""命令行入口:运行演练想定并打印复盘报告。

    python -m c2sim.cli                       # 点状防护演示想定
    python -m c2sim.cli --scenario border     # 边境带状防护
    python -m c2sim.cli --events              # 打印逐条事件时间线
    python -m c2sim.cli --plot out.svg        # 生成态势图
    python -m c2sim.cli --scenario-file s.json   # 从 JSON 想定运行
    python -m c2sim.cli --monte-carlo 50      # 蒙特卡洛 50 次,打印效能度量
"""

from __future__ import annotations

import argparse
import json

from c2sim.engine import Engine, Scenario, SimResult

_SCENARIOS = {
    "point": "核心要域点状防护",
    "border": "边境线带状防护",
    "swarm": "蜂群突击(规模/饱和)",
    "decoy": "亚视场诱饵(误关联压力)",
}


def _named_builder(key: str):
    from c2sim.scenarios import (
        build_border_band_scenario,
        build_decoy_scenario,
        build_point_defense_scenario,
        build_swarm_scenario,
    )
    return {
        "point": build_point_defense_scenario,
        "border": build_border_band_scenario,
        "swarm": build_swarm_scenario,
        "decoy": build_decoy_scenario,
    }[key]


def _file_builder(path: str):
    """由 JSON 想定文件构造工厂:每次以指定种子重新解析,保证对象独立。"""
    from c2sim.scenario_io import from_dict

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def build(seed: int) -> Scenario:
        data = dict(raw)
        data["seed"] = seed
        return from_dict(data)

    return build


def _print_report(name: str, scenario: Scenario, result: SimResult, events: bool) -> None:
    print("=" * 64)
    print("  演练指挥控制系统 · Skyshield Nexus 仿真复盘")
    print(f"  部署样式:{name}")
    print("=" * 64)
    total_cov = sum(s.coverage_area_km2() for s in scenario.spotters)
    # 多站直接相加会重复计入重叠区,仅作粗略上界,故注明"含重叠"。
    overlap_note = "(各站合计,含重叠)" if len(scenario.spotters) > 1 else ""
    print(
        f"探测站 {len(scenario.spotters)} | 发射平台 {len(scenario.pads)} | "
        f"雷达覆盖{overlap_note} ≈ {total_cov:.1f} km²"
    )
    print(result.summary())
    print("-" * 64)
    if result.destroyed:
        print("已摧毁: " + ", ".join(result.destroyed))
    if result.soft_killed:
        print("软杀伤(干扰迫降/返航): " + ", ".join(result.soft_killed))
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
        "--scenario", choices=sorted(_SCENARIOS), default="point", help="内置部署样式"
    )
    parser.add_argument("--scenario-file", metavar="PATH", help="从 JSON 想定文件运行")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子(批量起始)")
    parser.add_argument("--events", action="store_true", help="打印逐条事件时间线")
    parser.add_argument("--plot", metavar="PATH", help="生成态势 SVG 图并写入路径")
    parser.add_argument(
        "--monte-carlo", type=int, metavar="N", default=0,
        help="蒙特卡洛运行 N 次(种子 seed..seed+N-1)并打印效能度量",
    )
    parser.add_argument(
        "--sensitivity", type=int, metavar="N", default=0,
        help="参数敏感性扫描(每点 N 次蒙特卡洛),打印各参数对突防率的摆幅",
    )
    args = parser.parse_args(argv)

    if args.scenario_file:
        builder = _file_builder(args.scenario_file)
        name = f"自定义想定({args.scenario_file})"
    else:
        builder = _named_builder(args.scenario)
        name = _SCENARIOS[args.scenario]

    # 参数敏感性模式。
    if args.sensitivity and args.sensitivity > 1:
        from c2sim.sensitivity import default_sweep, format_sweep

        seeds = range(args.seed, args.seed + args.sensitivity)
        print("=" * 64)
        print(f"  Skyshield Nexus · 参数敏感性 · 部署:{name}")
        print("=" * 64)
        print(format_sweep(default_sweep(builder, seeds)))
        print("=" * 64)
        return 0

    # 蒙特卡洛模式。
    if args.monte_carlo and args.monte_carlo > 1:
        from c2sim.metrics import BatchMoe, run_batch

        seeds = range(args.seed, args.seed + args.monte_carlo)
        runs = run_batch(builder, seeds)
        print("=" * 64)
        print(f"  Skyshield Nexus · 效能度量(MOE) · 部署:{name}")
        print("=" * 64)
        print(BatchMoe.aggregate(runs).table())
        print("=" * 64)
        return 0

    # 单次模式。
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
