"""命令行入口:运行演练想定并打印复盘报告。

    python -m c2sim.cli            # 运行内置演示想定
    python -m c2sim.cli --seed 7   # 指定随机种子
    python -m c2sim.cli --events   # 同时打印逐条事件时间线
"""

from __future__ import annotations

import argparse

from c2sim.engine import Engine, SimResult
from c2sim.scenarios import build_demo_scenario


def _print_report(result: SimResult, show_events: bool) -> None:
    print("=" * 60)
    print("  演练指挥控制系统 · 仿真复盘")
    print("=" * 60)
    print(result.summary())
    print("-" * 60)
    if result.destroyed:
        print("已摧毁: " + ", ".join(result.destroyed))
    if result.leaked:
        print("突防(未拦截): " + ", ".join(result.leaked))
    print(f"指令总数: {len(result.commands)}")

    if show_events:
        print("-" * 60)
        print("事件时间线:")
        for ev in result.events:
            print(f"  [{ev.time:7.1f}s] {ev.kind:4s} | {ev.detail}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="演练指挥控制系统仿真")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument(
        "--events", action="store_true", help="打印逐条事件时间线"
    )
    args = parser.parse_args(argv)

    scenario = build_demo_scenario(seed=args.seed)
    engine = Engine(scenario)
    result = engine.run()
    _print_report(result, show_events=args.events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
