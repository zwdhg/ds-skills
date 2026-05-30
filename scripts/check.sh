#!/usr/bin/env bash
# 提交前质量门:全量测试 + 各想定冒烟 + 蒙特卡洛 sanity。
# 用法: bash scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 单元测试 =="
python -m unittest discover -s tests

echo "== 2/3 想定冒烟 =="
for s in point border swarm decoy; do
  python -m c2sim.cli --scenario "$s" >/dev/null && echo "  ok: $s"
done

echo "== 3/3 蒙特卡洛 sanity（点状应高处置率）=="
python -m c2sim.cli --monte-carlo 20 | grep 处置率

echo "全部检查通过。"
