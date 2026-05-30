# AGENTS.md —— 长期记忆 / Agent 工作约定

> 任何 AI Agent 在本仓库工作前**必读**。这是跨会话的"宪法":定位、硬规则、
> 架构不变量、开发循环。短期状态见根目录 [`progress.txt`](progress.txt);
> 意图见 [`docs/prd.md`](docs/prd.md);验收见 [`docs/user-stories.md`](docs/user-stories.md);
> 边界见 [`docs/model-card.md`](docs/model-card.md)。

## 1. 这是什么(与不是什么)

`c2sim`:**离线、零依赖的反无人机(C-UAS)仿真分析平台**。
**纯软件仿真,非火控系统**——不接入真实装备、不读真实世界目标数据、不产生
现实动作。结论仅用于**相对比较**,不作绝对性能承诺。这条范围红线不可逾越。

## 2. 硬规则(违反即回退)

1. **零第三方依赖**:仅标准库。`python -m unittest discover -s tests` 必须全过。
2. **确定性**:结构性重构(不改算法语义的重构)须保持**点状防护逐种子逐字节
   不变**——核心是不改变随机数调用顺序(末段寻的→Pk→量测→截获)。改前后用
   `for s in 1 7 42 2026; do python -m c2sim.cli --seed $s; done` 核对。
3. **诚实度量**:任何**行为变更**必须用蒙特卡洛前后对照佐证(`--monte-carlo`)。
   **严禁为匹配目标值反向调参**(如把 Pk 设成 0.92 去"验证"规格 ≥90%)。宁可
   数字不好看也要真实涌现。
4. **真值隔离**:真值物理只在 `world.World`;指控链路只见带噪航迹。新增的真值
   变更须走 World 方法(如 `apply_jamming`),不得由控制器直写 World 状态。
5. **新增局限即声明**:发现/引入任何会影响结论的简化,写入 `docs/model-card.md`。

## 3. 架构与不变量

三分:**被控对象 / 控制器 / 记录**
- `world.World` —— 真值与物理(运动学、近炸引信、毁伤涌现、软杀伤、突防)。
- `engine.Engine` —— 控制器 + 编排(调度注入策略、台账、事件→统计/历史)。
- `engine.Trace`(result + history)—— 记录;可视化只依赖此窄视图。

**可替换算法经 `strategies.py` 协议依赖注入**(新增实现零改引擎):
`SensorModel` / `Tracker` / `ThreatModel` / `GuidanceLaw` / `WeaponTargetAssigner`。
已落地多实现:`TrackFusion`↔`CovarianceTracker`、`LeadPursuit`↔`PurePursuit`。

模块:`geometry`(数学)·`models`(数据)·`sensors`·`fusion`/`tracking`·`threat`·
`interception`·`weapons`·`world`·`engine`·`viz`·`metrics`·`sensitivity`·
`scenario_io`·`scenarios`·`cli`。依赖向 `geometry`/`models` 内核收敛,无环。

## 4. 开发循环(自迭代回路)

**不追求一次写对;每一轮都接一个"它骗不过的反馈源"**:

```
改动 → 跑测试(unittest)
     → 行为变更?→ 蒙特卡洛前后对照(--monte-carlo / sensitivity),如实记录
     → 多角度/对抗式审查(code-review;分布外、边界、稀疏、杂波输入)
     → 批判性复盘(质疑假设,不只查 bug)
     → 更新 progress.txt(短期记忆)+ 必要时 model-card / user-stories
```

**铁律(从踩坑中来)**:
- **测试全绿 ≠ 正确**:同一个脑子写的测试与代码共享盲区(我们曾 120 绿却被
  评审挖出 7 个真 bug)。用**独立于、敌视代码**的反馈:分布外输入 + 多角度评审。
- **改 bug 先写复现失败测试**,再修;**核对修复覆盖所有实例**(grep 清点,
  曾"修了 CMD/THDR 漏了 TRK")。
- **会迭代,也要知道何时停**:越过价值边界加功能就是镀金(全协方差/JPDA/IMM
  当前按需再做)。**有些错迭代修不掉**(无 oracle、循环标定)——那需要外部
  数据,不是再迭代几轮。

## 5. 扩展指引

- 加算法:实现对应 `strategies.py` 协议 → `Engine(scenario, tracker=/guidance=/...)`
  注入 → 加协议符合性 + 契约测试 + 与默认的 MOE 对照。
- 加想定:`scenarios.py` 构造器,或 JSON + `scenario_io`(同步 to_dict/校验白名单)。
- 加 MOE:`metrics.py`;加部署/压力想定后在 `user-stories.md` 补故事与测试。

## 6. 常用命令

```bash
python -m unittest discover -s tests          # 全部测试(质量门)
bash scripts/check.sh                          # 测试 + 冒烟(提交前)
python -m c2sim.cli --scenario {point,border,swarm,decoy} [--events|--plot out.svg]
python -m c2sim.cli --monte-carlo 50           # 效能度量
python -m c2sim.cli --sensitivity 20           # 参数敏感性
python scripts/benchmark_spatial.py            # 空间索引基准
```

## 7. 交付

开发分支 `claude/exercise-command-control-a7Qmu` → PR #1。提交信息说明"做了什么 +
为何可信(对照数据)+ 边界",符合上述硬规则。
