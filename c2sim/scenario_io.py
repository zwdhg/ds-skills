"""想定即数据 —— 声明式想定的 JSON 读写与校验。

让分析人员以 JSON 编写/版本化想定,无需改代码。仅用标准库 ``json``;
校验为显式手写(无 jsonschema 依赖),对缺失键/类型错误给出清晰报错。

结构示例见 ``docs/examples/scenario.json``。
"""

from __future__ import annotations

import json

from c2sim.engine import Scenario
from c2sim.geometry import Vec3
from c2sim.interception import EngagementPolicy
from c2sim.models import Target, TargetKind, ThreatLevel
from c2sim.sensors import SpotterPro
from c2sim.threat import ThreatPolicy
from c2sim.weapons import HunterMax, LaunchPad


class ScenarioError(ValueError):
    """想定数据非法。"""


def _vec3(value, where: str) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ScenarioError(f"{where}: 期望 [x, y, z] 三元组,得到 {value!r}")
    try:
        return Vec3(float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        raise ScenarioError(f"{where}: 坐标必须为数字,得到 {value!r}")


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise ScenarioError(f"{where}: 缺少必需字段 '{key}'")
    return d[key]


def _positive(value, where: str) -> float:
    """要求为正数,否则报错(拦截负速度/负距离等无意义输入)。"""
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ScenarioError(f"{where}: 期望数字,得到 {value!r}")
    if not x > 0.0:
        raise ScenarioError(f"{where}: 必须为正数,得到 {x}")
    return x


def _apply(cls, d: dict, where: str, *, positional: dict, optional: set):
    """构造 dataclass:positional 为 名→转换后值,optional 中的键若存在则透传。"""
    kwargs = dict(positional)
    allowed = optional
    for k, v in d.items():
        if k in positional:
            continue
        if k not in allowed:
            raise ScenarioError(f"{where}: 未知字段 '{k}'")
        kwargs[k] = v
    return cls(**kwargs)


# --- 反序列化 -------------------------------------------------------------


def _target(d: dict, i: int) -> Target:
    where = f"targets[{i}]"
    if not isinstance(d, dict):
        raise ScenarioError(f"{where}: 期望对象")
    kind_str = d.get("kind", TargetKind.FIXED_WING_UAV.value)
    try:
        kind = TargetKind(kind_str)
    except ValueError:
        valid = ", ".join(k.value for k in TargetKind)
        raise ScenarioError(f"{where}.kind: 非法类型 '{kind_str}'(可选:{valid})")
    return Target(
        target_id=str(_require(d, "target_id", where)),
        position=_vec3(_require(d, "position", where), f"{where}.position"),
        aim=_vec3(_require(d, "aim", where), f"{where}.aim"),
        cruise_speed=_positive(_require(d, "cruise_speed", where),
                               f"{where}.cruise_speed"),
        kind=kind,
        rcs=_positive(d.get("rcs", 0.1), f"{where}.rcs"),
        terminal_speed=(None if d.get("terminal_speed") is None
                        else _positive(d["terminal_speed"], f"{where}.terminal_speed")),
        terminal_range=_positive(d.get("terminal_range", 1500.0),
                                 f"{where}.terminal_range"),
        emits_rf=bool(d.get("emits_rf", True)),
    )


def _spotter(d: dict, i: int) -> SpotterPro:
    where = f"spotters[{i}]"
    sid = str(_require(d, "station_id", where))
    pos = _vec3(_require(d, "position", where), f"{where}.position")
    opt = {"azimuth_center_deg", "azimuth_width_deg", "rf_range", "rf_detect_prob",
           "radar_ref_range", "radar_ref_rcs", "radar_range_exp", "coverage_radius",
           "radar_blind_zone", "radar_elevation_max_deg", "radar_sigma_range",
           "radar_tas_capacity", "radar_detect_prob", "eo_range", "eo_capacity",
           "eo_sigma_range", "eo_sigma_ang", "eo_classify_prob"}
    extra = {k: v for k, v in d.items()
             if k not in ("station_id", "position")}
    bad = set(extra) - opt
    if bad:
        raise ScenarioError(f"{where}: 未知字段 {sorted(bad)}")
    return SpotterPro(sid, pos, **extra)


def _pad(d: dict, i: int) -> LaunchPad:
    where = f"pads[{i}]"
    pid = str(_require(d, "pad_id", where))
    pos = _vec3(_require(d, "position", where), f"{where}.position")
    opt = {"inventory", "operating_radius", "thunder_max_speed", "lethal_radius",
           "acquisition_range", "acquisition_prob", "seeker_sigma",
           "terminal_range_margin"}
    extra = {k: v for k, v in d.items() if k not in ("pad_id", "position")}
    bad = set(extra) - opt
    if bad:
        raise ScenarioError(f"{where}: 未知字段 {sorted(bad)}")
    return LaunchPad(pid, pos, **extra)


def _jammer(d: dict, i: int) -> HunterMax:
    where = f"jammers[{i}]"
    jid = str(_require(d, "jammer_id", where))
    pos = _vec3(_require(d, "position", where), f"{where}.position")
    opt = {"jam_range", "hold_time"}
    extra = {k: v for k, v in d.items() if k not in ("jammer_id", "position")}
    bad = set(extra) - opt
    if bad:
        raise ScenarioError(f"{where}: 未知字段 {sorted(bad)}")
    return HunterMax(jid, pos, **extra)


def _threat_policy(d: dict) -> ThreatPolicy:
    opt = {f.name for f in ThreatPolicy.__dataclass_fields__.values()}
    bad = set(d) - opt
    if bad:
        raise ScenarioError(f"threat_policy: 未知字段 {sorted(bad)}")
    return ThreatPolicy(**d)


def _engagement_policy(d: dict) -> EngagementPolicy:
    d = dict(d)
    if "engage_level" in d and isinstance(d["engage_level"], str):
        try:
            d["engage_level"] = ThreatLevel[d["engage_level"]]
        except KeyError:
            raise ScenarioError(
                f"engagement_policy.engage_level: 非法等级 '{d['engage_level']}'")
    allowed = {"engage_level", "prefer_jamming", "engage_radius_frac"}
    bad = set(d) - allowed
    if bad:
        raise ScenarioError(f"engagement_policy: 未知/不支持字段 {sorted(bad)}")
    return EngagementPolicy(**d)


def from_dict(data: dict) -> Scenario:
    """由字典构造想定(带校验)。"""
    if not isinstance(data, dict):
        raise ScenarioError("顶层应为对象")
    targets = [_target(t, i) for i, t in enumerate(_require(data, "targets", "root"))]
    spotters = [_spotter(s, i) for i, s in enumerate(data.get("spotters", []))]
    pads = [_pad(p, i) for i, p in enumerate(data.get("pads", []))]
    jammers = [_jammer(j, i) for i, j in enumerate(data.get("jammers", []))]
    if not targets:
        raise ScenarioError("targets 不可为空")

    kwargs = {}
    if "threat_policy" in data:
        kwargs["threat_policy"] = _threat_policy(data["threat_policy"])
    if "engagement_policy" in data:
        kwargs["engagement_policy"] = _engagement_policy(data["engagement_policy"])
    for k in ("dt", "max_time", "seed", "single_shot_pk"):
        if k in data:
            kwargs[k] = data[k]

    return Scenario(
        asset=_vec3(_require(data, "asset", "root"), "asset"),
        targets=targets, spotters=spotters, pads=pads, jammers=jammers, **kwargs,
    )


def load(path: str) -> Scenario:
    with open(path, encoding="utf-8") as f:
        return from_dict(json.load(f))


# --- 序列化 ---------------------------------------------------------------


def to_dict(scenario: Scenario) -> dict:
    """把想定导出为可 JSON 化的字典(便于落盘/版本化)。"""
    def v(p: Vec3):
        return [p.x, p.y, p.z]

    return {
        "asset": v(scenario.asset),
        "dt": scenario.dt,
        "max_time": scenario.max_time,
        "seed": scenario.seed,
        "single_shot_pk": scenario.single_shot_pk,
        "spotters": [
            {"station_id": s.station_id, "position": v(s.position),
             "azimuth_center_deg": s.azimuth_center_deg,
             "azimuth_width_deg": s.azimuth_width_deg}
            for s in scenario.spotters
        ],
        "pads": [
            {"pad_id": p.pad_id, "position": v(p.position),
             "inventory": p.inventory, "operating_radius": p.operating_radius,
             "thunder_max_speed": p.thunder_max_speed}
            for p in scenario.pads
        ],
        "jammers": [
            {"jammer_id": j.jammer_id, "position": v(j.position),
             "jam_range": j.jam_range, "hold_time": j.hold_time}
            for j in scenario.jammers
        ],
        "targets": [
            {"target_id": t.target_id, "position": v(t.position), "aim": v(t.aim),
             "cruise_speed": t.cruise_speed, "kind": t.kind.value, "rcs": t.rcs,
             "terminal_speed": t.terminal_speed, "emits_rf": t.emits_rf}
            for t in scenario.targets
        ],
    }


def save(scenario: Scenario, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(scenario), f, ensure_ascii=False, indent=2)
