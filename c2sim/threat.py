"""威胁研判。

输入是指控链路可见的**航迹**(只含运动学,不含真值身份),输出是
针对被掩护要地(defended asset)的威胁研判结论。研判由四个因子合成:

* **企图(intent)**——航迹是否会穿入要地防护半径(用 CPA 判断);
* **紧迫(urgency)**——预计抵达要地的时间,越短越紧迫;
* **逼近(proximity)**——当前距要地的远近;
* **杀伤(lethality)**——由航迹运动学反推的目标类型对应的杀伤性。

目标类型靠运动学(速度/高度)在线分类得到,而非读取真值——这与真实
指控系统"先发现航迹、再判性质"的流程一致。
"""

from __future__ import annotations

from dataclasses import dataclass

from c2sim.geometry import Vec3, closest_point_of_approach
from c2sim.models import TargetKind, ThreatAssessment, ThreatLevel, Track


@dataclass
class ThreatPolicy:
    """威胁研判的可调参数(战场/想定相关)。"""

    defended_radius: float = 5_000.0   # "安全穹顶"防护半径(米)
    horizon: float = 120.0             # 研判时间视界(秒)
    max_range: float = 10_000.0        # 逼近因子归一化用的参考距离(米)
    # 四因子权重(自动归一化)。
    w_intent: float = 0.40
    w_urgency: float = 0.30
    w_proximity: float = 0.15
    w_lethality: float = 0.15
    # 威胁分到等级的阈值(升序)。
    level_thresholds: tuple[float, float, float, float] = (0.15, 0.35, 0.55, 0.78)


def classify_track(track: Track) -> TargetKind:
    """推断航迹的目标类型。

    **优先采用光电(EO)识别结果**(若航迹已被光电确认);否则退回到基于
    运动学的粗分类——这恰好体现了为何需要光电识别:仅凭运动学难以可靠
    区分固定翼无人机与巡飞弹。
    """
    if track.classification is not None:
        return track.classification
    speed = track.velocity.norm()
    if speed > 60.0:
        return TargetKind.LOITERING_MUNITION
    if speed >= 30.0:
        return TargetKind.FIXED_WING_UAV
    if speed >= 8.0:
        return TargetKind.ROTARY_UAV
    return TargetKind.MICRO_UAV


def assess_track(
    track: Track, asset: Vec3, policy: ThreatPolicy
) -> ThreatAssessment:
    """对单条航迹做威胁研判。"""
    dist_now = track.position.distance_to(asset)
    t_cpa, d_cpa = closest_point_of_approach(
        track.position, track.velocity, asset, Vec3()
    )

    will_penetrate = d_cpa <= policy.defended_radius
    speed = track.velocity.norm()

    # 抵达时间:仅当确将穿入防护半径时才有意义。
    time_to_impact: float | None
    if will_penetrate and speed > 1e-3:
        # 估算进入防护半径的时刻(早于最近接近时刻)。
        # 用 CPA 几何近似:穿入弦的半长除以速度。
        chord = max(0.0, policy.defended_radius**2 - d_cpa**2) ** 0.5
        time_to_impact = max(0.0, t_cpa - chord / speed)
    else:
        time_to_impact = None

    # --- 四因子 ---
    if will_penetrate:
        intent = 1.0
    else:
        # 擦肩而过:按偏离防护半径的程度衰减。
        intent = max(0.0, policy.defended_radius / max(d_cpa, 1e-3))
        intent = min(intent, 0.6)

    if time_to_impact is None:
        urgency = 0.0
    else:
        urgency = max(0.0, 1.0 - time_to_impact / policy.horizon)

    proximity = max(0.0, 1.0 - dist_now / policy.max_range)

    kind = classify_track(track)
    lethality = kind.lethality

    wsum = (
        policy.w_intent
        + policy.w_urgency
        + policy.w_proximity
        + policy.w_lethality
    )
    raw = (
        policy.w_intent * intent
        + policy.w_urgency * urgency
        + policy.w_proximity * proximity
        + policy.w_lethality * lethality
    ) / wsum

    # 航迹置信度对威胁分作折扣:质量差的航迹不轻易拉高威胁。
    score = round(raw * (0.5 + 0.5 * track.confidence), 4)
    level = _score_to_level(score, policy.level_thresholds)

    rationale = (
        f"{kind.value}: CPA={d_cpa/1000:.1f}km "
        f"{'穿透' if will_penetrate else '掠过'}, "
        f"距要地={dist_now/1000:.1f}km, "
        f"抵达={'%.0fs' % time_to_impact if time_to_impact is not None else 'N/A'}, "
        f"航迹置信={track.confidence:.2f}"
    )

    return ThreatAssessment(
        track_id=track.track_id,
        score=score,
        level=level,
        time_to_impact=time_to_impact,
        closest_approach=d_cpa,
        rationale=rationale,
    )


def _score_to_level(
    score: float, thresholds: tuple[float, float, float, float]
) -> ThreatLevel:
    t_low, t_med, t_high, t_crit = thresholds
    if score >= t_crit:
        return ThreatLevel.CRITICAL
    if score >= t_high:
        return ThreatLevel.HIGH
    if score >= t_med:
        return ThreatLevel.MEDIUM
    if score >= t_low:
        return ThreatLevel.LOW
    return ThreatLevel.NONE


def assess(
    tracks: list[Track], asset: Vec3, policy: ThreatPolicy
) -> list[ThreatAssessment]:
    """对一批航迹研判,按威胁分从高到低排序返回。"""
    results = [assess_track(t, asset, policy) for t in tracks]
    results.sort(key=lambda a: a.score, reverse=True)
    return results
