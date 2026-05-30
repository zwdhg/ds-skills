"""态势可视化 —— 生成俯视 SVG 态势图(无第三方依赖)。

绘制内容:安全穹顶、Spotter Pro 探测站及覆盖扇区、发射平台及作业半径、
Hunter Max 干扰圈、各目标航迹(按结局着色)、Thunder 飞行轨迹,以及
摧毁/软杀伤/突防事件标记。

输入为想定(:class:`c2sim.engine.Scenario`)与运行记录
(:class:`c2sim.engine.Trace`,含 ``history`` 与 ``result``)——只依赖这两个
窄视图,不依赖整个 :class:`c2sim.engine.Engine`(接口隔离)。SVG 为纯文本
矢量图,任意浏览器可直接查看,无需额外依赖。
"""

from __future__ import annotations

import math

# 结局配色。
_FATE_COLOR = {
    "destroyed": "#e23b3b",   # Thunder 硬杀伤
    "soft_killed": "#1ca9a0",  # Hunter Max 软杀伤
    "leaked": "#7a0010",      # 突防
    "unresolved": "#888888",  # 在途未决
    "inbound": "#e8862b",     # 默认来袭
}


def _fate_of(tid: str, result) -> str:
    if tid in result.destroyed:
        return "destroyed"
    if tid in result.soft_killed:
        return "soft_killed"
    if tid in result.leaked:
        return "leaked"
    if tid in result.unresolved:
        return "unresolved"
    return "inbound"


class _Canvas:
    """世界坐标→SVG 像素的等比变换(y 轴翻转,使北向上)。"""

    def __init__(self, pts, width, height, margin):
        xs = [p[0] for p in pts] or [0.0]
        ys = [p[1] for p in pts] or [0.0]
        self.minx, self.maxx = min(xs), max(xs)
        self.miny, self.maxy = min(ys), max(ys)
        spanx = max(self.maxx - self.minx, 1.0)
        spany = max(self.maxy - self.miny, 1.0)
        self.scale = min((width - 2 * margin) / spanx, (height - 2 * margin) / spany)
        self.ox = (width - spanx * self.scale) / 2.0
        self.oy = (height - spany * self.scale) / 2.0
        self.height = height

    def x(self, wx: float) -> float:
        return self.ox + (wx - self.minx) * self.scale

    def y(self, wy: float) -> float:
        return self.height - (self.oy + (wy - self.miny) * self.scale)

    def r(self, wr: float) -> float:
        return wr * self.scale


def render_svg(scenario, trace, path: str, title: str = "态势图",
               width: int = 920, height: int = 920) -> str:
    """渲染态势图并写入 ``path``,返回 SVG 文本。

    参数:
        scenario: 想定(要地/探测站/发射平台/干扰单元布局)。
        trace: 运行记录(``trace.history`` 航迹、``trace.result`` 结果)。
    """
    s = scenario
    hist = trace.history
    result = trace.result

    # 收集用于确定视野范围的所有世界坐标点。
    pts: list[tuple[float, float]] = [(s.asset.x, s.asset.y)]
    for unit in (*s.spotters, *s.pads, *s.jammers):
        pts.append((unit.position.x, unit.position.y))
    for p in s.pads:
        pts += [(p.position.x + p.operating_radius, p.position.y),
                (p.position.x - p.operating_radius, p.position.y),
                (p.position.x, p.position.y + p.operating_radius),
                (p.position.x, p.position.y - p.operating_radius)]
    for seq in hist.target_paths.values():
        pts += seq
    for seq in hist.thunder_paths.values():
        pts += seq

    c = _Canvas(pts, width, height, margin=70)
    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">'
    )
    out.append(f'<rect width="{width}" height="{height}" fill="#0e1116"/>')

    # 安全穹顶。
    ax, ay = c.x(s.asset.x), c.y(s.asset.y)
    dome = c.r(s.threat_policy.defended_radius)
    out.append(
        f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{dome:.1f}" fill="#3b82f6" '
        f'fill-opacity="0.10" stroke="#3b82f6" stroke-opacity="0.5" '
        f'stroke-dasharray="6 4"/>'
    )

    # Hunter Max 干扰圈。
    for j in s.jammers:
        out.append(
            f'<circle cx="{c.x(j.position.x):.1f}" cy="{c.y(j.position.y):.1f}" '
            f'r="{c.r(j.jam_range):.1f}" fill="#1ca9a0" fill-opacity="0.06" '
            f'stroke="#1ca9a0" stroke-opacity="0.35" stroke-dasharray="2 4"/>'
        )

    # 发射平台作业半径。
    for p in s.pads:
        out.append(
            f'<circle cx="{c.x(p.position.x):.1f}" cy="{c.y(p.position.y):.1f}" '
            f'r="{c.r(p.operating_radius):.1f}" fill="none" stroke="#9aa7b4" '
            f'stroke-opacity="0.18"/>'
        )

    # Spotter Pro 覆盖扇区。
    for sp in s.spotters:
        out.append(_coverage_sector(c, sp))

    # 目标航迹。
    for tid, seq in hist.target_paths.items():
        if len(seq) < 2:
            continue
        color = _FATE_COLOR[_fate_of(tid, result)]
        out.append(_polyline(c, seq, color, 2.4, opacity=0.95))
        out.append(_triangle(c.x(seq[0][0]), c.y(seq[0][1]), color))  # 起点

    # Thunder 轨迹。
    for seq in hist.thunder_paths.values():
        if len(seq) >= 2:
            out.append(_polyline(c, seq, "#5b8def", 1.0, opacity=0.5))

    # 事件标记。
    for mx, my, kind in hist.markers:
        out.append(_event_marker(c.x(mx), c.y(my), kind))

    # 单元图标。
    for sp in s.spotters:
        out.append(_icon(c.x(sp.position.x), c.y(sp.position.y), "#7dd3fc", "▣"))
    for p in s.pads:
        out.append(_icon(c.x(p.position.x), c.y(p.position.y), "#cbd5e1", "▲"))
    for j in s.jammers:
        out.append(_icon(c.x(j.position.x), c.y(j.position.y), "#2dd4bf", "≈"))
    out.append(_icon(ax, ay, "#fbbf24", "◎"))  # 要地

    out.append(_legend(width, title, result))
    out.append("</svg>")
    svg = "\n".join(out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return svg


# --- 绘制基元 -------------------------------------------------------------


def _polyline(c, seq, color, w, opacity=1.0) -> str:
    pts = " ".join(f"{c.x(x):.1f},{c.y(y):.1f}" for x, y in seq)
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="{w}" stroke-opacity="{opacity}"/>'
    )


def _coverage_sector(c, sp) -> str:
    cx, cy = c.x(sp.position.x), c.y(sp.position.y)
    rr = c.r(sp.coverage_radius)
    if sp.azimuth_width_deg >= 360.0:
        return (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rr:.1f}" fill="#38bdf8" '
            f'fill-opacity="0.05" stroke="#38bdf8" stroke-opacity="0.25"/>'
        )
    a0 = math.radians(sp.azimuth_center_deg - sp.azimuth_width_deg / 2.0)
    a1 = math.radians(sp.azimuth_center_deg + sp.azimuth_width_deg / 2.0)
    # 世界方位 → SVG(y 翻转,用 -sin)。
    p0 = (cx + rr * math.cos(a0), cy - rr * math.sin(a0))
    p1 = (cx + rr * math.cos(a1), cy - rr * math.sin(a1))
    large = 1 if sp.azimuth_width_deg > 180 else 0
    return (
        f'<path d="M{cx:.1f},{cy:.1f} L{p0[0]:.1f},{p0[1]:.1f} '
        f'A{rr:.1f},{rr:.1f} 0 {large} 0 {p1[0]:.1f},{p1[1]:.1f} Z" '
        f'fill="#38bdf8" fill-opacity="0.06" stroke="#38bdf8" stroke-opacity="0.25"/>'
    )


def _triangle(x, y, color) -> str:
    return (
        f'<path d="M{x:.1f},{y-5:.1f} L{x-4.5:.1f},{y+4:.1f} L{x+4.5:.1f},{y+4:.1f} Z" '
        f'fill="{color}"/>'
    )


def _event_marker(x, y, kind) -> str:
    if kind == "摧毁":
        return (
            f'<path d="M{x-5:.1f},{y-5:.1f} L{x+5:.1f},{y+5:.1f} '
            f'M{x-5:.1f},{y+5:.1f} L{x+5:.1f},{y-5:.1f}" '
            f'stroke="#ff5252" stroke-width="2.5"/>'
        )
    if kind == "软杀伤":
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="none" '
            f'stroke="#2dd4bf" stroke-width="2.5"/>'
        )
    # 突防
    return (
        f'<rect x="{x-4.5:.1f}" y="{y-4.5:.1f}" width="9" height="9" '
        f'fill="none" stroke="#ff2d55" stroke-width="2.5"/>'
    )


def _icon(x, y, color, glyph) -> str:
    return (
        f'<text x="{x:.1f}" y="{y+5:.1f}" font-size="15" fill="{color}" '
        f'text-anchor="middle">{glyph}</text>'
    )


def _legend(width, title, result) -> str:
    rows = [
        ("#fbbf24", "◎ 要地 / 安全穹顶"),
        ("#7dd3fc", "▣ Spotter Pro + 覆盖"),
        ("#cbd5e1", "▲ 发射平台 + 作业半径"),
        ("#2dd4bf", "≈ Hunter Max 干扰圈"),
        ("#e8862b", "── 目标航迹"),
        ("#5b8def", "── Thunder 轨迹"),
        ("#ff5252", "✕ 摧毁  ◯ 软杀伤  ☐ 突防"),
    ]
    parts = [
        f'<text x="20" y="30" font-size="18" fill="#e6edf3" '
        f'font-weight="bold">{title}</text>',
        f'<text x="20" y="52" font-size="13" fill="#9aa7b4">{result.summary()}</text>',
    ]
    y = 76
    for color, label in rows:
        parts.append(
            f'<text x="20" y="{y}" font-size="12.5" fill="{color}">{label}</text>'
        )
        y += 19
    return "\n".join(parts)
