"""均匀网格空间索引(无第三方依赖)。

把最近邻 / 半径查询从逐个比对的 O(N) 降到与局部密度相关的 O(k),为大规模
(集群/蜂群)场景下的"就近目标"查询提供加速路径。索引按水平 (x, y) 分桶;
因三维距离 ≥ 水平距离,以水平半径取候选再用真三维距离精筛,结果**精确**。

网格仅是加速结构:返回的最近邻 / 半径集合与暴力遍历**完全一致**(见
``tests/test_spatial.py`` 的等价性测试),故可无损替换热路径中的暴力查询。
"""

from __future__ import annotations

from c2sim.geometry import Vec3


class SpatialGrid:
    """二维均匀网格;存放带载荷的三维点,支持半径与最近邻精确查询。"""

    def __init__(self, cell: float = 1000.0) -> None:
        self.cell = cell
        self._cells: dict[tuple[int, int], list[tuple[Vec3, object]]] = {}
        # 占用格的包围盒(min_gx, min_gy, max_gx, max_gy),用于最近邻搜索定界。
        self._bbox: tuple[int, int, int, int] | None = None

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell), int(y // self.cell))

    def insert(self, pos: Vec3, payload: object) -> None:
        gx, gy = self._key(pos.x, pos.y)
        self._cells.setdefault((gx, gy), []).append((pos, payload))
        if self._bbox is None:
            self._bbox = (gx, gy, gx, gy)
        else:
            x0, y0, x1, y1 = self._bbox
            self._bbox = (min(x0, gx), min(y0, gy), max(x1, gx), max(y1, gy))

    def query_radius(self, center: Vec3, radius: float) -> list[object]:
        """返回与 ``center`` 三维距离 ≤ ``radius`` 的全部载荷(精确)。"""
        out: list[object] = []
        span = int(radius // self.cell) + 1
        cx, cy = self._key(center.x, center.y)
        for gx in range(cx - span, cx + span + 1):
            for gy in range(cy - span, cy + span + 1):
                for pos, payload in self._cells.get((gx, gy), ()):
                    if center.distance_to(pos) <= radius:
                        out.append(payload)
        return out

    def nearest(self, center: Vec3) -> object | None:
        """返回距 ``center`` 最近的载荷(精确,与暴力遍历一致);空索引返回 None。

        按倍增扩张水平搜索 span;一旦当前最优在**保证已覆盖半径**
        ``(span-1)·cell`` 内即可确认为全局最近;``max_span`` 由占用格包围盒相对
        查询点算出(O(1)),确保查询点落在占用区**之外**时也能扩到覆盖全部占用
        格,杜绝漏检/返回 None。
        """
        if self._bbox is None:
            return None
        cx, cy = self._key(center.x, center.y)
        x0, y0, x1, y1 = self._bbox
        # 覆盖全部占用格所需的最大 span。
        max_span = max(abs(cx - x0), abs(cx - x1), abs(cy - y0), abs(cy - y1)) + 1
        span = 1
        while True:
            span = min(span, max_span)
            best, best_d = None, float("inf")
            for gx in range(cx - span, cx + span + 1):
                for gy in range(cy - span, cy + span + 1):
                    for pos, payload in self._cells.get((gx, gy), ()):
                        d = center.distance_to(pos)
                        if d < best_d:
                            best, best_d = payload, d
            if best is not None and best_d <= (span - 1) * self.cell:
                return best
            if span >= max_span:
                return best  # 已检视全部占用格,best 即全局最近
            span *= 2
