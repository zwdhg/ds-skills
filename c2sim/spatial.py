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

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell), int(y // self.cell))

    def insert(self, pos: Vec3, payload: object) -> None:
        self._cells.setdefault(self._key(pos.x, pos.y), []).append((pos, payload))

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
        """返回距 ``center`` 最近的载荷(精确);空索引返回 None。

        从一个网格半径起按倍增扩张半径,直至命中;由于半径查询精确,首个
        非空半径内的最小三维距离即全局最近邻。
        """
        if not self._cells:
            return None
        radius = self.cell
        # 索引非空,扩张上界以保证终止(覆盖整个已用范围)。
        max_radius = self.cell * (self._extent() + 2)
        while radius <= max_radius:
            best, best_d = None, float("inf")
            span = int(radius // self.cell) + 1
            cx, cy = self._key(center.x, center.y)
            for gx in range(cx - span, cx + span + 1):
                for gy in range(cy - span, cy + span + 1):
                    for pos, payload in self._cells.get((gx, gy), ()):
                        d = center.distance_to(pos)
                        if d < best_d:
                            best, best_d = payload, d
            # 已检候选覆盖到 span*cell 的水平范围;若最近者在更内圈则可信。
            if best is not None and best_d <= (span - 1) * self.cell:
                return best
            if best is not None and radius >= max_radius:
                return best
            radius *= 2.0
        return best

    def _extent(self) -> int:
        xs = [k[0] for k in self._cells]
        ys = [k[1] for k in self._cells]
        return max(max(xs) - min(xs), max(ys) - min(ys)) if xs else 0
