"""三维矢量与基础运动学工具。

坐标系约定:右手系,单位米;``z`` 为高度(海拔)。时间单位为秒,
速度单位为米/秒。仿真在直角坐标系下进行,不涉及地理投影。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec3:
    """不可变三维矢量。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Vec3":
        return Vec3(self.x / k, self.y / k, self.z / k)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def unit(self) -> "Vec3":
        """返回单位矢量;零矢量返回自身,避免除零。"""
        n = self.norm()
        return self if n == 0.0 else self / n

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).norm()

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def closest_point_of_approach(
    p1: Vec3, v1: Vec3, p2: Vec3, v2: Vec3
) -> tuple[float, float]:
    """计算两个匀速运动点的最近接近(CPA)。

    返回 ``(t_cpa, d_cpa)``:到达最近距离的时间(秒)与该最近距离(米)。
    ``t_cpa`` 被裁剪到非负——过去的最接近时刻对研判无意义。
    """
    dp = p1 - p2
    dv = v1 - v2
    dv2 = dv.dot(dv)
    if dv2 == 0.0:
        # 相对静止:距离恒定,最近时刻取当前。
        return 0.0, dp.norm()
    t = -dp.dot(dv) / dv2
    t = max(0.0, t)
    closest = (p1 + v1 * t) - (p2 + v2 * t)
    return t, closest.norm()


def lead_intercept_time(
    shooter: Vec3, target_pos: Vec3, target_vel: Vec3, speed: float
) -> float | None:
    """求解定速拦截弹命中匀速目标所需的飞行时间。

    在 ``shooter`` 发射、以恒定标量速度 ``speed`` 飞行的拦截弹,
    与从 ``target_pos`` 以 ``target_vel`` 匀速运动的目标,二者位置相等时:

        |target_pos + target_vel * t - shooter| = speed * t

    展开为关于 ``t`` 的二次方程。返回最小正实根(秒);若不可达,返回
    ``None``。
    """
    if speed <= 0.0:
        return None
    r = target_pos - shooter
    a = target_vel.dot(target_vel) - speed * speed
    b = 2.0 * r.dot(target_vel)
    c = r.dot(r)

    if abs(a) < 1e-9:
        # 退化为线性:b t + c = 0。
        if abs(b) < 1e-9:
            return 0.0 if c == 0.0 else None
        t = -c / b
        return t if t > 0.0 else None

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    roots = ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a))
    positive = sorted(t for t in roots if t > 1e-9)
    return positive[0] if positive else None
