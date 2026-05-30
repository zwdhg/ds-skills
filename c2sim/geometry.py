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


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def los_basis(sensor: Vec3, target: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    """构造以传感器→目标视线(LOS)为基准的正交基。

    返回 ``(e_range, e_az, e_el)``:

    * ``e_range`` —— 沿视线方向(测距误差作用轴);
    * ``e_az``    —— 水平面内、垂直于视线(方位角误差作用轴);
    * ``e_el``    —— 垂直于前两者(俯仰角误差作用轴,低仰角时近似铅垂)。

    当视线近似铅垂导致 ``e_az`` 退化时,回退到世界 x 轴构造,保证正交。
    """
    e_range = (target - sensor).unit()
    up = Vec3(0.0, 0.0, 1.0)
    e_az = up_cross = _cross(up, e_range)
    if e_az.norm() < 1e-6:
        e_az = _cross(Vec3(1.0, 0.0, 0.0), e_range)
    e_az = e_az.unit()
    e_el = _cross(e_range, e_az).unit()
    return e_range, e_az, e_el


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def angular_measurement_noise(
    sensor: Vec3,
    target: Vec3,
    sigma_range: float,
    sigma_az_rad: float,
    sigma_el_rad: float,
    rng,
) -> Vec3:
    """在 LOS 基下生成各向异性量测噪声并叠加到真值位置。

    距离误差沿视线;方位/俯仰角误差按 ``距离 × 角误差`` 折算为横向位移,
    分别作用于 ``e_az`` / ``e_el`` 轴。俯仰角误差通常大于方位角误差,因而
    高度方向误差更大——这正是参考资料指出的"雷达俯仰测量误差大"。
    """
    e_range, e_az, e_el = los_basis(sensor, target)
    r = sensor.distance_to(target)
    return (
        target
        + e_range * rng.gauss(0.0, sigma_range)
        + e_az * rng.gauss(0.0, r * sigma_az_rad)
        + e_el * rng.gauss(0.0, r * sigma_el_rad)
    )


def turn_towards(current: Vec3, desired: Vec3, max_angle: float) -> Vec3:
    """把 ``current`` 朝 ``desired`` 方向至多旋转 ``max_angle`` 弧度。

    返回矢量的**模长取 ``desired`` 的模长**(即保持期望速率),方向被限制在
    单步可达的转角内——用于对拦截弹施加转弯率(横向过载)约束。两矢量近似
    同向或反向退化时做安全回退。
    """
    speed = desired.norm()
    if speed == 0.0:
        return Vec3()
    cu = current.unit()
    du = desired.unit()
    if cu.norm() == 0.0:
        return desired
    dot = max(-1.0, min(1.0, cu.dot(du)))
    angle = math.acos(dot)
    if angle <= max_angle or angle == 0.0:
        return desired
    perp = du - cu * dot
    if perp.norm() < 1e-9:
        # 几乎反向:任取一条与 cu 垂直的方向起转。
        ref = Vec3(1.0, 0.0, 0.0) if abs(cu.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
        perp = ref - cu * cu.dot(ref)
    perp = perp.unit()
    new_dir = cu * math.cos(max_angle) + perp * math.sin(max_angle)
    return new_dir * speed


def los_diag_var(
    sensor: Vec3, target: Vec3, sigma_range: float,
    sigma_az_rad: float, sigma_el_rad: float,
) -> Vec3:
    """LOS 各向异性量测误差投影到世界轴的**对角方差**(var_x, var_y, var_z)。

    在 LOS 基下,误差沿距离/方位/俯仰三轴的方差分别为 ``σr²``、``(r·σ_az)²``、
    ``(r·σ_el)²``;将协方差 ``R = B·diag·Bᵀ`` 的对角投影到世界轴,得到各世界轴
    的方差。俯仰误差大 → 高度(z)方差大,正是协方差跟踪器据以"少信高度"的
    依据。忽略轴间相关项(对角近似),足以体现多模态融合收紧高度的工程价值。
    """
    e_r, e_az, e_el = los_basis(sensor, target)
    r = sensor.distance_to(target)
    vr, va, ve = sigma_range**2, (r * sigma_az_rad)**2, (r * sigma_el_rad)**2
    return Vec3(
        e_r.x**2 * vr + e_az.x**2 * va + e_el.x**2 * ve,
        e_r.y**2 * vr + e_az.y**2 * va + e_el.y**2 * ve,
        e_r.z**2 * vr + e_az.z**2 * va + e_el.z**2 * ve,
    )


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


def segment_cpa(rel_pos: Vec3, rel_vel: Vec3, dt: float) -> tuple[float, float]:
    """两点在 ``[0, dt]`` 内做匀速相对运动时的最近接近。

    给定相对位置 ``rel_pos`` 与相对速度 ``rel_vel``,返回 ``(t*, d*)``:
    区间内最近时刻(裁剪到 ``[0, dt]``)与该最近距离。用于近炸引信:判断
    Thunder 是否在本仿真步内掠过目标的杀伤半径。
    """
    v2 = rel_vel.dot(rel_vel)
    if v2 == 0.0:
        return 0.0, rel_pos.norm()
    t = -rel_pos.dot(rel_vel) / v2
    t = max(0.0, min(dt, t))
    return t, (rel_pos + rel_vel * t).norm()


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
