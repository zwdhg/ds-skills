"""Thunder 拦截武器模型(纯仿真)。

包含两类对象:

* :class:`ThunderBattery` —— 固定布站的发射单元,持有有限的拦截弹库存、
  作用距离与杀伤半径;
* :class:`Interceptor` —— 已发射、朝预测拦截点定速飞行的拦截弹。

是否命中由引擎在拦截弹引爆时,依据其与目标**真值**位置之差和杀伤半径
判定(见 :mod:`c2sim.engine`)。本模块只负责弹道推进与库存管理。
"""

from __future__ import annotations

from dataclasses import dataclass

from c2sim.geometry import Vec3
from c2sim.models import next_id


@dataclass
class Interceptor:
    """在飞拦截弹。"""

    interceptor_id: str
    battery_id: str
    target_track_id: str
    position: Vec3
    velocity: Vec3
    speed: float
    lethal_radius: float
    intercept_point: Vec3
    intercept_time: float  # 绝对仿真时间
    launch_time: float
    terminal_range: float = 3_000.0  # 末制导导引头截获距离(米)
    seeker_sigma: float = 10.0       # 末制导导引头测量误差(米)
    alive: bool = True
    detonated: bool = False

    def steer_to(self, intercept_point: Vec3, intercept_time: float) -> None:
        """中段制导:把速度矢量重新指向更新后的预测拦截点。"""
        self.intercept_point = intercept_point
        self.intercept_time = intercept_time
        self.velocity = (intercept_point - self.position).unit() * self.speed

    def advance(self, dt: float, step_start: float) -> None:
        """推进弹道。

        ``step_start`` 为本仿真步起始时刻。若计划引爆时刻落在本步之内,
        则只推进到引爆时刻并标记引爆——避免大步长导致的越界脱靶。
        """
        if not self.alive or self.detonated:
            return
        t_remain = self.intercept_time - step_start
        if t_remain <= dt:
            self.position = self.position + self.velocity * max(t_remain, 0.0)
            self.detonated = True
        else:
            self.position = self.position + self.velocity * dt


@dataclass
class ThunderBattery:
    """Thunder 发射单元。"""

    battery_id: str
    position: Vec3
    inventory: int = 8                  # 拦截弹库存
    interceptor_speed: float = 1000.0   # 拦截弹速度(米/秒)
    max_range: float = 80_000.0         # 最大拦截作用距离(米)
    min_range: float = 2_000.0          # 最小作用距离(过近不交战)
    lethal_radius: float = 60.0         # 杀伤半径(米)
    terminal_range: float = 3_000.0     # 拦截弹末制导截获距离(米)
    seeker_sigma: float = 10.0          # 拦截弹导引头测量误差(米)

    def can_reach(self, point: Vec3) -> bool:
        """预测拦截点是否落在本单元作用距离内且尚有库存。"""
        if self.inventory <= 0:
            return False
        d = self.position.distance_to(point)
        return self.min_range <= d <= self.max_range

    def fire(
        self,
        track_id: str,
        intercept_point: Vec3,
        intercept_time: float,
        now: float,
    ) -> Interceptor:
        """发射一发拦截弹至预测拦截点,库存减一。

        调用方须先用 :meth:`can_reach` 校验;此处仅在库存耗尽时报错。
        """
        if self.inventory <= 0:
            raise RuntimeError(f"发射单元 {self.battery_id} 库存已空")
        self.inventory -= 1
        direction = (intercept_point - self.position).unit()
        return Interceptor(
            interceptor_id=next_id("THDR"),
            battery_id=self.battery_id,
            target_track_id=track_id,
            position=self.position,
            velocity=direction * self.interceptor_speed,
            speed=self.interceptor_speed,
            lethal_radius=self.lethal_radius,
            intercept_point=intercept_point,
            intercept_time=intercept_time,
            launch_time=now,
            terminal_range=self.terminal_range,
            seeker_sigma=self.seeker_sigma,
        )
