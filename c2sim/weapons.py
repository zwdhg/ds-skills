"""Thunder 自主截击机与发射平台模型(纯仿真)。

* :class:`LaunchPad` —— 发射平台:存放/填装 Thunder,远程控制发射;受**作业
  半径**约束(Thunder 只能在以平台为中心的作业半径内交战)。
* :class:`Thunder` —— 已发射、自主飞行的截击机,经历三个阶段:

  1. **起飞抵近(APPROACH)**:依指控上行航迹做指令制导,飞向预测拦截点;
  2. **目标搜索(SEARCH)**:抵近后弹载 AI 启动,按截获概率搜索发现目标;
  3. **末段拦截(TERMINAL)**:稳定锁定后以弹载图像 AI 高精度寻的,动能撞击。

阶段转换与制导由引擎驱动(见 :mod:`c2sim.engine`);本模块负责弹道推进、
作业半径判定与库存管理。是否命中由引擎在引爆时依真值与杀伤半径判定。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from c2sim.geometry import Vec3
from c2sim.models import next_id


class Phase(enum.Enum):
    APPROACH = "起飞抵近"
    SEARCH = "目标搜索"
    TERMINAL = "末段拦截"


@dataclass
class Thunder:
    """在飞的 Thunder 截击机。"""

    interceptor_id: str
    pad_id: str
    target_track_id: str
    origin: Vec3               # 发射平台位置(作业半径基准)
    operating_radius: float    # 作业半径(米)
    position: Vec3
    velocity: Vec3
    max_speed: float
    lethal_radius: float
    acquisition_range: float   # 弹载传感器截获距离(米)
    acquisition_prob: float    # 进入截获距离后每帧锁定概率
    seeker_sigma: float        # 末段图像寻的测量误差(米)
    intercept_point: Vec3
    intercept_time: float      # 绝对仿真时间(仅供制导/复盘参考)
    launch_time: float
    phase: Phase = Phase.APPROACH
    acquired: bool = False
    locked_target_id: str | None = None  # 末段锁定的真实目标(弹上导引头)
    miss_distance: float = float("inf")   # 引爆时的真实脱靶量(米)
    # 已锁定目标后投入攻击,作业半径留有余度(末段不因边界而放弃)。
    terminal_range_margin: float = 1.5
    alive: bool = True
    detonated: bool = False

    def steer_to(self, point: Vec3, intercept_time: float) -> None:
        """把速度矢量指向(更新后的)拦截点,按最大速度飞行。"""
        self.intercept_point = point
        self.intercept_time = intercept_time
        self.velocity = (point - self.position).unit() * self.max_speed

    def out_of_range(self) -> bool:
        """是否已飞出作业半径(应自毁/返航,不再有效)。

        未锁定时按作业半径约束;已锁定(末段攻击)时放宽至
        ``operating_radius × terminal_range_margin``。
        """
        limit = self.operating_radius
        if self.acquired:
            limit *= self.terminal_range_margin
        return self.origin.distance_to(self.position) > limit


@dataclass
class LaunchPad:
    """Thunder 发射平台。"""

    pad_id: str
    position: Vec3
    inventory: int = 4                 # 在架 Thunder 数量
    operating_radius: float = 5_000.0  # 作业半径(米)
    thunder_max_speed: float = 66.7    # ≈240 km/h
    lethal_radius: float = 12.0        # 动能撞击/战斗部有效半径(米)
    acquisition_range: float = 1_200.0
    acquisition_prob: float = 0.7
    seeker_sigma: float = 4.0          # 末段图像寻的测量误差(米)

    def can_reach(self, point: Vec3) -> bool:
        """预测拦截点是否在作业半径内且尚有在架 Thunder。"""
        if self.inventory <= 0:
            return False
        return self.position.distance_to(point) <= self.operating_radius

    def fire(
        self,
        track_id: str,
        intercept_point: Vec3,
        intercept_time: float,
        now: float,
    ) -> Thunder:
        """发射一架 Thunder 飞向预测拦截点,库存减一。"""
        if self.inventory <= 0:
            raise RuntimeError(f"发射平台 {self.pad_id} 已无在架 Thunder")
        self.inventory -= 1
        direction = (intercept_point - self.position).unit()
        return Thunder(
            interceptor_id=next_id("THDR"),
            pad_id=self.pad_id,
            target_track_id=track_id,
            origin=self.position,
            operating_radius=self.operating_radius,
            position=self.position,
            velocity=direction * self.thunder_max_speed,
            max_speed=self.thunder_max_speed,
            lethal_radius=self.lethal_radius,
            acquisition_range=self.acquisition_range,
            acquisition_prob=self.acquisition_prob,
            seeker_sigma=self.seeker_sigma,
            intercept_point=intercept_point,
            intercept_time=intercept_time,
            launch_time=now,
        )
