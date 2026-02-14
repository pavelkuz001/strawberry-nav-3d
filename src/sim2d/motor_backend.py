from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import Sim2DConfig
from .motor_math import limit_vw


@dataclass
class MotorOutput:
    v: float
    w: float


class MotorBackend:
    """Abstract motor backend. Later can be replaced with ROS-based backend."""
    def set_cmd(self, v_cmd: float, w_cmd: float, dt: float) -> MotorOutput:
        raise NotImplementedError


@dataclass
class SimMotorBackend(MotorBackend):
    cfg: Sim2DConfig
    v_prev: float = 0.0
    w_prev: float = 0.0

    def set_cmd(self, v_cmd: float, w_cmd: float, dt: float) -> MotorOutput:
        v, w = limit_vw(v_cmd, w_cmd, self.v_prev, self.w_prev, dt=dt, cfg=self.cfg)
        self.v_prev, self.w_prev = v, w
        return MotorOutput(v=v, w=w)

    def reset(self) -> None:
        self.v_prev = 0.0
        self.w_prev = 0.0
