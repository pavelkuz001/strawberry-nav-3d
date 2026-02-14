from __future__ import annotations

from dataclasses import dataclass


from pathlib import Path
import importlib.util
import numpy as np
from .config import Sim2DConfig
from .motor_math import limit_vw, restrict_vel_acc as _fallback_restrict_vel_acc
@dataclass
class MotorOutput:
    v: float
    w: float


class MotorBackend:
    """Abstract motor backend. Later can be replaced with ROS-based backend."""

    def set_cmd(self, v_cmd: float, w_cmd: float, dt: float) -> MotorOutput:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state (e.g., previous velocities)."""
        return None


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



def _try_load_runbot_restrict_vel_acc():
    """
    Try to load restrict_vel_acc from local dump:
      src_motors_gamepad/motors/runbot_motion_control/scripts/vel_acc_constraint.py
    If missing -> return (None, "fallback").
    """
    candidate = Path("src_motors_gamepad/motors/runbot_motion_control/scripts/vel_acc_constraint.py")
    if candidate.is_file():
        spec = importlib.util.spec_from_file_location("runbot_vel_acc_constraint", str(candidate))
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, "restrict_vel_acc", None)
            if callable(fn):
                return fn, str(candidate)
    return None, "fallback"


@dataclass
class RunbotMathBackend(MotorBackend):
    """
    Uses runbot_motion_control vel/acc limiter math (restrict_vel_acc).
    If local src_motors_gamepad dump exists -> loads it dynamically.
    Else -> uses internal fallback (byte-identical logic).
    """
    cfg: Sim2DConfig
    v_prev: float = 0.0
    w_prev: float = 0.0
    source: str = "fallback"
    restrict_fn: object = None  # set in __post_init__

    def __post_init__(self):
        fn, src = _try_load_runbot_restrict_vel_acc()
        self.restrict_fn = fn if fn is not None else _fallback_restrict_vel_acc
        self.source = src

    def set_cmd(self, v_cmd: float, w_cmd: float, dt: float) -> MotorOutput:
        max_v = float(getattr(self.cfg, "MAX_V", 0.4))
        max_w = float(getattr(self.cfg, "MAX_W", 0.8))
        max_a = float(getattr(self.cfg, "MAX_A", 0.8))
        max_aw = float(getattr(self.cfg, "MAX_W_ACC", 1.6))

        target = np.array([v_cmd, w_cmd], dtype=np.float32)
        prev = np.array([self.v_prev, self.w_prev], dtype=np.float32)
        max_vel = np.array([max_v, max_w], dtype=np.float32)

        max_acc = np.array([max_a, max_aw], dtype=np.float32) * float(dt)
        if np.linalg.norm(np.abs(max_acc)) < 1e-5:
            max_acc = np.ones(2, dtype=np.float32) * 1e-5

        out = self.restrict_fn(target, prev, max_vel, max_acc)
        v, w = float(out[0]), float(out[1])
        self.v_prev, self.w_prev = v, w
        return MotorOutput(v=v, w=w)

    def reset(self) -> None:
        self.v_prev = 0.0
        self.w_prev = 0.0


@dataclass
class RawMotorBackend(MotorBackend):
    """No motor limiting; passes commands through (debug mode)."""

    def set_cmd(self, v_cmd: float, w_cmd: float, dt: float) -> MotorOutput:
        return MotorOutput(v=float(v_cmd), w=float(w_cmd))


@dataclass
class RosStubMotorBackend(SimMotorBackend):
    """Placeholder for future ROS integration. For now uses same math limiting."""
    pass


def make_motor_backend(cfg: Sim2DConfig) -> MotorBackend:
    name = str(getattr(cfg, "MOTOR_BACKEND", "math")).lower().strip()

    if name in ("math", "sim", "default"):
        return SimMotorBackend(cfg)
    if name in ("runbot", "runbot-math"):
        return RunbotMathBackend(cfg)


    if name in ("raw", "passthrough"):
        return RawMotorBackend()

    if name in ("ros", "ros-stub", "rosstub"):
        return RosStubMotorBackend(cfg)

    # fallback
    return SimMotorBackend(cfg)
