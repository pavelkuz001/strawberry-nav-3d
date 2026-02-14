from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Callable, Optional, Tuple

import numpy as np


# Cache for optional runbot limiter
_RUNBOT_LOOKED_UP = False
_RUNBOT_RESTRICT_FN: Optional[Callable] = None
_RUNBOT_SRC: Optional[Path] = None


def _repo_root() -> Path:
    # src/sim2d/motor_math.py -> repo_root = parents[2]
    return Path(__file__).resolve().parents[2]


def _runbot_vel_acc_path() -> Path:
    return (
        _repo_root()
        / "src_motors_gamepad"
        / "motors"
        / "runbot_motion_control"
        / "scripts"
        / "vel_acc_constraint.py"
    )


def _get_runbot_restrict_fn() -> Optional[Callable]:
    """Try to load runbot vel_acc_constraint.restrict_vel_acc. Return None if unavailable."""
    global _RUNBOT_LOOKED_UP, _RUNBOT_RESTRICT_FN, _RUNBOT_SRC
    if _RUNBOT_LOOKED_UP:
        return _RUNBOT_RESTRICT_FN

    _RUNBOT_LOOKED_UP = True
    p = _runbot_vel_acc_path()
    _RUNBOT_SRC = p

    if not p.exists():
        _RUNBOT_RESTRICT_FN = None
        return None

    try:
        spec = importlib.util.spec_from_file_location("runbot_vel_acc_constraint", str(p))
        if spec is None or spec.loader is None:
            _RUNBOT_RESTRICT_FN = None
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        fn = getattr(mod, "restrict_vel_acc", None)
        _RUNBOT_RESTRICT_FN = fn if callable(fn) else None
        return _RUNBOT_RESTRICT_FN
    except Exception:
        _RUNBOT_RESTRICT_FN = None
        return None


def restrict_vel_acc_math(
    target_vel: np.ndarray, prev_vel: np.ndarray, max_vel: np.ndarray, max_acc: np.ndarray
) -> np.ndarray:
    """
    Pure-python copy of runbot_motion_control/scripts/vel_acc_constraint.py::restrict_vel_acc
    Vectors are [v, w].
    """
    target_vel = target_vel.astype(np.float32, copy=False)
    prev_vel = prev_vel.astype(np.float32, copy=False)
    max_vel = max_vel.astype(np.float32, copy=False)
    max_acc = max_acc.astype(np.float32, copy=False)

    assert prev_vel.shape == target_vel.shape
    assert max_vel.shape == target_vel.shape
    assert max_acc.shape == target_vel.shape

    assert np.min(max_vel) > 1e-7
    assert np.min(max_acc) > 1e-7

    t_vel = np.linalg.norm(target_vel / max_vel)
    if t_vel > 1:
        target_vel = target_vel / t_vel

    t_acc = np.linalg.norm((prev_vel - target_vel) / max_acc)
    if t_acc < 1:
        t_acc = 1

    v = prev_vel + (target_vel - prev_vel) / t_acc
    return v


def restrict_vel_acc(
    target_vel: np.ndarray, prev_vel: np.ndarray, max_vel: np.ndarray, max_acc: np.ndarray
) -> np.ndarray:
    """
    If runbot limiter is available locally -> use it.
    Otherwise -> fallback to restrict_vel_acc_math.
    """
    fn = _get_runbot_restrict_fn()
    if fn is not None:
        try:
            out = fn(target_vel, prev_vel, max_vel, max_acc)
            return np.array(out, dtype=np.float32)
        except Exception:
            pass
    return restrict_vel_acc_math(target_vel, prev_vel, max_vel, max_acc)


def limit_vw(v_cmd: float, w_cmd: float, v_prev: float, w_prev: float, dt: float, cfg) -> Tuple[float, float]:
    """
    Motor-like limiting of (v,w) using cfg limits and dt (like motors_node_x4).
    Requires cfg.MAX_V, cfg.MAX_A. For angular part uses cfg.MAX_W, cfg.MAX_W_ACC if present,
    otherwise falls back to reasonable defaults.
    """
    max_v = float(getattr(cfg, "MAX_V", 0.4))
    max_w = float(getattr(cfg, "MAX_W", 0.8))
    max_a = float(getattr(cfg, "MAX_A", 0.8))
    max_aw = float(getattr(cfg, "MAX_W_ACC", 1.6))

    target = np.array([v_cmd, w_cmd], dtype=np.float32)
    prev = np.array([v_prev, w_prev], dtype=np.float32)
    max_vel = np.array([max_v, max_w], dtype=np.float32)

    max_acc = np.array([max_a, max_aw], dtype=np.float32) * float(dt)
    if np.linalg.norm(np.abs(max_acc)) < 1e-5:
        max_acc = np.ones(2, dtype=np.float32) * 1e-5

    out = restrict_vel_acc(target, prev, max_vel, max_acc)
    return float(out[0]), float(out[1])
