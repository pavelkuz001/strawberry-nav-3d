import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import Sim2DConfig
from .geometry import pixel_depth_to_cam_xyz
from .frames import (
    cam_xyz_to_robot_xy,
    robot_xy_to_world_xy,
    world_xy_to_robot_xy,
    robot_xy_to_cam_xyz,
)


def _extract_detections(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Support a few possible keys
    for k in ("detections", "strawberries", "results"):
        v = data.get(k)
        if isinstance(v, list):
            return v
    return []


def _get_bbox_center(det: Dict[str, Any]) -> Optional[tuple]:
    bbox = det.get("bbox", {})
    if "center_x" in bbox and "center_y" in bbox:
        return float(bbox["center_x"]), float(bbox["center_y"])
    # fallback
    if "x1" in bbox and "x2" in bbox and "y1" in bbox and "y2" in bbox:
        return 0.5 * (float(bbox["x1"]) + float(bbox["x2"])), 0.5 * (float(bbox["y1"]) + float(bbox["y2"]))
    return None


def _get_depth_center_m(det: Dict[str, Any]) -> Optional[float]:
    d = det.get("depth", {})
    for k in ("center_meters", "mean_meters", "median_meters"):
        if k in d and d[k] is not None:
            return float(d[k])
    return None


@dataclass
class WorldBerry:
    id: int
    p_world: np.ndarray  # (x,y)
    class_id: Optional[int] = None
    class_name: Optional[str] = None


def world_from_detector_json(json_path: str, K: np.ndarray, pose0: np.ndarray, randomize_angle: bool = False) -> Dict[str, Any]:
    """
    One-time init:
      detector JSON (u,v,depth) @ pose0  -> fixed world points.
    """
    data = json.load(open(json_path, "r"))
    dets = _extract_detections(data)

    berries: List[WorldBerry] = []
    for det in dets:
        cid = det.get("id", len(berries))
        uv = _get_bbox_center(det)
        z = _get_depth_center_m(det)
        if uv is None or z is None:
            continue

        u, v = uv
        xyz_cam = pixel_depth_to_cam_xyz(u, v, z, K)
        p_robot = cam_xyz_to_robot_xy(xyz_cam)          # (x_fwd, y_left)

        if randomize_angle:
            # Randomize angle while keeping the same distance
            import random
            dist = np.linalg.norm(p_robot)
            angle = random.uniform(-np.pi, np.pi)
            p_robot = np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32)

        p_world = robot_xy_to_world_xy(p_robot, pose0)  # (x,y)

        berries.append(
            WorldBerry(
                id=int(cid),
                p_world=p_world.astype(np.float32),
                class_id=det.get("class_id"),
                class_name=det.get("class_name"),
            )
        )

    return {
        "source_json": str(json_path),
        "berries": berries,
    }


def simulate_detector_json(world: Dict[str, Any], pose: np.ndarray, cfg: Sim2DConfig, K: np.ndarray) -> Dict[str, Any]:
    """
    "Camera" refresh:
      fixed world berries + current pose -> JSON-like detector output.
    """
    fx, fy, cx, cy = float(cfg.FX), float(cfg.FY), float(cfg.CX), float(cfg.CY)

    detections: List[Dict[str, Any]] = []
    closest_id = None
    closest_z = float("inf")

    for b in world["berries"]:
        p_robot = world_xy_to_robot_xy(b.p_world, pose)  # (x_fwd, y_left)
        x_fwd = float(p_robot[0])
        if x_fwd <= 1e-6:
            continue  # behind camera

        xyz_cam = robot_xy_to_cam_xyz(p_robot)
        X, Y, Z = float(xyz_cam[0]), float(xyz_cam[1]), float(xyz_cam[2])

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy  # Y==0 -> v==cy

        # FOV / image bounds
        if not (0.0 <= u < cfg.IMG_W and 0.0 <= v < cfg.IMG_H):
            continue

        det = {
            "id": b.id,
            "class_id": b.class_id,
            "class_name": b.class_name,
            "bbox": {"center_x": float(u), "center_y": float(v)},
            "depth": {"center_meters": float(Z)},
        }
        detections.append(det)

        if Z < closest_z:
            closest_z = Z
            closest_id = b.id

    stats = {
        "closest_strawberry_id": closest_id,
        "closest_distance_meters": None if closest_id is None else round(float(closest_z), 4),
    }

    return {
        "detections": detections,
        "statistics": stats,
    }


if __name__ == "__main__":
    import argparse
    from .config import K_from_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    cfg = Sim2DConfig()
    K = K_from_cfg(cfg)
    pose0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    world = world_from_detector_json(args.json, K, pose0)
    out = simulate_detector_json(world, pose0, cfg, K)

    print("world berries:", len(world["berries"]))
    print("visible now:", len(out["detections"]))
    print("closest:", out["statistics"])
