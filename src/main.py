import json
from pathlib import Path

import numpy as np

from src.core.vision import approx_K, goal_point_from_results
from src.core.map3d import build_voxel_map, idx_to_point
from src.core.astar3d import astar
from src.core.controller import follow_waypoints


def main():
    # входные файлы (пока фиксированные для демо)
    results_json = Path("results/strawberry_in_green_house.json")
    depth_path = Path("results/strawberry_in_green_house_depth.npy")
    masks_path = Path("results/strawberry_in_green_house_masks_combined.npy")

    data = json.loads(results_json.read_text())
    depth = np.load(depth_path)
    masks = np.load(masks_path)

    W = int(data["image_size"]["width"])
    H = int(data["image_size"]["height"])
    K = approx_K(W, H)

    start_p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    goal_p, closest_id, z_med = goal_point_from_results(data, depth, masks, K)

    print(f"closest_id={closest_id}")
    print(f"goal(camera frame)={goal_p}")

    vmap = build_voxel_map(depth, K, start_p, goal_p)

    path = astar(vmap.start_idx, vmap.goal_idx, vmap.occ, vmap.bounds_max)
    if path is None:
        raise SystemExit("NO PATH FOUND")

    waypoints = [idx_to_point(p, vmap.origin) for p in path]

    print(f"Waypoints: {len(waypoints)}")
    print(f"Final waypoint (camera frame): {waypoints[-1]}")

    follow_waypoints(waypoints)


if __name__ == "__main__":
    main()
