"""
Main StrawberryDetector class combining depth estimation and segmentation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from . import config
from .depth import DepthEstimator
from .segmentation import StrawberrySegmenter
from .utils import load_image, visualize_results, save_visualization


class StrawberryDetector:
    """
    Combined strawberry detection with depth estimation.
    
    Detects strawberries, segments them, and estimates their distance.
    """
    
    def __init__(self, 
                 yolo_weights: Optional[Path] = None,
                 depth_weights: Optional[Path] = None,
                 device: Optional[str] = None):
        """
        Initialize detector.
        
        Args:
            yolo_weights: Path to YOLO weights (downloads if None)
            depth_weights: Path to depth weights (downloads if None)
            device: Device to use ('cuda' or 'cpu')
        """
        print("🍓 Initializing StrawberryDetector...")
        
        self.device = device
        
        # Initialize segmenter
        self.segmenter = StrawberrySegmenter(
            weights_path=yolo_weights,
            device=device
        )
        
        # Initialize depth estimator
        self.depth_estimator = DepthEstimator(device=device)
        
        print("✅ StrawberryDetector ready")
    
    def detect(self, 
               image: Union[str, Path, np.ndarray],
               conf_threshold: float = 0.25,
               return_numpy: bool = False) -> dict:
        """
        Detect strawberries with depth estimation.
        
        Args:
            image: Image path, URL, or numpy array
            conf_threshold: Confidence threshold for detection
            return_numpy: If True, include numpy arrays in output
        
        Returns:
            Dict with detections, depth info, and statistics
        """
        # Record image path
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            image_path = "<numpy array>"
        
        # Load image
        img = load_image(image)
        h, w = img.shape[:2]
        
        # Run depth estimation
        depth_map = self.depth_estimator.estimate(img)
        
        # Resize depth map to match image if needed
        if depth_map.shape != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Run segmentation
        detections = self.segmenter.segment(img, conf_threshold=conf_threshold)
        
        # Add depth info to each detection
        for det in detections:
            self._add_depth_info(det, depth_map)
        
        # Compute statistics
        stats = self._compute_statistics(detections)
        
        # Build result
        result = {
            "image_path": image_path,
            "image_size": {"width": w, "height": h},
            "depth_map_shape": list(depth_map.shape),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "detections_count": len(detections),
            "detections": detections,
            "statistics": stats,
        }
        
        # Add numpy arrays if requested
        if return_numpy:
            result["_numpy"] = {
                "image": img,
                "depth_map": depth_map,
            }
        
        # Remove binary masks from output (not JSON serializable)
        for det in result["detections"]:
            if "mask" in det and "binary_mask" in det["mask"]:
                del det["mask"]["binary_mask"]
        
        return result
    
    def _add_depth_info(self, detection: dict, depth_map: np.ndarray):
        """Add depth statistics to a detection using its mask."""
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        cx, cy = bbox["center_x"], bbox["center_y"]
        
        # Get mask or use bbox
        if "mask" in detection and "binary_mask" in detection["mask"]:
            mask = detection["mask"]["binary_mask"]
            # Resize mask to depth map size if needed
            if mask.shape != depth_map.shape:
                mask = cv2.resize(
                    mask.astype(np.float32), 
                    (depth_map.shape[1], depth_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            # Convert to boolean mask
            bool_mask = mask > 0.5
            masked_depth = depth_map[bool_mask]
        else:
            # Use bounding box region
            masked_depth = depth_map[y1:y2, x1:x2].flatten()
        
        if len(masked_depth) > 0:
            depth_info = {
                "mean_meters": round(float(np.mean(masked_depth)), 4),
                "median_meters": round(float(np.median(masked_depth)), 4),
                "min_meters": round(float(np.min(masked_depth)), 4),
                "max_meters": round(float(np.max(masked_depth)), 4),
                "std_meters": round(float(np.std(masked_depth)), 4),
            }
            
            # Add center point depth
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                depth_info["center_meters"] = round(float(depth_map[cy, cx]), 4)
            
            # Add cm values for convenience
            depth_info["mean_cm"] = round(depth_info["mean_meters"] * 100, 2)
            depth_info["median_cm"] = round(depth_info["median_meters"] * 100, 2)
            depth_info["min_cm"] = round(depth_info["min_meters"] * 100, 2)
            depth_info["max_cm"] = round(depth_info["max_meters"] * 100, 2)
        else:
            depth_info = {
                "mean_meters": None,
                "error": "No depth data available"
            }
        
        detection["depth"] = depth_info
    
    def _compute_statistics(self, detections: list) -> dict:
        """Compute summary statistics for all detections."""
        stats = {
            "total_ripe": 0,
            "total_unripe": 0,
            "total_other": 0,
        }
        
        closest_id = None
        closest_dist = float("inf")
        furthest_id = None
        furthest_dist = 0
        
        for det in detections:
            # Count by class (using actual YOLO model class names)
            class_name = det.get("class_name", "other").lower()
            # NOTE: "ripe" is a substring of "unripe", so check "unripe" first
            if "unripe" in class_name:
                stats["total_unripe"] += 1
            elif "ripe" in class_name:
                stats["total_ripe"] += 1
            elif class_name == "strawberry" or "strawberry" in class_name:
                stats["total_unripe"] += 1
            else:
                stats["total_other"] += 1
            
            # Track closest/furthest
            if "depth" in det and det["depth"].get("mean_meters"):
                dist = det["depth"]["mean_meters"]
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = det["id"]
                if dist > furthest_dist:
                    furthest_dist = dist
                    furthest_id = det["id"]
        
        if closest_id is not None:
            stats["closest_strawberry_id"] = closest_id
            stats["closest_distance_meters"] = round(closest_dist, 4)
            stats["closest_distance_cm"] = round(closest_dist * 100, 2)
        
        if furthest_id is not None:
            stats["furthest_strawberry_id"] = furthest_id
            stats["furthest_distance_meters"] = round(furthest_dist, 4)
            stats["furthest_distance_cm"] = round(furthest_dist * 100, 2)
        
        return stats
    
    def detect_folder(self,
                      folder_path: Union[str, Path],
                      output_folder: Optional[Union[str, Path]] = None,
                      conf_threshold: float = 0.25,
                      visualize: bool = False,
                      save_full: bool = False,
                      extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> list:
        """
        Detect strawberries in all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            output_folder: Path to save results (optional)
            conf_threshold: Confidence threshold
            visualize: If True, save visualization images
            save_full: If True, save masks and depth maps as .npy files
            extensions: Image file extensions to process
        
        Returns:
            List of detection results for all images
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Create output folder if specified
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        image_files = sorted(set(image_files))
        print(f"📁 Found {len(image_files)} images in {folder_path}")
        
        results = []
        
        for i, img_path in enumerate(image_files):
            print(f"\n🖼️ Processing [{i+1}/{len(image_files)}]: {img_path.name}")
            
            try:
                if save_full and output_folder:
                    # Save full output including masks and depth
                    result = self.detect_and_save_full(
                        img_path,
                        output_dir=output_folder,
                        conf_threshold=conf_threshold,
                        save_visualization=visualize
                    )
                elif visualize and output_folder:
                    vis_path = output_folder / f"{img_path.stem}_vis.jpg"
                    result, _ = self.detect_and_visualize(
                        img_path,
                        output_image_path=vis_path,
                        conf_threshold=conf_threshold
                    )
                    # Save JSON
                    json_path = output_folder / f"{img_path.stem}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    result = self.detect(img_path, conf_threshold=conf_threshold)
                    if output_folder:
                        json_path = output_folder / f"{img_path.stem}.json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                
                results.append(result)
                print(f"   Found {result['detections_count']} strawberries")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({"image_path": str(img_path), "error": str(e)})
        
        print(f"\n✅ Processed {len(results)} images")
        return results
    
    def detect_and_visualize(self, 
                              image: Union[str, Path, np.ndarray],
                              output_image_path: Optional[Union[str, Path]] = None,
                              conf_threshold: float = 0.25) -> tuple:
        """
        Detect and optionally save visualization.
        
        Args:
            image: Input image
            output_image_path: Path to save visualization
            conf_threshold: Confidence threshold
        
        Returns:
            Tuple of (result_dict, visualization_image)
        """
        result = self.detect(image, conf_threshold=conf_threshold, return_numpy=True)
        
        # Create visualization
        img = result["_numpy"]["image"]
        vis = visualize_results(img, result["detections"])
        
        # Save if path provided
        if output_image_path:
            save_visualization(vis, output_image_path)
        
        # Remove numpy data from result
        del result["_numpy"]
        
        return result, vis
    
    def detect_and_save_full(self,
                             image: Union[str, Path, np.ndarray],
                             output_dir: Union[str, Path],
                             name_prefix: Optional[str] = None,
                             conf_threshold: float = 0.25,
                             save_visualization: bool = True) -> dict:
        """
        Detect and save full results including masks and depth maps.
        
        Saves:
            - {prefix}.json: Detection results
            - {prefix}_depth.npy: Full depth map
            - {prefix}_vis.jpg: Visualization (if enabled)
            - {prefix}_mask_{id}.npy: Individual mask for each detection
            - {prefix}_masks_combined.npy: All masks combined
        
        Args:
            image: Input image path, URL, or numpy array
            output_dir: Directory to save outputs
            name_prefix: Prefix for output files (uses image filename if None)
            conf_threshold: Confidence threshold
            save_visualization: Whether to save visualization image
        
        Returns:
            Detection result dict with file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine prefix
        if name_prefix is None:
            if isinstance(image, (str, Path)):
                name_prefix = Path(image).stem
            else:
                name_prefix = "detection"
        
        # Load image
        img = load_image(image)
        h, w = img.shape[:2]
        
        # Run depth estimation
        depth_map = self.depth_estimator.estimate(img)
        if depth_map.shape != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Run segmentation (get raw detections with masks)
        detections = self.segmenter.segment(img, conf_threshold=conf_threshold)
        
        # Save depth map
        depth_path = output_dir / f"{name_prefix}_depth.npy"
        np.save(depth_path, depth_map)
        
        # Process masks and add depth info
        masks_combined = np.zeros((h, w), dtype=np.uint8)
        mask_paths = []
        
        for det in detections:
            det_id = det["id"]
            self._add_depth_info(det, depth_map)
            
            # Save individual mask if available
            if "mask" in det and "binary_mask" in det["mask"]:
                mask = det["mask"]["binary_mask"]
                # Resize mask to original size if needed
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h), 
                                     interpolation=cv2.INTER_NEAREST)
                
                mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                mask_path = output_dir / f"{name_prefix}_mask_{det_id}.npy"
                np.save(mask_path, mask_uint8)
                mask_paths.append(str(mask_path))
                
                # Add to combined mask (with unique ID)
                masks_combined[mask > 0.5] = det_id + 1
        
        # Save combined masks
        combined_mask_path = output_dir / f"{name_prefix}_masks_combined.npy"
        np.save(combined_mask_path, masks_combined)
        
        # Remove binary masks from detections (not JSON serializable)
        for det in detections:
            if "mask" in det and "binary_mask" in det["mask"]:
                del det["mask"]["binary_mask"]
        
        # Compute statistics
        stats = self._compute_statistics(detections)
        
        # Build result
        result = {
            "image_path": str(image) if isinstance(image, (str, Path)) else "<numpy array>",
            "image_size": {"width": w, "height": h},
            "depth_map_shape": list(depth_map.shape),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "detections_count": len(detections),
            "detections": detections,
            "statistics": stats,
            "output_files": {
                "depth_map": str(depth_path),
                "masks_combined": str(combined_mask_path),
                "individual_masks": mask_paths,
            }
        }
        
        # Save visualization
        if save_visualization:
            vis = visualize_results(img, detections, depth_map)
            vis_path = output_dir / f"{name_prefix}_vis.jpg"
            cv2.imwrite(str(vis_path), vis)
            result["output_files"]["visualization"] = str(vis_path)
        
        # Save JSON
        json_path = output_dir / f"{name_prefix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        result["output_files"]["json"] = str(json_path)
        
        return result


# Convenience function for quick detection
_detector_instance = None

def detect(image: Union[str, Path, np.ndarray], 
           conf_threshold: float = 0.25,
           **kwargs) -> dict:
    """
    Quick detection function (singleton detector).
    
    Args:
        image: Image path, URL, or numpy array
        conf_threshold: Confidence threshold
        **kwargs: Additional args for StrawberryDetector
    
    Returns:
        Detection result dict
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = StrawberryDetector(**kwargs)
    
    return _detector_instance.detect(image, conf_threshold=conf_threshold)
