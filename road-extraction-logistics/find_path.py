"""
Road Pathfinding Script

This script finds the shortest path between two points on predicted road masks
using the A* algorithm. It converts binary road masks into a traversable graph
and visualizes the resulting path.

Usage:
    python find_path.py

You can also use this as a module:
    from find_path import RoadPathfinder
    pathfinder = RoadPathfinder("path/to/mask.png")
    path = pathfinder.find_path((x1, y1), (x2, y2))
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, List, Optional

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config as cfg
from src.pathfinder import AStarPathfinder
from src.post_process import get_skeleton
from src.path_utils import smooth_path, compute_path_distance


class RoadPathfinder:
    """
    Pathfinding on road segmentation masks.
    
    Converts a binary road mask into a traversable graph and uses A*
    to find the shortest path between two points on the road network.
    """
    
    def __init__(self, mask_path: str, road_threshold: int = 128, use_skeleton: bool = None):
        """
        Initialize the road pathfinder with a mask image.

        Args:
            mask_path:      Path to the binary road mask image.
            road_threshold: Pixel value threshold for road detection (default 128).
            use_skeleton:   If True, run A* on the 1-px-wide skeleton centerline instead of
                            the full thick mask — dramatically smaller search space.
                            Defaults to cfg.USE_SKELETON.
        """
        if use_skeleton is None:
            use_skeleton = cfg.USE_SKELETON

        self.mask_path      = mask_path
        self.road_threshold = road_threshold
        self.use_skeleton   = use_skeleton

        self.mask_image = Image.open(mask_path).convert('L')
        self.mask_array = np.array(self.mask_image)
        self.height, self.width = self.mask_array.shape

        # Thick mask used for snapping off-road points and visualization
        self.display_mask = self.mask_array >= road_threshold

        if use_skeleton:
            # Skeletonize the thick mask and use the 1-px centerline for A*
            skel_array    = get_skeleton(self.mask_array)
            self.road_mask = skel_array > 127
            nav_pixels     = int(np.sum(self.road_mask))
        else:
            self.road_mask = self.display_mask
            nav_pixels     = int(np.sum(self.road_mask))

        self.pathfinder = AStarPathfinder()

        print(f"Loaded mask: {mask_path}  ({self.width}x{self.height})")
        print(f"Navigation pixels: {nav_pixels}  (skeleton={use_skeleton})")
    
    def is_road(self, point: Tuple[int, int]) -> bool:
        """Check if a point is on the navigable road (skeleton or thick mask)."""
        x, y = point
        if 0 <= x < self.width and 0 <= y < self.height:
            return bool(self.road_mask[y, x])
        return False

    def is_any_road(self, point: Tuple[int, int]) -> bool:
        """Check against the thick display mask — used for snapping off-road points."""
        x, y = point
        if 0 <= x < self.width and 0 <= y < self.height:
            return bool(self.display_mask[y, x])
        return False
    
    def get_neighbors(self, mask: np.ndarray, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring road pixels (8-connected).
        
        Args:
            mask: The road mask (not used directly, we use self.road_mask)
            node: Current pixel coordinate (x, y)
        
        Returns:
            List of neighboring road pixel coordinates
        """
        x, y = node
        neighbors = []
        
        # 8-directional movement (including diagonals)
        directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check bounds and if it's a road pixel
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.road_mask[ny, nx]:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def get_cost(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
        """
        Get movement cost between two adjacent pixels.
        Diagonal movement costs sqrt(2), orthogonal costs 1.
        """
        dx = abs(node1[0] - node2[0])
        dy = abs(node1[1] - node2[1])
        if dx + dy == 2:  # Diagonal
            return 1.414
        return 1.0
    
    def find_nearest_road(self, point: Tuple[int, int], max_search: int = 50) -> Optional[Tuple[int, int]]:
        """
        Find the nearest navigable road pixel to a given point.

        When using skeleton mode, snaps first to the thick mask then finds the
        nearest skeleton pixel — so off-road points can still be snapped reliably.
        """
        if self.is_road(point):
            return point

        x, y = point
        # Spiral outward search against the navigable mask (road_mask)
        for radius in range(1, max_search + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        candidate = (x + dx, y + dy)
                        if self.is_road(candidate):
                            return candidate

        # If skeleton mode and no skeleton pixel found nearby, fall back to thick mask
        if self.use_skeleton:
            for radius in range(1, max_search + 1):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:
                            candidate = (x + dx, y + dy)
                            if self.is_any_road(candidate):
                                # Find nearest skeleton pixel to this thick-mask point
                                skel_pixels = np.argwhere(self.road_mask)
                                if len(skel_pixels) > 0:
                                    cx, cy = candidate
                                    dists = np.sqrt((skel_pixels[:, 1] - cx)**2 + (skel_pixels[:, 0] - cy)**2)
                                    nearest = skel_pixels[np.argmin(dists)]
                                    return (int(nearest[1]), int(nearest[0]))

        return None
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  snap_to_road: bool = True) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path between two points on the road network.
        
        Args:
            start: Starting point (x, y)
            goal: Goal point (x, y)
            snap_to_road: If True, snap start/goal to nearest road if not on road
        
        Returns:
            List of (x, y) coordinates representing the path, or None if no path exists
        """
        # Snap to nearest road if needed
        if snap_to_road:
            if not self.is_road(start):
                nearest_start = self.find_nearest_road(start)
                if nearest_start is None:
                    print(f"Could not find road near start point {start}")
                    return None
                print(f"Snapped start {start} to nearest road {nearest_start}")
                start = nearest_start
            
            if not self.is_road(goal):
                nearest_goal = self.find_nearest_road(goal)
                if nearest_goal is None:
                    print(f"Could not find road near goal point {goal}")
                    return None
                print(f"Snapped goal {goal} to nearest road {nearest_goal}")
                goal = nearest_goal
        
        # Validate points are on road
        if not self.is_road(start):
            print(f"Start point {start} is not on a road")
            return None
        if not self.is_road(goal):
            print(f"Goal point {goal} is not on a road")
            return None
        
        print(f"Finding path from {start} to {goal}...")
        
        # Run A* algorithm
        path = self.pathfinder.find_path(
            graph=self.road_mask,
            start=start,
            goal=goal,
            get_neighbors=self.get_neighbors,
            get_cost=self.get_cost
        )

        if path:
            path = smooth_path(path, epsilon=cfg.RDP_EPSILON)
            px, m = compute_path_distance(path, cfg.PIXEL_RESOLUTION_METERS)
            dist_str = f"{m:.1f} m" if m is not None else f"{px:.1f} px"
            print(f"Path found — {len(path)} waypoints, distance: {dist_str}")
        else:
            print("No path found between the two points")

        return path
    
    def visualize_path(self, path: List[Tuple[int, int]], 
                       output_path: str = None,
                       line_color: Tuple[int, int, int] = (255, 0, 0),
                       line_width: int = 3,
                       show_endpoints: bool = True) -> Image.Image:
        """
        Visualize the path on the road mask.
        
        Args:
            path: List of (x, y) coordinates
            output_path: If provided, save the visualization to this path
            line_color: RGB color for the path line (default: red)
            line_width: Width of the path line
            show_endpoints: If True, draw circles at start and end points
        
        Returns:
            PIL Image with the path drawn on it
        """
        rgb = np.where(self.display_mask[:, :, np.newaxis],
                       np.array([255, 255, 255], dtype=np.uint8),
                       np.array([30,  30,  30],  dtype=np.uint8))
        vis_image = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
        
        draw = ImageDraw.Draw(vis_image)
        
        # Draw the path
        if len(path) > 1:
            draw.line(path, fill=line_color, width=line_width)
        
        # Draw start and end points
        if show_endpoints and len(path) >= 2:
            start, end = path[0], path[-1]
            radius = 8
            # Start point (green)
            draw.ellipse([start[0]-radius, start[1]-radius, 
                         start[0]+radius, start[1]+radius], 
                        fill=(0, 255, 0), outline=(0, 200, 0))
            # End point (blue)
            draw.ellipse([end[0]-radius, end[1]-radius, 
                         end[0]+radius, end[1]+radius], 
                        fill=(0, 100, 255), outline=(0, 50, 200))
        
        if output_path:
            vis_image.save(output_path)
            print(f"Visualization saved to: {output_path}")
        
        return vis_image
    
    def visualize_path_on_satellite(self, path: List[Tuple[int, int]],
                                     satellite_path: str,
                                     output_path: str = None,
                                     line_color: Tuple[int, int, int] = (255, 0, 0),
                                     line_width: int = 3,
                                     opacity: float = 0.7) -> Image.Image:
        """
        Overlay the path on the original satellite image.
        
        Args:
            path: List of (x, y) coordinates
            satellite_path: Path to the satellite image
            output_path: If provided, save the visualization to this path
            line_color: RGB color for the path line
            line_width: Width of the path line
            opacity: Opacity of the road mask overlay (0-1)
        
        Returns:
            PIL Image with path overlaid on satellite image
        """
        # Load satellite image
        sat_image = Image.open(satellite_path).convert('RGB')
        sat_image = sat_image.resize((self.width, self.height))
        
        overlay_arr = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        overlay_arr[self.display_mask] = [255, 255, 0, int(255 * opacity * 0.3)]
        road_overlay = Image.fromarray(overlay_arr, mode='RGBA')
        
        # Composite
        sat_image = sat_image.convert('RGBA')
        result = Image.alpha_composite(sat_image, road_overlay)
        
        # Draw path
        draw = ImageDraw.Draw(result)
        if len(path) > 1:
            draw.line(path, fill=line_color + (255,), width=line_width)
        
        # Draw endpoints
        if len(path) >= 2:
            start, end = path[0], path[-1]
            radius = 8
            draw.ellipse([start[0]-radius, start[1]-radius, 
                         start[0]+radius, start[1]+radius], 
                        fill=(0, 255, 0, 255))
            draw.ellipse([end[0]-radius, end[1]-radius, 
                         end[0]+radius, end[1]+radius], 
                        fill=(0, 100, 255, 255))
        
        result = result.convert('RGB')
        
        if output_path:
            result.save(output_path)
            print(f"Satellite overlay saved to: {output_path}")
        
        return result


def pick_demo_endpoints(road_mask: np.ndarray):
    """
    Pick a start (near top-left) and goal (near bottom-right) from road pixels.
    Uses the 10th and 90th percentile of (row + col) distance as a heuristic.

    Returns:
        (start, goal) as (x, y) tuples, or (None, None) if fewer than 2 road pixels.
    """
    road_pixels = np.argwhere(road_mask)
    if len(road_pixels) < 2:
        return None, None
    distances = road_pixels[:, 0] + road_pixels[:, 1]
    indices = np.argsort(distances)
    s = indices[len(indices) // 10]
    g = indices[len(indices) * 9 // 10]
    start = (int(road_pixels[s, 1]), int(road_pixels[s, 0]))
    goal  = (int(road_pixels[g, 1]), int(road_pixels[g, 0]))
    return start, goal


# ── Configuration ─────────────────────────────────────────────────────────────
MASK_DIR   = "data/masks/predicted"
OUTPUT_DIR = "data/paths"


def main():
    """Demo: Find and visualize a path on a predicted road mask."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find a mask file to use
    if not os.path.exists(MASK_DIR):
        print(f"Mask directory not found: {MASK_DIR}")
        print("Run inference first to generate predicted masks.")
        return
    
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
    if not mask_files:
        print(f"No mask files found in {MASK_DIR}")
        return
    
    # Use first mask as demo
    mask_file = mask_files[0]
    mask_path = os.path.join(MASK_DIR, mask_file)
    
    print("=" * 60)
    print("Road Pathfinding Demo")
    print("=" * 60)
    
    # Initialize pathfinder
    pathfinder = RoadPathfinder(mask_path)
    
    start, goal = pick_demo_endpoints(pathfinder.display_mask)
    if start is None:
        print("Not enough road pixels in the mask")
        return
    
    print(f"\nStart point: {start}")
    print(f"Goal point: {goal}")
    print()
    
    # Find path
    path = pathfinder.find_path(start, goal)
    
    if path:
        # Visualize and save
        output_name = mask_file.replace('_roadmask.png', '_path.png')
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        pathfinder.visualize_path(path, output_path)
        
        # Path statistics
        px_dist, m_dist = compute_path_distance(path, cfg.PIXEL_RESOLUTION_METERS)
        euclidean_px = float(np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2))

        print(f"\nPath Statistics:")
        print(f"  Waypoints:          {len(path)}")
        print(f"  Road distance:      {px_dist:.1f} px  ({m_dist:.1f} m)" if m_dist else f"  Road distance:      {px_dist:.1f} px")
        print(f"  Euclidean distance: {euclidean_px:.1f} px")
        print(f"  Path efficiency:    {euclidean_px/px_dist*100:.1f}%")
        
        # Try to overlay on satellite if available
        sat_name = mask_file.replace('_roadmask.png', '_sat.jpg')
        sat_path = os.path.join("data/raw/test", sat_name)
        if os.path.exists(sat_path):
            overlay_name = mask_file.replace('_roadmask.png', '_path_overlay.png')
            overlay_path = os.path.join(OUTPUT_DIR, overlay_name)
            pathfinder.visualize_path_on_satellite(path, sat_path, overlay_path)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
