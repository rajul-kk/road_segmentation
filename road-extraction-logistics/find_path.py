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

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.pathfinder import AStarPathfinder


class RoadPathfinder:
    """
    Pathfinding on road segmentation masks.
    
    Converts a binary road mask into a traversable graph and uses A*
    to find the shortest path between two points on the road network.
    """
    
    def __init__(self, mask_path: str, road_threshold: int = 128):
        """
        Initialize the road pathfinder with a mask image.
        
        Args:
            mask_path: Path to the binary road mask image
            road_threshold: Pixel value threshold for road detection (default: 128)
                           Pixels >= threshold are considered road
        """
        self.mask_path = mask_path
        self.road_threshold = road_threshold
        
        # Load and process mask
        self.mask_image = Image.open(mask_path).convert('L')  # Convert to grayscale
        self.mask_array = np.array(self.mask_image)
        self.height, self.width = self.mask_array.shape
        
        # Create binary road mask (True = road, False = not road)
        self.road_mask = self.mask_array >= road_threshold
        
        # Initialize A* pathfinder
        self.pathfinder = AStarPathfinder()
        
        print(f"Loaded mask: {mask_path}")
        print(f"Dimensions: {self.width} x {self.height}")
        print(f"Road pixels: {np.sum(self.road_mask)} ({100*np.sum(self.road_mask)/(self.width*self.height):.2f}%)")
    
    def is_road(self, point: Tuple[int, int]) -> bool:
        """Check if a point is on the road."""
        x, y = point
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.road_mask[y, x]
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
        Find the nearest road pixel to a given point.
        
        Args:
            point: (x, y) coordinate
            max_search: Maximum search radius
        
        Returns:
            Nearest road pixel coordinate, or None if not found
        """
        x, y = point
        
        # If already on road, return the point
        if self.is_road(point):
            return point
        
        # Spiral outward search
        for radius in range(1, max_search + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        nx, ny = x + dx, y + dy
                        if self.is_road((nx, ny)):
                            return (nx, ny)
        
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
            print(f"Path found! Length: {len(path)} pixels")
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
        # Convert mask to RGB for visualization
        vis_image = Image.new('RGB', (self.width, self.height))
        
        # Draw road pixels in white, background in dark gray
        for y in range(self.height):
            for x in range(self.width):
                if self.road_mask[y, x]:
                    vis_image.putpixel((x, y), (255, 255, 255))
                else:
                    vis_image.putpixel((x, y), (30, 30, 30))
        
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
        
        # Create road overlay
        road_overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        for y in range(self.height):
            for x in range(self.width):
                if self.road_mask[y, x]:
                    road_overlay.putpixel((x, y), (255, 255, 0, int(255 * opacity * 0.3)))
        
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


# ============== CONFIGURATION ==============
MASK_DIR = "data/masks/predicted"        # Directory with predicted road masks
OUTPUT_DIR = "data/paths"                # Directory to save path visualizations
# Example coordinates (will be adjusted based on actual mask)
# ============================================


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
    
    # Find road pixels to use as start/end (for demo)
    # Get some road pixels from different regions
    road_pixels = np.argwhere(pathfinder.road_mask)
    
    if len(road_pixels) < 2:
        print("Not enough road pixels in the mask")
        return
    
    # Pick start from top-left region, goal from bottom-right region
    # Sort by distance from top-left corner
    distances_from_tl = road_pixels[:, 0] + road_pixels[:, 1]
    sorted_indices = np.argsort(distances_from_tl)
    
    # Start: one of the closest to top-left
    start_idx = sorted_indices[len(sorted_indices) // 10]  # 10th percentile
    start = (int(road_pixels[start_idx, 1]), int(road_pixels[start_idx, 0]))  # (x, y)
    
    # Goal: one of the closest to bottom-right
    goal_idx = sorted_indices[len(sorted_indices) * 9 // 10]  # 90th percentile
    goal = (int(road_pixels[goal_idx, 1]), int(road_pixels[goal_idx, 0]))  # (x, y)
    
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
        
        # Calculate path statistics
        path_length = len(path)
        euclidean_dist = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        
        print(f"\nPath Statistics:")
        print(f"  Path length: {path_length} pixels")
        print(f"  Euclidean distance: {euclidean_dist:.1f} pixels")
        print(f"  Path efficiency: {euclidean_dist/path_length*100:.1f}%")
        
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
