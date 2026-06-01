"""
Batch Road Pathfinding Script

This script runs the pathfinding algorithm on multiple images and outputs
the results with visualizations.

Usage:
    python run_batch_pathfinding.py [num_images]
    
    num_images: Number of images to process (default: 10)
"""

import os
import sys
import numpy as np
from PIL import Image

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config as cfg
from find_path import RoadPathfinder, pick_demo_endpoints
from src.path_utils import compute_path_distance

# ============== CONFIGURATION ==============
MASK_DIR = "data/masks/predicted"        # Directory with predicted road masks
OUTPUT_DIR = "data/paths"                # Directory to save path visualizations
DEFAULT_NUM_IMAGES = 10
# ============================================


def process_single_mask(mask_path: str, output_dir: str, mask_name: str) -> dict:
    """
    Process a single mask file and return statistics.
    
    Args:
        mask_path: Full path to the mask file
        output_dir: Directory to save output
        mask_name: Name of the mask file (for output naming)
    
    Returns:
        Dictionary with path statistics or error info
    """
    result = {
        'mask_name': mask_name,
        'success': False,
        'path_length': 0,
        'euclidean_distance': 0,
        'efficiency': 0,
        'error': None
    }
    
    try:
        # Initialize pathfinder
        pathfinder = RoadPathfinder(mask_path)
        
        start, goal = pick_demo_endpoints(pathfinder.display_mask)
        if start is None:
            result['error'] = "Not enough road pixels"
            return result

        # Find path (smoothing applied inside find_path via path_utils)
        path = pathfinder.find_path(start, goal)

        if path:
            output_name = mask_name.replace('_roadmask.png', '_path.png')
            output_path = os.path.join(output_dir, output_name)
            pathfinder.visualize_path(path, output_path)

            px_dist, m_dist       = compute_path_distance(path, cfg.PIXEL_RESOLUTION_METERS)
            euclidean_px          = float(np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2))
            efficiency            = euclidean_px / px_dist * 100 if px_dist > 0 else 0

            result['success']            = True
            result['path_length']        = len(path)
            result['road_distance_px']   = round(px_dist, 1)
            result['road_distance_m']    = round(m_dist, 1) if m_dist is not None else None
            result['euclidean_distance'] = round(euclidean_px, 1)
            result['efficiency']         = round(efficiency, 1)
            result['output_file']        = output_path
        else:
            result['error'] = "No path found between points"
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    """Run pathfinding on multiple mask images."""
    
    # Get number of images from command line or use default
    num_images = DEFAULT_NUM_IMAGES
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of images: {sys.argv[1]}, using default: {DEFAULT_NUM_IMAGES}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find mask files
    if not os.path.exists(MASK_DIR):
        print(f"Mask directory not found: {MASK_DIR}")
        print("Run inference first to generate predicted masks.")
        return
    
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    if not mask_files:
        print(f"No mask files found in {MASK_DIR}")
        return
    
    # Limit to requested number
    mask_files = mask_files[:num_images]
    
    print("=" * 70)
    print(f"Batch Road Pathfinding - Processing {len(mask_files)} images")
    print("=" * 70)
    print()
    
    results = []
    successful = 0
    failed = 0
    
    for i, mask_file in enumerate(mask_files, 1):
        mask_path = os.path.join(MASK_DIR, mask_file)
        
        print(f"\n[{i}/{len(mask_files)}] Processing: {mask_file}")
        print("-" * 50)
        
        result = process_single_mask(mask_path, OUTPUT_DIR, mask_file)
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"  ✓ Path found!")
            print(f"    Waypoints:          {result['path_length']}")
            dist_str = (f"{result['road_distance_px']} px  ({result['road_distance_m']} m)"
                        if result['road_distance_m'] is not None
                        else f"{result['road_distance_px']} px")
            print(f"    Road distance:      {dist_str}")
            print(f"    Euclidean distance: {result['euclidean_distance']} px")
            print(f"    Path efficiency:    {result['efficiency']}%")
            print(f"    Output:             {result['output_file']}")
        else:
            failed += 1
            print(f"  ✗ Failed: {result['error']}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {len(mask_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        ok = [r for r in results if r['success']]
        avg_dist_px  = sum(r['road_distance_px'] for r in ok) / successful
        avg_dist_m   = sum(r['road_distance_m']  for r in ok if r['road_distance_m'] is not None)
        avg_eff      = sum(r['efficiency'] for r in ok) / successful
        dist_summary = (f"{avg_dist_px:.1f} px  ({avg_dist_m/successful:.1f} m avg)"
                        if any(r['road_distance_m'] for r in ok) else f"{avg_dist_px:.1f} px")
        print(f"\nAverage road distance:  {dist_summary}")
        print(f"Average path efficiency: {avg_eff:.1f}%")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()
