"""
A* Pathfinding Algorithm Implementation

This module implements the A* algorithm for finding optimal paths on road networks.
A* is an informed search algorithm that uses a heuristic to efficiently find the shortest path.
"""

import heapq
from typing import List, Tuple, Optional, Callable
import numpy as np


class AStarPathfinder:
    """
    A* pathfinding algorithm implementation.
    
    Uses a priority queue (heap) to explore nodes, prioritizing those with lower
    estimated total cost (g(n) + h(n)), where:
    - g(n): actual cost from start to current node
    - h(n): heuristic estimate from current node to goal
    """
    
    def __init__(self, heuristic: Optional[Callable] = None):
        """
        Initialize A* pathfinder.
        
        Args:
            heuristic: Optional heuristic function. If None, uses Euclidean distance.
                       Should take (node1, node2) and return estimated distance.
        """
        self.heuristic = heuristic or self._euclidean_distance
    
    def _euclidean_distance(self, node1: Tuple[float, float], node2: Tuple[float, float]) -> float:
        """
        Euclidean distance heuristic (admissible for 2D coordinates).
        
        Args:
            node1: (x, y) coordinates of first node
            node2: (x, y) coordinates of second node
        
        Returns:
            Euclidean distance between nodes
        """
        return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
    
    def find_path(self, 
                  graph: dict, 
                  start: Tuple[float, float], 
                  goal: Tuple[float, float],
                  get_neighbors: Callable = None,
                  get_cost: Callable = None) -> Optional[List[Tuple[float, float]]]:
        """
        Find shortest path from start to goal using A* algorithm.
        
        Args:
            graph: Graph representation. Can be:
                   - NetworkX graph object
                   - Dictionary of {node: [neighbors]}
                   - Any graph structure (requires get_neighbors function)
            start: Starting node coordinates (x, y)
            goal: Goal node coordinates (x, y)
            get_neighbors: Function to get neighbors of a node.
                          If None, assumes graph is dict or NetworkX graph.
            get_cost: Function to get edge cost between two nodes.
                     If None, uses Euclidean distance.
        
        Returns:
            List of nodes representing the path from start to goal, or None if no path exists
        """
        # Initialize data structures
        # Use counter for tie-breaking when f_scores are equal
        counter = 0
        open_set = []  # Priority queue: (f_score, counter, node)
        heapq.heappush(open_set, (0, counter, start))
        
        came_from = {}  # Track path reconstruction
        g_score = {start: 0}  # Actual cost from start to node
        f_score = {start: self.heuristic(start, goal)}  # Estimated total cost
        
        visited = set()
        
        # Helper functions for different graph types
        if get_neighbors is None:
            get_neighbors = self._get_neighbors_default
        if get_cost is None:
            get_cost = self._get_cost_default
        
        while open_set:
            # Get node with lowest f_score
            current_f, _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check if we reached the goal
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            neighbors = get_neighbors(graph, current)
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + get_cost(current, neighbor)
                
                # If this path to neighbor is better, record it
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # No path found
        return None
    
    def _get_neighbors_default(self, graph, node):
        """Default neighbor getter for dict or NetworkX graphs."""
        if isinstance(graph, dict):
            return graph.get(node, [])
        else:
            # Assume NetworkX graph
            try:
                return list(graph.neighbors(node))
            except AttributeError:
                return []
    
    def _get_cost_default(self, node1: Tuple[float, float], node2: Tuple[float, float]) -> float:
        """Default cost function: Euclidean distance."""
        return self._euclidean_distance(node1, node2)
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Reconstruct path from came_from dictionary.
        
        Args:
            came_from: Dictionary mapping node to its predecessor
            current: Current node (goal node)
        
        Returns:
            List of nodes from start to goal
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get path from start to goal


def astar_path(graph, start: Tuple[float, float], goal: Tuple[float, float], 
               heuristic: Optional[Callable] = None) -> Optional[List[Tuple[float, float]]]:
    """
    Convenience function for A* pathfinding.
    
    Args:
        graph: Graph representation (dict, NetworkX graph, etc.)
        start: Starting node coordinates
        goal: Goal node coordinates
        heuristic: Optional heuristic function
    
    Returns:
        Path from start to goal, or None if no path exists
    
    Example:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_edge((0, 0), (1, 1), weight=1.0)
        >>> path = astar_path(G, (0, 0), (1, 1))
        >>> print(path)
        [(0, 0), (1, 1)]
    """
    pathfinder = AStarPathfinder(heuristic=heuristic)
    return pathfinder.find_path(graph, start, goal)


def astar_path_with_networkx(graph, start: Tuple[float, float], goal: Tuple[float, float],
                             weight: str = 'weight') -> Optional[List[Tuple[float, float]]]:
    """
    A* pathfinding specifically for NetworkX graphs with edge weights.
    
    Args:
        graph: NetworkX graph object
        start: Starting node
        goal: Goal node
        weight: Edge attribute to use as weight (default: 'weight')
    
    Returns:
        Path from start to goal, or None if no path exists
    """
    def get_neighbors(g, node):
        return list(g.neighbors(node))
    
    def get_cost(node1, node2):
        try:
            return graph[node1][node2].get(weight, 1.0)
        except:
            return 1.0
    
    pathfinder = AStarPathfinder()
    return pathfinder.find_path(graph, start, goal, 
                               get_neighbors=get_neighbors, 
                               get_cost=get_cost)

