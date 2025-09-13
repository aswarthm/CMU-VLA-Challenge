import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.spatial import cKDTree
import numpy as np
from typing import Optional

class GoalSnapperNode(Node):
    def __init__(self):
        super().__init__('goal_snapper_node')
        
        self.traversable_points_3d = None
        self.kdtree = None
        
        self.traversable_sub = self.create_subscription(
            PointCloud2,
            '/traversable_area',
            self.traversable_callback,
            5
        )
        
        self.get_logger().info("Goal Snapper Node is ready. Waiting for traversable area...")

    def traversable_callback(self, msg: PointCloud2):
        """Receives the traversable area and builds the k-d tree."""
        if self.kdtree is not None:
            return

        points_generator = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.traversable_points_3d = np.array(list(points_generator))
        
        if len(self.traversable_points_3d) > 0:        
            self.kdtree = cKDTree(self.traversable_points_3d)
            self.get_logger().info(f"k-d tree built successfully with {len(self.traversable_points_3d)} points.")
            self.destroy_subscription(self.traversable_sub) # We only need the map once.

    def find_closest_traversable_point(self, target_point_3d: list, k_neighbors: int = 10, pull_in_factor: float = 0.3) -> Optional[list]:
        if self.kdtree is None:
            self.get_logger().warn("k-d tree is not built yet. Cannot find closest point.")
            return None

        try:
            distances, indices = self.kdtree.query(target_point_3d, k=k_neighbors)
            closest_point = self.traversable_points_3d[indices[0]]
            neighbor_points = self.traversable_points_3d[indices]
            centroid = np.mean(neighbor_points, axis=0)
            direction_vector = centroid - closest_point
            norm = np.linalg.norm(direction_vector)
            if norm > 0:
                direction_vector /= norm
            safer_point = closest_point + direction_vector * pull_in_factor
            return safer_point.tolist()
        except Exception as e:
            self.get_logger().error(f"Error during k-d tree query: {e}")
            return None