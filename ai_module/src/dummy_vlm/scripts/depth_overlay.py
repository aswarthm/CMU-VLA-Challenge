import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from tf_transformations import quaternion_matrix, quaternion_from_euler
import cv2
from cv_bridge import CvBridge
import numpy as np
import message_filters
import math

point_x = 0
point_y = 0

class DepthOverlayNode(Node):
    def __init__(self):
        super().__init__('depth_overlay_node')
        # self.node = node
        
        self.bridge = CvBridge()
        
        # --- Synchronize inputs ---
        self.image_msg = None
        self.pcd_msg = None
        self.odom_msg = None
        self.marker_array_msg = None
        self.cv_image = None
        self.first_run = True
        self.data_ready = False

        self.max = 0
        self.min = 999999999
        
        # --- Define the static transform from sensor to camera ---
        self.static_trans_s_c = np.array([0.0, 0.0, 0.235])
        roll, pitch, yaw = -1.5707963, 0.0, -1.5707963
        q_s_c = quaternion_from_euler(roll, pitch, yaw)
        self.static_rot_mat_s_c = quaternion_matrix(q_s_c)[:3, :3]
        image_height = 640
        image_width = 1920
        self.lookup_table = np.zeros((image_height, image_width, 3), dtype=np.float32)


        self.overlay_pub = self.create_publisher(Image, '/image_depth_overlay', 10)

        self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        # self.create_subscription(PointCloud2, '/registered_scan', self.pcd_callback, 10)
        # self.create_subscription(Odometry, '/state_estimation', self.odom_callback, 10)
        self.create_subscription(MarkerArray, '/object_markers', self.marker_callback, 10)

        image_sub = message_filters.Subscriber(self, Image, '/camera/image')
        pcd_sub = message_filters.Subscriber(self, PointCloud2, '/registered_scan')
        odom_sub = message_filters.Subscriber(self, Odometry, '/state_estimation')
        marker_sub = message_filters.Subscriber(self, MarkerArray, '/object_markers')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_sub, pcd_sub, odom_sub],
            queue_size=10,
            slop=0.3,
            allow_headerless=True
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        self.get_logger().info("Depth Overlay Node (TF-less) started. Waiting for synchronized messages.")
    
    def image_callback(self, img):
        if(self.first_run):
            self.overlay_pub.publish(img)

    def marker_callback(self, marker_array):
        self.marker_array_msg = marker_array

    # def pcd_callback(self, msg):
    #     self.pcd_msg = msg
    #     self.try_synchronized_callback()

    # def odom_callback(self, msg):
    #     self.odom_msg = msg

    # def try_synchronized_callback(self):
    #     if self.image_msg and self.pcd_msg and self.odom_msg:
    #         self.synchronized_callback(self.image_msg, self.pcd_msg, self.odom_msg)

    #         self.image_msg = None
    #         self.pcd_msg = None
    #         self.odom_msg = None


    def synchronized_callback(self, image_msg, pcd_msg, odom_msg):
        # self.get_logger().info("synchronised")
        # --- Get Robot Pose (map -> sensor) ---
        self.first_run = False
        odom_pose = odom_msg.pose.pose
        self.odom_msg = odom_msg

        pos_m_s = np.array([odom_pose.position.x, odom_pose.position.y, odom_pose.position.z])
        q_m_s = np.array([odom_pose.orientation.x, odom_pose.orientation.y, odom_pose.orientation.z, odom_pose.orientation.w])
        rot_mat_m_s = quaternion_matrix(q_m_s)[:3, :3]

        # Camera position in map frame
        camera_position_in_map = pos_m_s + np.dot(rot_mat_m_s, self.static_trans_s_c)
        rot_mat_s_m = rot_mat_m_s.T  # Inverse rotation

        # Convert image to numpy
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self.cv_image = cv_image
        h, w, _ = cv_image.shape
        overlay = np.zeros_like(cv_image, dtype=np.uint8)

        # Read all points at once
        points_map = np.array(list(point_cloud2.read_points(pcd_msg, field_names=("x", "y", "z"), skip_nans=True)))
        if points_map.shape[0] == 0:
            return  # No points to process



        # Transform all points: map -> camera frame
        points_rel_camera = points_map - camera_position_in_map
        points_camera = np.dot(points_rel_camera, rot_mat_s_m.T)  # shape (N, 3)

        # Project all points to image
        x = points_camera[:, 0]
        y = points_camera[:, 1]
        z = points_camera[:, 2]
        depth = np.sqrt(x**2 + y**2 + z**2)
        valid = depth > 0.1

        theta = np.arctan2(-y[valid], x[valid])
        phi = np.arcsin(z[valid] / depth[valid])
        u = ((w / (2 * np.pi)) * (theta + np.pi)).astype(int)
        vfov_rad = np.radians(120.0)
        v = (h * (0.5 - (phi / vfov_rad))).astype(int)

        # Create a single-channel mask (same height/width as image)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Filter valid pixel indices
        valid_mask = (0 <= u) & (u < w) & (0 <= v) & (v < h)
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        # Set valid pixels to white (255)
        mask[v_valid, u_valid] = 255

        # Optionally, convert mask to 3-channel for overlay
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        points_map_valid = points_map[valid_mask]
        self.lookup_table[v_valid, u_valid] = points_map_valid

        cv2.circle(cv_image, (point_x, point_y), radius=8, color=(0, 0, 255), thickness=-1)
        annotated_image = self.annotate_image_with_boxes(self.marker_array_msg)

        final_image = cv2.addWeighted(cv_image, 0.6, annotated_image, 0.4, 0)
        self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(final_image, "bgr8"))
        # self.get_logger().info("pub depth img")
        self.data_ready = True

    def get_3d_point(self, v, u, search_radius=10): # input y, x
        global point_x, point_y
        self.data_ready = False
        # while not self.data_ready:
        #     # self.get_logger().info("waiting for data")
        #     rclpy.spin_once(self, timeout_sec=0.01)
        point_y = v
        point_x = u
        pt = self.lookup_table[v, u]
        
        if not np.any(pt):  # If [0,0,0], try to find nearest nonzero
            h, w, _ = self.lookup_table.shape
            for r in range(1, search_radius+1):
                for dv in range(-r, r+1):
                    for du in range(-r, r+1):
                        vv, uu = v+dv, u+du
                        if 0 <= vv < h and 0 <= uu < w:
                            candidate = self.lookup_table[vv, uu]
                            if np.any(candidate):
                                return candidate
            return None  # No valid point found nearby
        return pt.tolist()

    def project_3d_box_to_2d(self, box_center_map, box_size, image_dims):
        """
        Projects a 3D bounding box from the map frame to a 2D bounding box on the
        360-degree equirectangular image, using your specific transformation logic.

        Args:
            box_center_map (np.array): The [X, Y, Z] center of the box in the map frame.
            box_size (np.array): The [width, height, depth] of the box.
            odom_pose (Pose): The robot's current odometry pose message.
            image_dims (tuple): The (width, height) of the camera image.

        Returns:
            A tuple (u_min, v_min, u_max, v_max) for the 2D bounding box, or None.
        """
        odom_pose=self.odom_msg.pose.pose
        image_width, image_height = image_dims

        # --- 1. Get Transformation from Odometry (from your code) ---
        pos_m_s = np.array([odom_pose.position.x, odom_pose.position.y, odom_pose.position.z])
        q_m_s = np.array([odom_pose.orientation.x, odom_pose.orientation.y, odom_pose.orientation.z, odom_pose.orientation.w])
        rot_mat_m_s = quaternion_matrix(q_m_s)[:3, :3]
        rot_mat_s_m = rot_mat_m_s.T
        
        # NOTE: This assumes your static transform is zero for this calculation,
        # as per your provided `synchronized_callback` logic.
        camera_position_in_map = pos_m_s 

        # --- 2. Define the 8 corners of the 3D box ---
        w, h, d = box_size / 2
        corners_local = np.array([
            [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
            [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]
        ])
        # Assume box is axis-aligned in the map frame for simplicity
        corners_map = corners_local + box_center_map

        # --- 3. Project all 8 corners ---
        projected_points = []
        for corner in corners_map:
            # Transform point from map to the simple, non-rotated camera frame
            p_relative_to_camera = corner - camera_position_in_map
            p_camera_np = np.dot(rot_mat_s_m, p_relative_to_camera)
            x, y, z = p_camera_np
            
            depth = np.linalg.norm(p_camera_np)
            if depth < 0.1: continue

            # Project 3D point to 2D pixel
            theta = math.atan2(-y, x)
            phi = math.asin(z / depth)
            
            u = int((image_width / (2 * math.pi)) * (theta + math.pi))
            vfov_rad = math.radians(120.0)
            v = int(image_height * (0.5 - (phi / vfov_rad)))

            if 0 <= v < image_height: # Check vertical bounds
                projected_points.append([u, v])

        if not projected_points:
            return None

        # --- 4. Find the min/max coordinates to define the 2D box ---
        # This correctly handles the 360-degree wrap-around case for the u-coordinate
        points_np = np.array(projected_points)
        u_coords = points_np[:, 0]
        v_coords = points_np[:, 1]

        # Handle the wrap-around for horizontal coordinate 'u'
        if np.max(u_coords) - np.min(u_coords) > image_width / 2:
            u_coords[u_coords < image_width / 2] += image_width

        u_min, v_min = np.min(u_coords, axis=0) % image_width, np.min(v_coords, axis=0)
        u_max, v_max = np.max(u_coords, axis=0) % image_width, np.max(v_coords, axis=0)

        if u_min > u_max:
            # This condition is a clear sign of a wrap-around.
            # We will return a list containing two bounding boxes.
            
            # Box 1: from the left edge to u_max
            box1 = (0, v_min, u_max, v_max)
            
            # Box 2: from u_min to the right edge
            box2 = (u_min, v_min, image_width - 1, v_max)
            
            return None
        # else:
        #     # If no wrap-around, return the single box in a list for consistency
        #     return [(u_min, v_min, u_max, v_max)]
        # area = (u_max-u_min) *  (v_max-v_min)
        # self.max = max(self.max, area)
        # self.min = min(self.min, area)
        # self.get_logger().info(f"{self.min} {self.max}")
        # if(area <= self.max*0.005):
        #     return None

        return int(u_min), int(v_min), int(u_max), int(v_max)
    
    # Add this new function inside your DepthOverlayNode class

    def annotate_image_with_boxes(self, marker_array_msg):
        if(self.cv_image is None):
            return
        if(marker_array_msg is None):
            return self.cv_image
        cv_image = self.cv_image
        h, w, _ = cv_image.shape
        # Iterate through every object marker provided
        for marker in marker_array_msg.markers:
            # Extract 3D box properties from the marker
            box_center_map = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
            box_size = np.array([marker.scale.x, marker.scale.y, marker.scale.z])
            
            # Project the 3D box to a 2D box using your existing helper function
            bbox_2d = self.project_3d_box_to_2d(box_center_map, box_size, (w, h))
            if bbox_2d:
                u_min, v_min, u_max, v_max = bbox_2d
                
                # --- Draw the Bounding Box (Green) ---
                cv2.rectangle(cv_image, (u_min, v_min), (u_max, v_max), (0, 255, 0), 2)
                
                # --- Draw the ID Label (White Text with Black Outline) ---
                label = f"{marker.id}_{marker.ns}"
                if any(x in label for x in ["floor", "ceiling", "wall", "unknown"]):
                    continue
                font_scale = 1
                font_thickness = 2
                
                # Position text above the box, but move it below if it's off-screen
                text_pos = (u_min, v_min - 10 if v_min > 20 else v_max + 20)
                
                # Draw a black outline for better readability
                cv2.putText(cv_image, label, text_pos, cv2.FONT_HERSHEY_DUPLEX, 
                            font_scale, (0, 0, 0), font_thickness + 4, cv2.LINE_AA)
                # Draw the white text
                cv2.putText(cv_image, label, text_pos, cv2.FONT_HERSHEY_DUPLEX, 
                            font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            # if bbox_2d_list:
            #     for bbox_2d in bbox_2d_list:
            
                
        return cv_image



def main(args=None):
    rclpy.init(args=args)

    depthOverlayNode = DepthOverlayNode()

    rclpy.spin(depthOverlayNode)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    depthOverlayNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




    '''
    
    
        def synchronized_callback(self, image_msg, pcd_msg, odom_msg):
        # --- Get Robot Pose (map -> sensor) ---
        odom_pose = odom_msg.pose.pose
        pos_m_s = np.array([odom_pose.position.x, odom_pose.position.y, odom_pose.position.z])
        q_m_s = np.array([odom_pose.orientation.x, odom_pose.orientation.y, odom_pose.orientation.z, odom_pose.orientation.w])
        rot_mat_m_s = quaternion_matrix(q_m_s)[:3, :3]

        # --- THIS IS THE FIX ---
        # We only care about the camera's POSITION, not its physical rotation,
        # because the image has already been remapped to be upright.
        
        # 1. Calculate the camera's position in the map frame
        camera_position_in_map = pos_m_s + np.dot(rot_mat_m_s, self.static_trans_s_c)

        # 2. To transform a point from the map to the camera's view, we need the inverse:
        #    - Rotate it by the inverse of the robot's rotation
        #    - Translate it by the inverse of the camera's position
        rot_mat_s_m = rot_mat_m_s.T
        
        # --- The rest of the logic ---
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        h, w, _ = cv_image.shape
        overlay = np.zeros_like(cv_image, dtype=np.uint8)

        points_generator = point_cloud2.read_points(pcd_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        if points_map.shape[0] == 0:
            return # No points to process
        for p_map in points_generator:
            # Manually transform point from map to a simple, non-rotated camera frame
            p_relative_to_camera = p_map - camera_position_in_map
            p_camera_np = np.dot(rot_mat_s_m, p_relative_to_camera)
            x, y, z = p_camera_np
            
            # Project 3D point to 2D pixel
            depth = math.sqrt(x**2 + y**2 + z**2)
            if depth < 0.1: continue

            theta = math.atan2(-y, x)
            phi = math.asin(z / depth)
            
            u = int((w / (2 * math.pi)) * (theta + math.pi))
            # Convert 120 degrees to radians for the calculation
            vfov_rad = math.radians(120.0)

            # This correctly maps the vertical angle to the pixel space using the 120-degree VFOV
            v = int(h * (0.5 - (phi / vfov_rad)))

            if 0 <= u < w and 0 <= v < h:
                # Colorize and Draw
                color = self.depth_to_color(depth)
                cv2.circle(overlay, (u, v), radius=2, color=color, thickness=-1)

        final_image = cv2.addWeighted(cv_image, 0.6, overlay, 0.4, 0)
        self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(final_image, "bgr8"))
        self.get_logger().info("pubbbbbbb")
    '''