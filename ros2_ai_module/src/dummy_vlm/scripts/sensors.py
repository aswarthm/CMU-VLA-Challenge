#!/usr/bin/env python3

from random import randrange
import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import quaternion_from_euler
from sensor_msgs.msg import PointCloud2, Image as RosImage
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
from PIL import Image


class SensorNode(Node):

    def __init__(self):
        super().__init__('sensor_node')

        self.cv_bridge = CvBridge()

        self.vehicle_x = 0.0
        self.vehicle_y = 0.0

        self.traversible = None
        self.cleaned_traversible = None
        self.cur_semantic = {}
        self.total_semantic = {}
        self.total_semantic["formatted"] = {}
        self.total_semantic["raw"] = {}
        self.img = None

        self.subscription_pose = self.create_subscription(
            Odometry,
            "/state_estimation",
            self.pose_cb,
            5
        )

        self.subscription_semantic = self.create_subscription(
            MarkerArray,
            "/object_markers",
            self.semantic_cb,
            5
        )

        self.subscription_traversible_area= self.create_subscription(
            PointCloud2,
            "/traversable_area",
            self.traversible_cb,
            5
        )

        self.subscription_image = self.create_subscription(
            RosImage,
            "/camera/image",
            self.image_cb,
            5
        )

        self.state_publisher = self.create_publisher(
            Marker,#custom
            "/get_sensor_state",
            5
        )
        
        
    def pose_cb(self, odom: Odometry):
        self.vehicle_x = odom.pose.pose.position.x
        self.vehicle_y = odom.pose.pose.position.y

    def semantic_cb(self, markers: MarkerArray):
        cur = {}
        cur["formatted"] = {}
        cur["raw"] = {}
        for marker in markers.markers:
            key = f"{marker.id}_{marker.ns}"
            cur["formatted"][key] = {
                "id": marker.id,
                "type": marker.ns,
                "pose": [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],

            }
            cur["raw"][key] = marker
            if(key not in self.total_semantic["formatted"]):
                self.total_semantic["formatted"][key] = cur["formatted"][key]
                self.total_semantic["raw"] = cur["raw"][key]
        self.cur_semantic = cur
        self.get_logger().info(str(cur["formatted"]))
    
    def image_cb(self, image: RosImage):
        cv_img = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.img = pil_img
        # pil_img.save("/home/aswarth/CMU-VLA-Challenge/ros2_ai_module/src/dummy_vlm/scripts/ok.png")
        # self.get_logger().info("done")
        # cv2.imshow("hi", cv_img)
        # cv2.waitKey(1)

    
    def traversible_cb(self, area: PointCloud2):
        self.traversible = area
        self.cleaned_traversible = point_cloud2.read_points_list(area)



def main(args=None):
    rclpy.init(args=args)

    sensorNode = SensorNode()

    rclpy.spin(sensorNode)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    sensorNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
