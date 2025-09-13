#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, Image as RosImage
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
from PIL import Image
import base64
import io

class SensorNode(Node):

    def __init__(self):
        super().__init__('sensor_node')

        self.cv_bridge = CvBridge()

        self.vehicle_x = 0.0
        self.vehicle_y = 0.0

        self.traversible = None
        self.cleaned_traversible = None
        self.cur_semantic = {}
        self.cur_semantic["formatted"] = {}
        self.cur_semantic["raw"] = {}
        self.total_semantic = {}
        self.total_semantic["formatted"] = {}
        self.total_semantic["raw"] = {}
        self.img = None
        self.cv_img = None
        self.twist = None
        self.annotated_img = None
        self.annotated_cv_img = None

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
            "/image_depth_overlay",
            self.annotated_image_cb,
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
        self.get_logger().info("Started sensor Node")
        # self.genai_client = genai.Client()
        
        
    def pose_cb(self, odom: Odometry):
        self.vehicle_x = odom.pose.pose.position.x
        self.vehicle_y = odom.pose.pose.position.y
        self.twist = odom.twist.twist
        # self.get_logger().info("pose cb")

    def semantic_cb(self, markers: MarkerArray):
        cur = {}
        cur["formatted"] = {}
        cur["raw"] = {}
        for marker in markers.markers:
            key = f"{marker.id}_{marker.ns}"
            cur["formatted"][key] = {
                "id": str(marker.id),
                "type": marker.ns,
                # "pose": [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],

            }
            cur["raw"][key] = marker
            if(key not in self.total_semantic["formatted"]):
                self.total_semantic["formatted"][key] = cur["formatted"][key]
                self.total_semantic["raw"][key] = cur["raw"][key]
        self.cur_semantic = cur
        # self.get_logger().info(str(list(self.cur_semantic["formatted"].keys())))
    
    def annotated_image_cb(self, image: RosImage):
        cv_img = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        

        # Encode to base64
        img_b64 = self.get_base64_img(cv_img)

        # Now img_b64 is ready to send to Gemini GenAI
        self.annotated_img = img_b64
        self.annotated_cv_img = cv_img

    def image_cb(self, image: RosImage):
        cv_img = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        # Encode to base64
        img_b64 = self.get_base64_img(cv_img)

        # Now img_b64 is ready to send to Gemini GenAI
        self.img = img_b64
        self.cv_img = cv_img
        # response = self.genai_client.models.generate_content(
        #     model="gemini-2.5-pro", 
        #     contents=[self.img, "Explain how AI works in a few words"]
        # )
        # self.get_logger().info(f"{response}")
        # cv2.imshow("hi", cv_img)
        # cv2.waitKey(0)
        # pil_img.save("/home/aswarth/CMU-VLA-Challenge/ros2_ai_module/src/dummy_vlm/scripts/ok.png")
        # self.get_logger().info("done")
        # cv2.imshow("hi", cv_img)
        # cv2.waitKey(1)
    
    def get_base64_img(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Convert PIL image to bytes in PNG format
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        img_bytes = buf.getvalue()

        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_b64

    
    def traversible_cb(self, area: PointCloud2):
        self.traversible = area
        self.cleaned_traversible = point_cloud2.read_points_list(area)
    

    def get_sensor_state(self):
        # Check all required data exists
        if (
            self.img is None
            # or "formatted" not in self.cur_semantic or not self.cur_semantic["formatted"]
            # or "formatted" not in self.total_semantic or not self.total_semantic["formatted"]
        ):
            # self.get_logger().warn("Sensor state incomplete: missing image or semantic data")
            # self.get_logger().info(f"{self.img}")
            return None
        
        data = {
            "image": self.img,
            "cv_img": self.cv_img,
            "vehicle_x": self.vehicle_x,
            "vehicle_y": self.vehicle_y,
            "velocity": self.twist,
            "annotated_image": self.annotated_img,
            "cur_semantic": list(self.cur_semantic["formatted"].keys()),
            "total_semantic": list(self.total_semantic["formatted"].keys())
        }
        return data

    def get_full_id(self, id):
        id = str(id)
        with open("semantic.txt", "w") as f:
            json.dump(self.total_semantic["formatted"], f, indent=2)

        # Check for key match
        for key in self.cur_semantic["formatted"]:
            if key.startswith(id + "_") or key == id:
                return key  # Return the actual matching key

        # Check for value match
        for key, obj in self.cur_semantic["formatted"].items():
            if obj.get("id") == id:
                return key  # Return the key for the matching value

        return None  # No match found
    
    def get_obj_from_id(self, id):
        return self.total_semantic["raw"][id]



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
