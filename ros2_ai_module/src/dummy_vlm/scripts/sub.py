#!/usr/bin/env python3

from random import randrange
import math
import json
import cv2
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Int32

from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker
from tf_transformations import quaternion_from_euler
import requests

from goalsnapper import GoalSnapperNode
from depth_overlay import DepthOverlayNode
from sensors import SensorNode


class DummyVLM(Node):

    def __init__(self):
        super().__init__('dummy_VLM')

        self.obj_mid_x = 3.3707286769194647
        self.obj_mid_y = -2.092013168894541
        self.obj_mid_z = 0.50101238489151
        self.obj_l = 2.8618994
        self.obj_w = 1.198852
        self.obj_h = 1.0225521
        self.object_heading = 0.001148942756478987

        self.waypoint_x = [-0.1767035871744156, 4.454692840576172]
        self.waypoint_y = [-1.3834829330444336, -4.007359027862549]
        self.waypoint_heading = [0.0, 0.0]
        self.waypoint_dis = 0.3

        self.sensorSnapshot = None
        self.waypoint_msg = Pose2D()
        self.navigation_in_progress = False
        self.ai_request_in_progress = False
        self.stop_detected_time = None

        self.question = ""
        self.waiting_for_stop = False

        self.waypointID = 0

        self.depthOverlay = depthOverlay
        self.goalSnapper = goalSnapper
        self.sensorNode = sensorNode

        # self.depth_overlay.get_3d_point(3, 4)

        self.subscription_question = self.create_subscription(
            String,
            "/challenge_question",
            self.question_cb,
            5
        )

        self.waypoint_publisher = self.create_publisher(
            Pose2D,
            "/way_point_with_heading",
            5
        )
        self.object_marker_publisher = self.create_publisher(
            Marker,
            "/selected_object_marker",
            5
        )
        self.numerical_answer_publisher = self.create_publisher(
            Int32,
            "/numerical_response",
            5
        )

        self.timer = self.create_timer(
            0.01,
            self.timer_cb
        )
        self.waypoint_timer = None
        self.get_logger().info("Awaiting question...")

    def question_cb(self, msg: String):
        self.question = msg.data
        self.get_logger().info(f"I heard: {self.question}")

    def stop_wait_callback(self):
        self.waiting_for_stop = False
        self.get_logger().info("Robot stop wait complete.")
        self.stop_wait_timer.cancel()

    def is_robot_stopped(self, linear_threshold=0.01, angular_threshold=0.01):

        velocity = self.sensorNode.twist
        if(self.sensorNode.twist == None):
            return
        
        linear_speed = math.sqrt(velocity.linear.x**2 + velocity.linear.y**2)
        angular_speed = abs(velocity.angular.z)
        
        return linear_speed < linear_threshold and angular_speed < angular_threshold

    def timer_cb(self):
        if self.navigation_in_progress:
            if self.is_robot_stopped():
                # If this is the first time we've detected a stop, record the time.
                if self.stop_detected_time is None:
                    self.stop_detected_time = self.get_clock().now()
                    # self.get_logger().info("Robot has stopped, starting 5-second confirmation timer...")
                
                # Check if 5 seconds have passed since the stop was first detected.
                elif (self.get_clock().now() - self.stop_detected_time).nanoseconds / 1e9 >= 3.0:
                    self.navigation_in_progress = False
                    self.stop_detected_time = None # Reset for next time
                    self.get_logger().info("Robot has been stopped. Navigation complete.")
            else:
                # If the robot starts moving again, reset the timer.
                if self.stop_detected_time is not None:
                    # self.get_logger().info("Robot started moving again, resetting stop timer.")
                    pass
                self.stop_detected_time = None
                        
            return # Keep waiting while navigation is in progress
        #     if self.waypoint_reached():
        #         self.navigation_in_progress = False
        #         self.waiting_for_stop = True
        #         self.stop_wait_timer = self.create_timer(5.0, self.stop_wait_callback)
        #         self.get_logger().info("Goal reached, waiting for robot to stop...")
        #     return
        # if self.waiting_for_stop:
        #     return
        if self.question == "" or self.ai_request_in_progress:
            return
        # determine question type
        # handle nagivation in another node
        sensorSnapshot = self.sensorNode.get_sensor_state()
        self.sensorSnapshot = sensorSnapshot
        # return

        if (sensorSnapshot == None):
            # self.get_logger().info("sensor data not ready")
            return
        self.get_logger().info("Making request to ai")
        self.ai_request_in_progress = True

        def ai_request():
            url = "http://127.0.0.1:5000/fn_call"
            data = {
                "challenge_question": self.question,
                "tasks": [],
                "current_view": sensorSnapshot.get("cur_semantic"),
                "all_objects_discovered": sensorSnapshot.get("total_semantic"),
                "image": sensorSnapshot.get("annotated_image")
            }
            try:
                response = requests.post(url, json=data).json()
                tool_name = response.get("name")
                self.get_logger().info(f"{response}")
                args = response.get("args")
                self.handle_ai_response(tool_name, args)
            finally:
                self.ai_request_in_progress = False

        threading.Thread(target=ai_request, daemon=True).start()
        # self.get_logger().info(f"Reasoning: {args.get('reasoning')}")

        # # self.question = ""
        # return

        # if (self.question.lower().find("find") == 0):
        #     self.get_logger().info("Marking and navigating to object")
        #     self.publish_object_marker()
        #     self.publish_object_waypoint()
        # elif (self.question.lower().find("how many") == 0):
        #     self.del_object_marker()
        #     ans = randrange(10)
        #     self.get_logger().info(f"{ans}")
        #     self.publish_numerical_answer(ans)
        # else:
        #     self.del_object_marker()
        #     self.get_logger().info("Navigation starts")
        #     self.publish_path_waypoints()
        # #     self.get_logger().info("Navigation ends")

        # self.question = ""
        # self.get_logger().info("Awaiting question...")

    def handle_ai_response(self, tool_name, args):
        if (tool_name == "navigate_to_point"):
            points = args.get("points")
            self.execute_navigate_tool(points)
        elif (tool_name == "describe_point"):
            description = args.get("description")
            self.handle_describe_point(description)
        elif (tool_name == "verify_object_exists"):
            answer = args.get("target_id")
            resp = self.sensorNode.get_full_id(answer)
            self.get_logger().info(f"answer{answer} resp {resp}")
            self.handle_verify_object_exists(resp, answer)
        elif (tool_name == "submit_final_object_reference"):
            answer = args.get("answer")
            self.handle_submit_object_reference(answer)
        elif (tool_name == "submit_final_count"):
            answer = args.get("answer")
            self.handle_submit_object_count(answer)
        elif (tool_name == "finish_instruction_following"):
            self.handle_finish_instrution_following()

    def handle_describe_point(self, description):
        self.get_logger().info("Making request to ai")

        url = "http://127.0.0.1:5000/get_pixel"
        data = {
            "description": description,
            "image": self.sensorSnapshot.get("image")
        }
        response = requests.post(url, json=data).json()
        tool_name = response.get("name")
        self.get_logger().info(f"{response}")
        args = response.get("args")
        self.handle_ai_response(tool_name, args)

    def handle_verify_object_exists(self, actualId, aiId):

        if (actualId):
            self.get_logger().info(f"{aiId} {actualId}")
            observation = {
                "tool_name": "verify_object_exists",
                "response": {
                    "exists": True,
                    "status": f"Verification successful. Object {aiId} is a known object."
                }
            }
        else:
            observation = {
                "name": "verify_object_exists",
                "response": {
                    "exists": False,
                    "status": f"Verification FAILED. Object {aiId} does not exist in the known objects list."
                }
            }

        self.get_logger().info("Making request to ai")

        url = "http://127.0.0.1:5000/add_tool_response"
        data = {
            "tool_response": observation,
        }
        response = requests.post(url, json=data).json()
        tool_name = response.get("name")
        self.get_logger().info(f"{response}")
        args = response.get("args")
        self.handle_ai_response(tool_name, args)

    def handle_get_visual_confirmation(self, id):

        obj = self.sensorNode.get_obj_from_id(id)

        box_center = np.array(
            [obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
        box_scale = np.array([obj.scale.x, obj.scale.y, obj.scale.z])

        bbox = self.depthOverlay.project_3d_box_to_2d(
            box_center, box_scale, (1920, 640))
        img = None

        if not bbox:
            self.get_logger().info("BB error")
            return

        u_min, v_min, u_max, v_max = bbox
        img = self.sensorSnapshot.get("cv_img")
        self.get_logger().info("drawing bb")

        def show_img():
            # Draw the rectangle on the image
            cv2.rectangle(img, (u_min, v_min), (u_max, v_max), (0, 255, 0), 2)
            cv2.imshow('Image with Point', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        threading.Thread(target=show_img, daemon=True).start()

        img = self.sensorNode.get_base64_img(img)
        observation = {
            "name": "get_visual_confirmation",
            "response": {
                "msg": "Action complete. The bounding box for the object ID you requested has been drawn on the new image. Analyze the image to confirm if this is the correct object you were looking for",
            },
            "img": img
        }

        self.get_logger().info("Making request to ai")

        url = "http://127.0.0.1:5000/add_tool_response"
        data = {
            "tool_response": observation,
        }
        response = requests.post(url, json=data).json()
        tool_name = response.get("name")
        self.get_logger().info(f"{response}")
        args = response.get("args")
        self.handle_ai_response(tool_name, args)

    def execute_navigate_tool(self, points):
        """Example of executing a navigation command."""

        for point in points:  # [y, x]
            px_x = point[1]*1920//1000
            px_y = point[0]*640//1000
            img = self.sensorSnapshot.get("cv_img")

            # cv2.circle(img, (px_x, px_y), radius=8, color=(0, 0, 255), thickness=-1)
            # # Show the image
            # cv2.imshow('Image with Point', img)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()

            world_coordinates = None
            self.get_logger().info("Waiting for DepthOverlayNode to provide a valid 3D point...")
            world_coordinates = self.depthOverlay.get_3d_point(v=px_y, u=px_x)

            # self.get_logger().info(f"world {world_coordinates}")
            if (world_coordinates is not None):
                self.get_logger().info("Found valid point in lidar")
                break
        if (world_coordinates is None):
            self.get_logger().info("No valid point found ask ai to try different point")
            observation = {
                "name": "navigate_to_point_fn",
                "response": {
                    "msg": "Observation: FAILED. None of the pixel coordinates you selected correspond to a valid 3D point in the environment. Please analyze the image again and keep calling the `describe_point_fn` tool with a new set of points until you succeed.",
                }
            }

            self.get_logger().info("Making request to ai")

            url = "http://127.0.0.1:5000/add_tool_response"
            data = {
                "tool_response": observation,
            }
            response = requests.post(url, json=data).json()
            tool_name = response.get("name")
            self.get_logger().info(f"{response}")
            args = response.get("args")
            self.handle_ai_response(tool_name, args)
            return

        # Wait for GoalSnapperNode's KD-tree to be ready
        self.get_logger().info("Waiting for GoalSnapperNode to build k-d tree...")
        while self.goalSnapper.kdtree is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        snapped_coordinates = self.goalSnapper.find_closest_traversable_point(
            world_coordinates)
        # self.get_logger().info(f"snapped {snapped_coordinates}")

        self.waypoint_msg.x = round(snapped_coordinates[0], 3)
        self.waypoint_msg.y = round(snapped_coordinates[1], 3)
        self.waypoint_msg.theta = 0.0
        self.get_logger().info(
            f"navigating from {self.sensorSnapshot.get('vehicle_x')}, {self.sensorSnapshot.get('vehicle_y')} to {self.waypoint_msg}")

        self.navigation_in_progress = True
        self.waypoint_publisher.publish(self.waypoint_msg)

        # self.get_logger().info("Waiting for robot to reach the goal...")
        # while not self.waypoint_reached():
        #     rclpy.spin_once(self, timeout_sec=0.1)
        #     # Optionally, add a timeout or break condition if needed

        # self.get_logger().info("Goal reached or navigation stopped.")

    def handle_submit_object_reference(self, id):
        obj_key = self.sensorNode.get_full_id(id)
        object = self.sensorNode.get_obj_from_id(obj_key)
        object.action = Marker.ADD
        object.type = Marker.CUBE
        self.object_marker_publisher.publish(object)

        self.question = ""
        self.get_logger().info("published object marker. stopping")

    def handle_submit_object_count(self, count):
        count = int(count)
        numerical_response = Int32()
        numerical_response.data = count

        self.numerical_answer_publisher.publish(numerical_response)

        self.question = ""
        self.get_logger().info("published object count. stopping")

    def handle_finish_instrution_following(self):
        self.question = ""
        self.get_logger().info("finished instruction following. stopping")

    def publish_object_waypoint(self):
        waypoint = Pose2D()
        waypoint.x = self.obj_mid_x
        waypoint.y = self.obj_mid_y
        waypoint.theta = 0.0

        self.waypoint_publisher.publish(waypoint)

    def publish_object_marker(self):
        object_marker = Marker()
        object_marker.header.frame_id = "map"
        object_marker.header.stamp = self.get_clock().now().to_msg()
        object_marker.ns = "sofa"
        object_marker.id = 0
        object_marker.action = Marker.ADD
        object_marker.type = Marker.CUBE
        object_marker.pose.position.x = self.obj_mid_x
        object_marker.pose.position.y = self.obj_mid_y
        object_marker.pose.position.z = self.obj_mid_z
        q = quaternion_from_euler(0, 0, self.object_heading)
        object_marker.pose.orientation.w = q[0]
        object_marker.pose.orientation.x = q[1]
        object_marker.pose.orientation.y = q[2]
        object_marker.pose.orientation.z = q[3]
        object_marker.scale.x = self.obj_l
        object_marker.scale.y = self.obj_w
        object_marker.scale.z = self.obj_h
        object_marker.color.a = 1.0
        object_marker.color.r = 0.0
        object_marker.color.g = 0.0
        object_marker.color.b = 1.0
        self.object_marker_publisher.publish(object_marker)

    def del_object_marker(self):
        object_marker = Marker()
        object_marker.header.frame_id = "map"
        object_marker.header.stamp = self.get_clock().now().to_msg()
        object_marker.ns = "sofa"
        object_marker.id = 0
        object_marker.action = Marker.DELETE
        object_marker.type = Marker.CUBE

        self.object_marker_publisher.publish(object_marker)

    def publish_numerical_answer(self, ans):
        numerical_response = Int32()
        numerical_response.data = ans

        self.numerical_answer_publisher.publish(numerical_response)

    def waypoint_reached(self):
        sensor_snapshot = self.sensorNode.get_sensor_state()
        dis_x = sensor_snapshot.get("vehicle_x") - self.waypoint_msg.x
        dis_y = sensor_snapshot.get("vehicle_y") - self.waypoint_msg.y

        # self.get_logger().info(f"{math.sqrt(dis_x*dis_x + dis_y*dis_y)}")

        return math.sqrt(dis_x*dis_x + dis_y*dis_y) < self.waypoint_dis

    def waypoint_timer_cb(self):
        waypoint_msg = Pose2D()
        if (self.waypoint_reached()):
            self.waypointID += 1

            if (self.waypointID == len(self.waypoint_x)-1):
                self.waypoint_timer.cancel()

            waypoint_msg.x = self.waypoint_x[self.waypointID]
            waypoint_msg.y = self.waypoint_y[self.waypointID]
            waypoint_msg.theta = self.waypoint_heading[self.waypointID]
            self.waypoint_publisher.publish(waypoint_msg)

    def publish_path_waypoints(self):
        self.waypointID = 0
        waypointNum = len(self.waypoint_x)

        waypoint_msg = Pose2D()

        if (waypointNum == 0):
            self.get_logger().info("No waypoint available, exit")

        waypoint_msg.x = self.waypoint_x[self.waypointID]
        waypoint_msg.y = self.waypoint_y[self.waypointID]
        waypoint_msg.theta = self.waypoint_heading[self.waypointID]
        self.waypoint_publisher.publish(waypoint_msg)
        self.waypoint_timer = self.create_timer(0.1, self.waypoint_timer_cb)
        self.get_logger().info("3")


def main(args=None):
    global goalSnapper, depthOverlay, sensorNode
    rclpy.init(args=args)

    goalSnapper = GoalSnapperNode()
    depthOverlay = DepthOverlayNode()
    sensorNode = SensorNode()

    dummyVLM = DummyVLM()

    executor = MultiThreadedExecutor()
    executor.add_node(goalSnapper)
    executor.add_node(depthOverlay)
    executor.add_node(sensorNode)

    executor.add_node(dummyVLM)

    try:
        executor.spin()
    finally:
        dummyVLM.destroy_node()
        goalSnapper.destroy_node()
        depthOverlay.destroy_node()
        rclpy.shutdown()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dummyVLM.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
