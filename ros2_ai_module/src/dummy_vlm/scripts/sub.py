#!/usr/bin/env python3

from random import randrange
import math

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Int32

from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker
from tf_transformations import quaternion_from_euler

from goalsnapper import GoalSnapperNode
from depth_overlay import DepthOverlayNode


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
        self.waypoint_dis = 1.0

        self.question = ""

        self.waypointID = 0

        self.depthOverlay = depthOverlay
        self.goalSnapper = goalSnapper

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

    def timer_cb(self):
        if (self.question == ""):
            return
        #determine question type
        #handle nagivation in another node
        

        if (self.question.lower().find("find") == 0):
            self.get_logger().info("Marking and navigating to object")
            self.publish_object_marker()
            self.publish_object_waypoint()
        elif (self.question.lower().find("how many") == 0):
            self.del_object_marker()
            ans = randrange(10)
            self.get_logger().info(f"{ans}")
            self.publish_numerical_answer(ans)
        else:
            self.del_object_marker()
            self.get_logger().info("Navigation starts")
            self.publish_path_waypoints()
        #     self.get_logger().info("Navigation ends")

        self.question = ""
        self.get_logger().info("Awaiting question...")

    
    def execute_navigate_tool(self, target_id):
        """Example of executing a navigation command."""
        

        px_x = 1920 * 60//100
        px_y = 286
        world_coordinates = self.depthOverlay.get_3d_point(v = px_y, u = px_x)
        self.get_logger().info(f"world {world_coordinates}")
        snapped_coordinates = self.goalSnapper.find_closest_traversable_point(world_coordinates)
        self.get_logger().info(f"snapped {snapped_coordinates}")

        waypoint_msg = Pose2D()

        waypoint_msg.x = snapped_coordinates[0]
        waypoint_msg.y = snapped_coordinates[1]
        waypoint_msg.theta = 0.0
        self.waypoint_publisher.publish(waypoint_msg)

        # 1. Get the ideal, but potentially unreachable, goal for the target_id
        unreachable_goal_pose = self.get_pose_from_map(target_id) # Your function to get the pose
        unreachable_goal_point = [
            unreachable_goal_pose.position.x,
            unreachable_goal_pose.position.y,
            unreachable_goal_pose.position.z
        ]
        
        # 2. Use the helper class to snap it to the closest traversable point
        if self.goal_snapper.kdtree:
            reachable_goal_point = self.goal_snapper.snap_point(unreachable_goal_point)
            
            if reachable_goal_point:
                self.get_logger().info(f"Snapped goal {unreachable_goal_point} to {reachable_goal_point}")
                # 3. Send the guaranteed-to-be-reachable goal to your navigator
                self.send_goal_to_navigator(reachable_goal_point)
            else:
                self.get_logger().error("Failed to snap goal.")
        else:
            self.get_logger().warn("Goal snapper not ready yet, cannot navigate.")

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
        dis_x = self.vehicle_x - self.waypoint_x[self.waypointID]
        dis_y = self.vehicle_y - self.waypoint_y[self.waypointID]

        self.get_logger().info(f"{math.sqrt(dis_x*dis_x + dis_y*dis_y)}")

        return math.sqrt(dis_x*dis_x + dis_y*dis_y) < self.waypoint_dis

    def waypoint_timer_cb(self):
        waypoint_msg = Pose2D()
        if (self.waypoint_reached()):
            self.waypointID += 1

            if(self.waypointID == len(self.waypoint_x)-1):
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
    global goalSnapper, depthOverlay
    rclpy.init(args=args)

    goalSnapper = GoalSnapperNode()
    depthOverlay = DepthOverlayNode()

    dummyVLM = DummyVLM()

    executor = MultiThreadedExecutor()
    executor.add_node(goalSnapper)
    executor.add_node(depthOverlay)

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
