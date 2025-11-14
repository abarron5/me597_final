#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import time
import math

from nav_msgs.msg import OccupancyGrid, Odometry


# Import other python packages that you think necessary


class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')

        # Subscribers
        #self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Publisher
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need

        # Modes
        self.mode = "EXPLORE"
        self.spin_start_time = time.time()
        self.spin_duration = 2.0      # seconds to spin in place (25)

        # Robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Map
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_res = 0.05
        self.map_origin = (0, 0)

        # Frontiers
        self.visited_frontiers = set()
        self.last_target = None

        self.get_logger().info('✅ Map started...')


    def timer_cb(self):
        #self.get_logger().info('Task1 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function
        vel = Twist()

        # -------------------- EXPLORE --------------------
        if self.mode == "EXPLORE":
            #vel = self.wall_following_behavior()
            vel = self.frontiers()
            #vel = self.vfh_explore()
            self.publisher_.publish(vel)

    # ---------------- LIDAR CALLBACK ----------------   
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)

        N = len(ranges)

        # Use minimum, not mean — catches table legs
        self.front = np.min(ranges[int(N*0.45):int(N*0.55)])
        self.left  = np.min(ranges[int(N*0.25):int(N*0.45)])
        self.right = np.min(ranges[int(N*0.55):int(N*0.75)])

        self.ranges_full = ranges
        print("min front:", self.front, "min left:", self.left, "min right:", self.right)

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2 * (q.w*q.z + q.x*q.y)
        cosy = 1 - 2 * (q.y*q.y + q.z*q.z)
        self.robot_yaw = math.atan2(siny, cosy)

        #self.get_logger().info(
        #    f"ODOM pose: ({self.robot_x:.2f}, {self.robot_y:.2f}), yaw={self.robot_yaw:.2f}"
        #)


    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_res = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

    def frontiers(self):
        vel = Twist()

        if self.map_data is None:
            return vel

        M = self.map_data
        H, W = self.map_height, self.map_width

        frontier_cells = []

        # --- Step 2 frontier detection ---
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if M[y, x] == 0:
                    nei = M[y-1:y+2, x-1:x+2]
                    if np.any(nei == -1):
                        frontier_cells.append((x, y))

        if not frontier_cells:
            self.get_logger().info("No frontiers!")
            return vel

        # --- Robot position in grid coords ---
        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)

        # --- Choose nearest frontier ---
        valid_frontiers = [f for f in frontier_cells if f not in self.visited_frontiers]

        if not valid_frontiers:
            self.get_logger().info("💤 All frontiers visited — resetting")
            self.visited_frontiers.clear()
            valid_frontiers = frontier_cells

        target_grid = min(valid_frontiers, key=lambda f: (f[0]-rx)**2 + (f[1]-ry)**2)
        self.last_target = target_grid
        tx, ty = target_grid

        # Convert target to world
        wx, wy = self.grid_to_world(tx, ty)

        #self.get_logger().info(f"🎯 Nearest frontier (grid): {target_grid}")
        #self.get_logger().info(f"🌍 Nearest frontier (world): ({wx:.2f}, {wy:.2f})")

        # --- Step 3: Rotate toward target ---
        vel = self.rotate_toward(wx, wy)
        if vel is not None:
            return vel  # still rotating

        # --- Step 4: Move toward target once aligned ---
        dx = wx - self.robot_x
        dy = wy - self.robot_y
        dist = math.hypot(dx, dy)

        # Safety: stop for obstacles
        if hasattr(self, "front") and self.front < 0.35:
            stop = Twist()
            stop.angular.z = 0.6   # turn away a little
            stop.linear.x = 0.0
            self.get_logger().info("🚫 Obstacle ahead — turning")
            return stop

        # If close to frontier, stop and allow new goal selection next cycle
        if dist < 0.05:     # ~15 cm
            stop = Twist()
            self.get_logger().info("🎉 Reached frontier region!")
            self.visited_frontiers.add(self.last_target)
            self.last_target = None
            return stop

            return stop

        # Otherwise move forward
        vel = Twist()
        vel.linear.x = 0.10          # forward speed
        vel.angular.z = 0.0

        self.get_logger().info(f"➡ Moving toward frontier. dist={dist:.2f}m")

        return vel


    
    def grid_to_world(self, gx, gy):
        wx = self.map_origin[0] + gx * self.map_res
        wy = self.map_origin[1] + gy * self.map_res
        return wx, wy
    
    def world_to_grid(self, wx, wy):
        gx = int((wx - self.map_origin[0]) / self.map_res)
        gy = int((wy - self.map_origin[1]) / self.map_res)
        return gx, gy

    def rotate_toward(self, wx, wy):
        vel = Twist()

        dx = wx - self.robot_x
        dy = wy - self.robot_y

        target_angle = math.atan2(dy, dx)
        yaw_error = target_angle - self.robot_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # === Stop when close enough ===
        if abs(yaw_error) < 0.20:   # wider threshold
            self.get_logger().info(f"✔ Aligned with frontier (yaw_error={yaw_error:.2f})")
            return None

        # === Proportional turn for stability ===
        k = 1.2
        turn_rate = k * yaw_error

        # Limit angular velocity
        turn_rate = np.clip(turn_rate, -0.6, 0.6)

        vel.angular.z = turn_rate
        vel.linear.x = 0.0

        self.get_logger().info(f"↻ Rotating... yaw_error={yaw_error:.2f}, cmd={turn_rate:.2f}")

        return vel





def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

