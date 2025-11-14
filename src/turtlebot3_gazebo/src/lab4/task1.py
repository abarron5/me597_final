#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import time

from nav_msgs.msg import OccupancyGrid

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

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Publisher
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()


        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need

        # Wall following
        self.front_distance = 5.0
        self.right_distance = 5.0
        self.left_distance = 5.0

        self.prev_right_distance = 5.0
        self.desired_wall_dist = 1.0
        self.k_wall = 0.5
        self.wall_follow_speed = 0.15


        # Random 
        self.last_random_change = time.time()
        self.random_interval = 8.0
        self.random_bias = 0.0


        # Modes
        self.mode = "EXPLORE"
        self.spin_start_time = time.time()
        self.spin_duration = 2.0      # seconds to spin in place (25)

        # Sweep control
        self.last_sweep_time = time.time()
        self.sweep_interval = 200.0
        self.sweep_duration = 10.0
        self.is_sweeping = False
        self.sweep_start_time = 0.0
        self.sweep_type = None  # "room" or "periodic"

        # Map
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_res = 0.05
        self.map_origin = (0, 0)

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
        target_grid = min(frontier_cells, key=lambda f: (f[0]-rx)**2 + (f[1]-ry)**2)
        tx, ty = target_grid

        # Convert target to world
        wx, wy = self.grid_to_world(tx, ty)

        self.get_logger().info(f"🎯 Nearest frontier (grid): {target_grid}")
        self.get_logger().info(f"🌍 Nearest frontier (world): ({wx:.2f}, {wy:.2f})")

        # Robot still does NOT move yet
        return vel

    
    
    def grid_to_world(self, gx, gy):
        wx = self.map_origin[0] + gx * self.map_res
        wy = self.map_origin[1] + gy * self.map_res
        return wx, wy
    
    def world_to_grid(self, wx, wy):
        gx = int((wx - self.map_origin[0]) / self.map_res)
        gy = int((wy - self.map_origin[1]) / self.map_res)
        return gx, gy




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

