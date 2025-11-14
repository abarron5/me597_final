#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import time

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
        self.mode = "SPIN_INIT"
        self.spin_start_time = time.time()
        self.spin_duration = 2.0      # seconds to spin in place (25)

        # Sweep control
        self.last_sweep_time = time.time()
        self.sweep_interval = 200.0
        self.sweep_duration = 10.0
        self.is_sweeping = False
        self.sweep_start_time = 0.0
        self.sweep_type = None  # "room" or "periodic"

        self.get_logger().info('✅ Map started...')


    def timer_cb(self):
        #self.get_logger().info('Task1 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function
        vel = Twist()

        # -------------------- INITIAL SPIN --------------------
        if self.mode == "SPIN_INIT":
            elapsed = time.time() - self.spin_start_time
            vel.linear.x = 0.0
            vel.angular.z = 0.25  # steady spin

            if elapsed > self.spin_duration:
                #self.mode = "SEARCH"
                #self.get_logger().info("🧭 Spin complete -> searching for nearest wall...")
                self.mode = "FOLLOW_WALL"
                self.get_logger().info("🚧 Wall detected -> switching to wall-following mode.")            
            self.publisher_.publish(vel)
            return

        # -------------------- SEARCH FOR WALL --------------------
        if self.mode == "SEARCH":
            vel.linear.x = 0.1
            vel.angular.z = 0.0
            
            # Once we detect a nearby wall, switch to wall following
            if self.front_distance < 2.5 or self.right_distance < 2.5:
                self.mode = "FOLLOW_WALL"
                self.get_logger().info("🚧 Wall detected -> switching to wall-following mode.")
            self.publisher_.publish(vel)
            return

        # -------------------- NORMAL WALL FOLLOW --------------------
        if self.mode == "FOLLOW_WALL":
            #vel = self.wall_following_behavior()
            #vel = self.dwa_like_explore()
            vel = self.vfh_explore()
            self.publisher_.publish(vel)

    # ---------------- LIDAR CALLBACK ----------------
    """def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        #mid = len(ranges) // 2
        #self.front_distance = np.mean(ranges[mid-5:mid+5])
        #right_side = ranges[int(len(ranges)*0.75):]
        #self.prev_right_distance = self.right_distance
        #self.right_distance = np.mean(right_side)
        #left_side = ranges[int(len(ranges)*0.1):int(len(ranges)*0.25)]
        #self.left_distance = np.mean(left_side)

        N = len(ranges)
        self.front_distance = np.mean(ranges[int(N*0.45):int(N*0.55)])  # 90° span
        self.left_distance  = np.mean(ranges[int(N*0.25):int(N*0.45)])  # 90° span
        self.right_distance = np.mean(ranges[int(N*0.55):int(N*0.75)])  # 90° span
"""
    
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)

        N = len(ranges)

        # Use minimum, not mean — catches table legs
        self.front = np.min(ranges[int(N*0.45):int(N*0.55)])
        self.left  = np.min(ranges[int(N*0.25):int(N*0.45)])
        self.right = np.min(ranges[int(N*0.55):int(N*0.75)])

        # For VFH steering
        self.ranges_full = ranges
        print("min front:", self.front, "min left:", self.left, "min right:", self.right)


    # ---------------- WALL FOLLOW + SWEEP ----------------
    def wall_following_behavior(self):
        vel = Twist()
        t = time.time()

        front = self.front_distance
        right = self.right_distance
        print("Front: ", front, ", Right: ", right)
        prev_right = self.prev_right_distance

        # ---------- ROOM DETECTION ----------
        if (not self.is_sweeping 
            and right > 1.2            # now open
            and prev_right < 0.9):     # previously corridor
            self.is_sweeping = True
            self.sweep_start_time = t
            self.sweep_type = "room"
            self.get_logger().info("🏠 Room detected -> sweep")
            return vel

        # ---------- PERIODIC SWEEP ----------
        if not self.is_sweeping and (t - self.last_sweep_time > self.sweep_interval):
            self.is_sweeping = True
            self.sweep_start_time = t
            self.sweep_type = "periodic"
            self.last_sweep_time = t
            self.get_logger().info("🔄 Periodic sweep")
            return vel

        # ---------- SWEEP IN PROGRESS ----------
        if self.is_sweeping:
            dt = t - self.sweep_start_time
            vel.linear.x = 0.0
            vel.angular.z = 0.25 if dt < self.sweep_duration/2 else -0.25

            if dt > self.sweep_duration:
                self.is_sweeping = False
                self.sweep_type = None
                self.get_logger().info("✅ Sweep complete")
            return vel

        """# ---------- NO NEARBY WALL ----------
        if right > 2.5 and front > 2.5:
            vel.linear.x = 0.3
            vel.angular.z = 0.2  # gentle turn to find a wall
            return vel"""

        
        """# ---------- CORNER HANDLING ----------
        if front < 0.9 and right < 1.2:
            print("front-right")
            vel.linear.x = 0.0
            vel.angular.z = 0.5
            return vel"""

        # ---------- FRONT WALL TOO CLOSE ----------
        """
        if front < 1.4:
            vel.linear.x = 0.0
            vel.angular.z = 0.5
            self.get_logger().info(f"⛔ Too close to wall! Turning in place (front={front:.2f}) (right={right:.2f})")
            return vel
        # ---------- DETECTING WALL ----------
        if front < 3.0:
            # Begin slowing down
            vel.linear.x = np.clip((front - 0.7) * 0.2, 0.0, 0.15)
            vel.angular.z = 0.0
            self.get_logger().info(f"⚠️ Wall detected: front={front:.2f} -> slowing/turning")
            return vel
        """
        """
        # ---------- APPROACHING WALL ----------
        if front < 1.4:
            # Begin slowing down and turning away
            vel.linear.x = np.clip((front - 0.8) * 0.2, 0.0, 0.05)
            vel.angular.z = max(0.5 * (1.6 - front), 0.10)  # turn more sharply as it gets closer
            self.get_logger().info(f"⚠️ Approaching wall: front={front:.2f} -> slowing/turning")
            return vel
"""
        # ---------- NORMAL WALL FOLLOWING ----------
        error = self.desired_wall_dist - right
        speed_scale = np.clip(front / 2.5, 0.1, 1.0)
        vel.linear.x = self.wall_follow_speed * speed_scale
        vel.angular.z = np.clip(self.k_wall * error, -0.3, 0.3)
        return vel
    
    def dwa_like_explore(self):
        vel = Twist()

        front = self.front_distance
        right = self.right_distance
        left = self.left_distance

        # ---------- EMERGENCY COLLISION AVOID ----------
        if front < 0.70:
            vel.linear.x = 0.0
            vel.angular.z = 0.6
            return vel

        # ---------- CAUTION ZONE ----------
        if front < 0.8:
            vel.linear.x = 0.05
            vel.angular.z = 0.45 if left > right else -0.45
            return vel

        # ---------- BASE FORWARD SPEED ----------
        vel.linear.x = 0.18

        # ---------- RANDOM WANDER BIAS ----------
        t = time.time()
        if t - self.last_random_change > self.random_interval:
            self.random_bias = np.random.uniform(-1.0, 1.0)
            self.last_random_change = t
            self.random_interval = np.random.uniform(5.0, 12.0)

        # ---------- NORMAL EXPLORATION ----------
        open_bias = (left - right) * 0.50
        wander = self.random_bias * 0.35

        vel.angular.z = np.clip(open_bias + wander, -1.0, 1.0)

        return vel
    
    def vfh_explore(self):
        vel = Twist()

        # 1. Immediate collision reflex
        if self.front < 0.35:
            vel.linear.x = 0.0
            vel.angular.z = 0.8  # turn left sharply
            return vel

        # 2. Slow approach
        if self.front < 0.7:
            vel.linear.x = 0.05
            vel.angular.z = 0.5 if self.left > self.right else -0.5
            return vel

        # 3. Compute best direction (VFH-lite)
        ranges = self.ranges_full
        N = len(ranges)

        # 60 candidate directions
        sectors = 60
        scores = []

        for i in range(sectors):
            # Sector center angle
            idx = int(i * N / sectors)
            window = ranges[max(0, idx-2):min(N, idx+3)]
            d = np.min(window)

            # Score = go toward maximum clearance
            scores.append(d)

        # Find clearest direction
        best = int(np.argmax(scores))

        # Convert to angular velocity
        target_angle = (best / sectors) * 2*np.pi - np.pi  # [-pi, pi]
        vel.angular.z = np.clip(target_angle * 0.7, -1.0, 1.0)

        # Forward velocity depends on clearance
        vel.linear.x = 0.2 if self.front > 1.0 else 0.1

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