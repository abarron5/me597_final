#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

import numpy as np
import math


class Task1(Node):
    def __init__(self):
        super().__init__('task1_algorithm')

        # ----------------------
        # SUBSCRIBERS
        # ----------------------
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # ----------------------
        # PUBLISHER
        # ----------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ----------------------
        # DATA STORAGE
        # ----------------------
        # map (store flat)
        self.map_flat = None        # 1D numpy array of occupancy values
        self.map_width = 0
        self.map_height = 0
        self.map_res = 0.05
        self.map_origin = (0.0, 0.0)

        # robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # LIDAR
        self.front_dist = 10.0

        # frontier target (world coords)
        self.target = None
        # also keep last chosen grid cell to avoid repeat selections
        self.last_chosen_grid = None

        self.mode = "EXPLORE"

        # timer
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info("🚀 Frontier Explorer (Stage 1) Started")


    # ============================================================
    # MAP CALLBACK
    # ============================================================
    def map_callback(self, msg: OccupancyGrid):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_res = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        # store as 1D numpy array (row-major: x + y*width)
        self.map_flat = np.array(msg.data, dtype=np.int8)
        # debug
        # self.get_logger().info(f"Map loaded: w={self.map_width} h={self.map_height} res={self.map_res} origin={self.map_origin}")


    # ============================================================
    # AMCL CALLBACK
    # ============================================================
    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # quaternion → yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)


    # ============================================================
    # SCAN CALLBACK
    # ============================================================
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        N = len(ranges)
        self.front_dist = np.min(ranges[int(N*0.45):int(N*0.55)])  # center 45 degrees


    # ============================================================
    # HELPERS: grid <-> world conversions (robust)
    # ============================================================
    def grid_to_world(self, col: int, row: int):
        """
        Convert grid indices (col, row) to world coordinates (x,y).
        Uses occupancy grid convention: index = col + row * width
        world_x = origin_x + (col + 0.5) * res
        world_y = origin_y + (row + 0.5) * res
        """
        ox, oy = self.map_origin
        x = ox + (col + 0.5) * self.map_res
        y = oy + (row + 0.5) * self.map_res
        return x, y

    def world_to_grid(self, x: float, y: float):
        """
        Convert world coordinates to grid (col, row). Clamp to map bounds.
        """
        ox, oy = self.map_origin
        col = int((x - ox) / self.map_res)
        row = int((y - oy) / self.map_res)

        # clamp
        col = max(0, min(self.map_width - 1, col))
        row = max(0, min(self.map_height - 1, row))
        return col, row

    def get_map_value(self, col: int, row: int):
        if self.map_flat is None:
            return -1
        if col < 0 or col >= self.map_width or row < 0 or row >= self.map_height:
            return 100  # treat out-of-bounds as obstacle
        idx = col + row * self.map_width
        return int(self.map_flat[idx])


    # ============================================================
    # FRONTIER DETECTION (explicit indexing)
    # ============================================================
    def find_frontiers(self):
        if self.map_flat is None:
            return []

        frontiers = []
        W = self.map_width
        H = self.map_height
        M = self.map_flat

        # iterate over interior cells
        for row in range(1, H - 1):
            base = row * W
            for col in range(1, W - 1):
                val = int(M[base + col])
                # consider free cell (0)
                if val == 0:
                    # check 8-neighborhood for unknown (-1)
                    neigh_unknown = False
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nidx = (col + dx) + (row + dy) * W
                            if int(M[nidx]) == -1:
                                neigh_unknown = True
                                break
                        if neigh_unknown:
                            break
                    if neigh_unknown:
                        frontiers.append((col, row))

        return frontiers


    # ============================================================
    # CHOOSE NEAREST FRONTIER (avoid immediate repeats)
    # ============================================================
    def choose_target(self, frontiers):
        if not frontiers:
            return None

        # robot grid coords
        rcol, rrow = self.world_to_grid(self.robot_x, self.robot_y)

        # compute squared distances, prefer cells not equal to last chosen
        best = None
        best_d2 = float('inf')
        for col, row in frontiers:
            d2 = (col - rcol)**2 + (row - rrow)**2
            # deprioritize the last chosen cell to avoid repeats
            if self.last_chosen_grid is not None and (col, row) == self.last_chosen_grid:
                d2 += 1e6
            if d2 < best_d2:
                best_d2 = d2
                best = (col, row)

        if best is None:
            return None

        self.last_chosen_grid = best
        wx, wy = self.grid_to_world(*best)
        # debug info returned as tuple
        return (wx, wy, best[0], best[1], math.sqrt(best_d2))


    # ============================================================
    # SIMPLE GO-TO CONTROLLER
    # ============================================================
    def go_to_point(self, gx, gy):
        vel = Twist()

        dx = gx - self.robot_x
        dy = gy - self.robot_y

        angle = math.atan2(dy, dx)
        yaw_error = angle - self.robot_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # Emergency obstacle stop
        if self.front_dist < 0.35:
            vel.linear.x = 0.0
            vel.angular.z = 0.6
            return vel

        # Turn toward goal
        if abs(yaw_error) > 0.3:
            vel.angular.z = 0.6 * yaw_error
            vel.linear.x = 0.0
        else:
            vel.angular.z = 0.4 * yaw_error
            vel.linear.x = 0.15

        return vel


    # ============================================================
    # MAIN TIMER LOOP
    # ============================================================
    def timer_cb(self):
        if self.mode != "EXPLORE" or self.map_flat is None:
            return

        frontiers = self.find_frontiers()

        # No more frontiers = done
        if not frontiers:
            self.get_logger().info("🎉 Exploration Complete! No more frontiers.")
            self.cmd_pub.publish(Twist())
            return

        # Need new target?
        if self.target is None:
            chosen = self.choose_target(frontiers)
            if chosen is not None:
                wx, wy, ccol, crow, d = chosen
                self.target = (wx, wy)
                self.get_logger().info(f"📍 New Frontier Target: world=({wx:.3f},{wy:.3f}) grid=({ccol},{crow}) approx_dist_cells={d:.1f}")
            else:
                self.get_logger().info("⚠️ No suitable frontier chosen.")
                return

        # debug : distance to target
        if self.target:
            tx, ty = self.target
            dist = math.hypot(tx - self.robot_x, ty - self.robot_y)
            self.get_logger().debug(f"Robot pos ({self.robot_x:.3f},{self.robot_y:.3f}) target ({tx:.3f},{ty:.3f}) dist {dist:.3f}")

            # Reached target?
            if dist < 0.30:
                self.get_logger().info("✔ Frontier Reached")
                self.target = None
                return

            # Move toward frontier
            vel = self.go_to_point(tx, ty)
            self.cmd_pub.publish(vel)


def main(args=None):
    rclpy.init(args=args)
    node = Task1()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
