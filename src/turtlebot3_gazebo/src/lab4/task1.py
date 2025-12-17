#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
import math
import time
import heapq
from collections import deque


# ---------- TUNABLE PARAMETERS ----------
OCCUPIED_THRESH = 15          # occupancy grid value >= this is considered an obstacle
UNKNOWN_VAL = -1
INFLATE_RADIUS = 5          
GOAL_NEAR_DIST_M = 0.25       # when within this world distance to goal waypoint, pop waypoint
MAX_LINEAR_SPEED = 0.8
MAX_ANGULAR_SPEED = 1.0
REPLAN_INTERVAL = 1.0         # seconds between forced replans
STUCK_TIME = 10.0             # seconds to consider stuck
STUCK_MOVE_THRESH = 0.05      # meters moved to consider not stuck
BACKUP_TIME = 0.6             # seconds to back up on recovery
RECOVERY_ROTATE_TIME = 1.0    # seconds to rotate during recovery
EMERGENCY_STOP = 0.35
LIDAR_ANGLE = 15
MAX_FRONTIER_DIST_M = 10.0
# ----------------------------------------


class Task1(Node):

    def __init__(self):
        super().__init__('task1_algorithm')

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.work_map_pub = self.create_publisher(OccupancyGrid, '/working_map', map_qos)

        self.timer = self.create_timer(0.1, self.timer_cb)

        # Robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Map
        self.map_data = None
        self.inflated_map = None
        self.map_width = 0
        self.map_height = 0
        self.map_res = 0.05
        self.map_origin = (0.0, 0.0)
        self.map_seq = 0  # increment on map 

        # LIDAR
        self.front = 10.0
        self.front_check = 10.0
        self.left = 10.0
        self.right = 10.0
        self.ranges_full = None

        # Planner state
        self.current_path = []
        self.current_path_world = []
        self.path_goal = None 
        self.last_plan_time = 0.0

        # Stuck detection
        self.last_pos = (0.0, 0.0)
        self.last_movement_time = time.time()
        self.last_cmd = Twist()

        # Oscillation detection (simple sign-flip detector)
        self.last_turn_sign = 0
        self.osc_count = 0
        self.oscillating = False
        self.osc_end = 0.0

        # Recovery, backup or rotate
        self.recovering = False
        self.recovery_end_time = 0.0
        self.recovery_stage = None

        # Exploration bias: persistent exploration direction in world coords (unit vector)
        self.exploration_dir = None

        # Frontiers
        self.visited_mask = None           # numpy bool array same shape as map
        self.VISITED_RADIUS = 5            # radius in grid cells to mark visited around goals (tune)
        self.visited_reset = time.time()
        self.visited_reset_time = 10
        self.failed_frontiers = {}         # dict {(gx,gy): fail_count}
        self.FAILED_LIMIT = 5              # after this many A* fails, temporarily ignore frontier
        self.FAILED_CLEAR_TIME = 1.0       # seconds after which failed_frontiers entry may be dropped
        self._failed_timestamps = {}       # {(gx,gy): last_fail_time}
        self.frontier_count = 100

        # Wall Following
        self.wall_following = False
        self.wall_follow_end_time = 0.0

        # Path Following
        self.last_path_idx = 0

        # Time
        self.start_time = time.time()
        self.end_time = time.time()

        # Emergency U-turn
        self.turning_around = False
        self.turn_end_time = 0.0

        self.get_logger().info("Task 1 node started.")
      

    # ---------------- TIMER LOOP ----------------
    def timer_cb(self):
        now = time.time()

        # ---------------- RECOVERY ----------------
        if self._handle_recovery(now):
            return

        # ---------------- OSCILLATION ----------------
        if self._handle_oscillation(now):
            return

        # ---------------- MAP NOT READY ----------------
        if self.map_data is None:
            self._publish_rotate()
            #self.stop_robot()
            return

        # ---------------- STUCK CHECK ----------------
        if self._check_stuck(now) and self.frontier_count != 0:
            return

        # ---------------- WALL FOLLOW MODE ----------------
        if self.wall_following:
            if time.time() < self.wall_follow_end_time:
                self.wall_follower()
                return
            else:
                self.get_logger().info("Exiting wall-follow, replanning")
                self.wall_following = False
                self.reset_path()
                return

        # ---------------- FOLLOW CURRENT PATH ----------------
        if self.current_path and self.current_path_world:
            if self.path_goal is None:
                pass
                       
            # Get path index
            idx = self.get_path_idx(self.current_path_world, self.last_path_idx)
            idx = max(0, min(idx, len(self.current_path_world)-1))
            current_goal = self.current_path_world[idx]
            speed, heading, dist = self.path_follower_from_xy(current_goal[0], current_goal[1])
            
            # Publish safe command
            self.move_ttbot_safe(speed, heading)
            self.last_cmd = Twist(); self.last_cmd.linear.x = speed; self.last_cmd.angular.z = heading
            self.last_path_idx = idx

            # If at end and close enough -> finish
            if idx >= len(self.current_path_world) - 1 and dist < GOAL_NEAR_DIST_M:
                #self.get_logger().info("Reached path goal grid={}, clearing path".format(self.path_goal))
                gx, gy = self.path_goal if self.path_goal is not None else (None, None)
                self.path_goal = None

                # mark visited area on the visited_mask to avoid reselecting nearby frontiers
                if self.visited_mask is not None and gx is not None:
                    rr = self.VISITED_RADIUS
                    H, W = self.map_height, self.map_width
                    x0 = max(0, gx - rr); x1 = min(W, gx + rr + 1)
                    y0 = max(0, gy - rr); y1 = min(H, gy + rr + 1)
                    ys = np.arange(y0, y1)
                    xs = np.arange(x0, x1)
                    dy = ys[:, None] - gy
                    dx = xs[None, :] - gx

                    disk = (dx*dx + dy*dy) <= (rr*rr)

                    self.visited_mask[y0:y1, x0:x1] |= disk

                # Reset exploration_dir
                #self.exploration_dir = None

                # Clear path lists
                self.current_path = []
                self.current_path_world = []
                self.last_path_idx = 0

            return

        # ---------------- PLAN NEW PATH ----------------
        if now - self.last_plan_time > REPLAN_INTERVAL or not self.current_path:
            if self.frontier_count > 0:
                self.plan_to_frontier(farthest=False)
                self.end_time = time.time()
            else:
                self.get_logger().info("Map complete (or no reachable frontiers). Stopping.")
                self.cmd_pub.publish(Twist())
                total = int(self.end_time - self.start_time)
                self.get_logger().info(f"{int(total / 60)} minutes {total % 60} seconds")
                return
            self.last_plan_time = now

        if not self.current_path:
            self.stop_robot()


    # -----------------------------------------------------
    #      RECOVERY
    # -----------------------------------------------------
    def _handle_recovery(self, now):
        if not self.recovering:
            return False

        # still in recovery
        if now < self.recovery_end_time:
            self.cmd_pub.publish(self.recovery_cmd())
            return True

        # exit recovery
        self.recovering = False
        self.recovery_stage = None
        self.get_logger().info("Recovery finished, replanning.")
        self.reset_path()
        return False
    

    def start_recovery(self):
        # start backup stage first then rotate
        self.recovering = True
        self.recovery_stage = "backup"
        self.recovery_end_time = time.time() + BACKUP_TIME


    def recovery_cmd(self):
        cmd = Twist()
        if self.recovery_stage == "backup":
            # back up slowly
            cmd.linear.x = -0.06
            cmd.angular.z = 0.0
            # if time passed, move to rotate
            if time.time() >= self.recovery_end_time:
                self.recovery_stage = "rotate"
                self.recovery_end_time = time.time() + RECOVERY_ROTATE_TIME
                return cmd
            return cmd
        elif self.recovery_stage == "rotate":
            # rotate in place
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6
            return cmd
        else:
            return Twist()
    

    def _handle_oscillation(self, now):
        if not self.oscillating:
            return False

        if now < self.osc_end:
            self.get_logger().info("Oscillation detected, linear push")
            cmd = Twist()
            cmd.linear.x = 0.12
            self.cmd_pub.publish(cmd)
            return True

        self.oscillating = False
        return False
    

    def _publish_rotate(self):
        self.get_logger().info("Map data not ready, rotating")
        cmd = Twist()
        cmd.angular.z = 0.4
        self.cmd_pub.publish(cmd)


    def _check_stuck(self, now):
        if time.time() - self.start_time < 20:
            return False
        
        if not hasattr(self, "last_cmd"):
            return False

        if getattr(self.last_cmd, "linear", None) is None:
            return False

        if self.last_cmd.linear.x <= 0.03:
            return False

        dx = self.robot_x - self.last_pos[0]
        dy = self.robot_y - self.last_pos[1]
        moved = math.hypot(dx, dy)

        if moved > STUCK_MOVE_THRESH:
            self.last_movement_time = now
            self.last_pos = (self.robot_x, self.robot_y)
            return False

        if now - self.last_movement_time > STUCK_TIME:
            self.get_logger().warning("Stuck detected, performing recovery")
            self.start_recovery()
            rc = self.recovery_cmd()
            self.cmd_pub.publish(rc)
            self.last_cmd = rc
            return True

        return False
    

    # -----------------------------------------------------
    #      CALLBACKS
    # -----------------------------------------------------
    # ---------------- LIDAR ----------------
    def scan_callback(self, msg):
        # Clean ranges
        ranges = np.array(msg.ranges, dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        self.ranges_full = ranges

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        N = len(ranges)

        def angle_to_idx(angle):
            return int((angle - angle_min) / angle_inc)

        # Define angles
        FRONT = 0.0
        LEFT  = math.pi / 2
        RIGHT = -math.pi / 2
        FRONT_RIGHT = -math.pi / 4
        BACK_RIGHT = -3 * math.pi / 4

        w_center = math.radians(25)
        w_side = math.radians(90)

        def median_in_sector(center_angle, w):
            idx0 = angle_to_idx(center_angle - w)
            idx1 = angle_to_idx(center_angle + w)

            idx0 %= N
            idx1 %= N

            if idx0 <= idx1:
                sector = ranges[idx0:idx1]
            else:
                sector = np.concatenate((ranges[idx0:], ranges[:idx1]))

            return float(np.median(sector))

        self.front = median_in_sector(FRONT, w_center)
        self.left  = median_in_sector(LEFT, w_side)
        self.right = median_in_sector(RIGHT, w_side)
        self.right_front  = median_in_sector(FRONT_RIGHT, w_side)
        self.right_back = median_in_sector(BACK_RIGHT, w_side)
        self.front_check = median_in_sector(FRONT, 45)


    # ---------------- ODOM ----------------
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self.robot_yaw = math.atan2(siny, cosy)


    # ---------------- MAP ----------------
    def map_callback(self, msg):
        self.map_data = np.array(msg.data, dtype=int).reshape(msg.info.height, msg.info.width)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_res = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        # bump seq to notice map changes
        self.map_seq += 1

        # build inflated map
        self.inflated_map = self.build_inflated_map(self.map_data, INFLATE_RADIUS)

        if self.visited_mask is None:
            self.visited_mask = np.zeros_like(self.map_data, dtype=bool)
            self.visited_reset = time.time()
        elif self.visited_mask.shape != self.map_data.shape:
            #self.get_logger().info("Visited Cells Cleared")
            self.visited_mask = np.zeros_like(self.map_data, dtype=bool)
            self.visited_reset = time.time()
        elif (time.time() - self.visited_reset) > self.visited_reset_time:
            self.get_logger().info("Visited Cells Cleared")
            self.visited_mask = np.zeros_like(self.map_data, dtype=bool)
            self.visited_reset = time.time()
        """else:
            # Resize mask if map grows (SLAM expanding map)
            if self.visited_mask.shape != self.map_data.shape:
                new_mask = np.zeros_like(self.map_data, dtype=bool)
                H_old, W_old = self.visited_mask.shape
                H_new, W_new = new_mask.shape
                h = min(H_old, H_new)
                w = min(W_old, W_new)
                new_mask[:h, :w] = self.visited_mask[:h, :w]
                self.visited_mask = new_mask"""
        
        self.publish_working_map(self.inflated_map)


    # -----------------------------------------------------
    #      MAP HELPERS
    # -----------------------------------------------------
    def build_inflated_map(self, M, radius_cells):
        H, W = M.shape
        binmap = np.zeros((H, W), dtype=np.uint8)
        binmap[M >= OCCUPIED_THRESH] = 1

        if radius_cells <= 0:
            return binmap

        inflated = binmap.copy()
        dist = np.full((H, W), -1, dtype=int)
        q = deque()

        ys, xs = np.where(binmap == 1)
        for y, x in zip(ys, xs):
            q.append((y, x))
            dist[y, x] = 0

        offs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        while q:
            y, x = q.popleft()
            d = dist[y, x]
            if d >= radius_cells:
                continue
            for dy, dx in offs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] == -1:
                    dist[ny, nx] = d + 1
                    inflated[ny, nx] = 1
                    q.append((ny, nx))

        return inflated
    
    def world_to_grid(self, wx, wy):
        gx = int((wx - self.map_origin[0]) / self.map_res)
        gy = int((wy - self.map_origin[1]) / self.map_res)
        # clamp
        gx = max(0, min(self.map_width - 1, gx))
        gy = max(0, min(self.map_height - 1, gy))
        return gx, gy


    def grid_to_world(self, gx, gy):
        wx = self.map_origin[0] + gx * self.map_res + 0.5*self.map_res
        wy = self.map_origin[1] + gy * self.map_res + 0.5*self.map_res
        return wx, wy


    # -----------------------------------------------------
    #      Frontiers
    # -----------------------------------------------------
    def get_frontier_cells(self):
        if self.map_data is None:
            return []

        frontiers = []
        M = self.map_data
        H, W = M.shape

        for y in range(1, H-1):
            for x in range(1, W-1):
                if not (0 <= M[y, x] <= OCCUPIED_THRESH):
                    continue
                neigh = M[y-1:y+2, x-1:x+2]
                if np.any(neigh == UNKNOWN_VAL):
                    if self.inflated_map is not None and np.any(self.inflated_map[y-1:y+2, x-1:x+2] == 1):
                        continue
                    frontiers.append((x, y))
        return frontiers
    

    def is_cell_free_inflated(self, gx, gy):
        if self.inflated_map is None:
            return False
        if not (0 <= gx < self.map_width and 0 <= gy < self.map_height):
            return False
        return self.inflated_map[gy, gx] == 0
    

    def bfs_distances(self, start_gxgy):  
        sx, sy = start_gxgy
        H, W = self.map_height, self.map_width

        dist = np.full((H, W), np.inf, dtype=float)
        if not (0 <= sx < W and 0 <= sy < H):
            return dist

        if self.inflated_map[sy, sx] != 0:
            return dist

        q = deque()
        dist[sy, sx] = 0.0
        q.append((sx, sy))

        while q:
            x, y = q.popleft()
            d = dist[y, x] + 1.0 

            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if 0 <= nx < W and 0 <= ny < H:
                    if self.inflated_map[ny, nx] == 0 and d < dist[ny, nx]:
                        dist[ny, nx] = d
                        q.append((nx, ny))

        return dist
    

    def plan_to_frontier(self, farthest=False):
        M = self.map_data
        H, W = M.shape
        if self.map_data is None or self.inflated_map is None:
            return 0

        # Get frontiers
        frontiers = self.get_frontier_cells()

        frontier_list = frontiers           
        self.frontier_count = len(frontier_list) 
        self.get_logger().info(f"Frontiers found: {self.frontier_count}")
        
        if not frontiers:
            return 0
        
        filtered_frontiers = []
        now = time.time()

        for fx, fy in frontiers:
            # skip if visited
            if (self.visited_mask is not None
                and 0 <= fy < self.visited_mask.shape[0]
                and 0 <= fx < self.visited_mask.shape[1]
                and self.visited_mask[fy, fx]):
                continue

            # skip if failed too many times
            fail_count = self.failed_frontiers.get((fx, fy), 0)
            if fail_count >= self.FAILED_LIMIT:
                # clear failed entry if stale
                ts = self._failed_timestamps.get((fx, fy), 0)
                if now - ts > self.FAILED_CLEAR_TIME:
                    self.failed_frontiers.pop((fx, fy), None)
                    self._failed_timestamps.pop((fx, fy), None)
                else:
                    self.get_logger().info("Failed frontier")
                    continue

            filtered_frontiers.append((fx, fy)) 
        
        self.get_logger().info(f"Filtered frontiers found: {len(filtered_frontiers)}")

        if not filtered_frontiers:
            self.get_logger().info("No frontiers after filtering")
            #self.visited_mask = np.zeros_like(self.map_data, dtype=bool)
            #self.visited_reset = time.time()
            #self.failed_frontiers = {} 
            #self._failed_timestamps = {} 
            self.start_wall_follow()
            return True

        # Find path distance to each frontier
        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)
        start = (rx, ry)
        self.get_logger().info(f"Start cell inflated_free={self.is_cell_free_inflated(rx, ry)}")
        if not self.is_cell_free_inflated(rx, ry):
            start = self.find_safe_target_near_frontier(rx, ry)
            self.get_logger().info(f"({rx},{ry}) is not free. Using {start}")
        dist_map = self.bfs_distances(start)
       
        F = np.array(filtered_frontiers, dtype=float)
        dx = F[:, 0] - rx
        dy = F[:, 1] - ry
        dist_euclid = np.hypot(dx, dy)

        path_dists = np.array([dist_map[int(fy), int(fx)] for fx, fy in filtered_frontiers])

        # Filter unreachable frontiers    
        reachable_mask = np.isfinite(path_dists)
        if not np.any(reachable_mask):
            self.get_logger().warn("No reachable frontiers (path distance).")
            #self.cmd_pub.publish(Twist())
            self.failed_frontiers = {} 
            self._failed_timestamps = {} 
            self.visited_mask = np.zeros_like(self.map_data, dtype=bool)
            self.visited_reset = time.time()
            self.start_wall_follow()
            return True

        F = F[reachable_mask]
        dx = dx[reachable_mask]
        dy = dy[reachable_mask]
        dist_euclid = dist_euclid[reachable_mask]
        path_dists = path_dists[reachable_mask]

        # Direction preference
        if self.exploration_dir is None:
            ex, ey = (math.cos(self.robot_yaw), math.sin(self.robot_yaw))
        else:
            ex, ey = self.exploration_dir

        with np.errstate(divide='ignore', invalid='ignore'):
            ux = np.where(dist_euclid > 0, dx / dist_euclid, 0.0)
            uy = np.where(dist_euclid > 0, dy / dist_euclid, 0.0)

        dots = ux * ex + uy * ey  # directional alignment score
        dist_m = path_dists * float(self.map_res)

        dist_norm = np.clip(dist_m / (MAX_FRONTIER_DIST_M + 1e-6), 0.0, 1.0)
        scores = -dist_norm + 0.15 * dots        
        candidate_order = np.argsort(scores)[::-1]

        # Try each candidate in ranked order
        for idx in candidate_order:
            fx, fy = int(F[idx, 0]), int(F[idx, 1])
            frontier = (fx, fy)

            safe_target = self.find_safe_target_near_frontier(fx, fy)
            if safe_target is None:
                # mark failure
                self.failed_frontiers[frontier] = self.failed_frontiers.get(frontier, 0) + 1
                self._failed_timestamps[frontier] = now
                self.get_logger().info("Not safe target")
                continue

            goal = safe_target
            path = self.astar_grid(start, goal)

            if path:
                # success
                self.current_path = path
                # convert to world coords
                self.current_path_world = [self.grid_to_world(x, y) for x, y in path]
                # smooth path for better following (use small window)
                if self.path_has_clearance(self.current_path, min_clear_cells=2):
                    self.current_path_world = self.smooth_path(self.current_path_world, window=0)

                # store frontier (not safe target)
                self.path_goal = frontier

                # SET exploration direction (world-frame unit vector)
                dx = fx - rx
                dy = fy - ry
                norm = math.hypot(dx, dy)
                if norm > 1e-6:
                    self.exploration_dir = (dx / norm, dy / norm)

                # reset progress tracking
                self.last_progress_dist = None
                self.last_progress_time = now
                self.last_progress_path_seq = self.map_seq

                # self.get_logger().info(f"Path to frontier={frontier} via safe={safe_target}, len={len(path)}")
                # reset path following index
                self.last_path_idx = 0

                # convert grid path to world coordinates and publish Path
                path_msg = Path()
                path_msg.header = Header()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = "map"

                self.path_world = []
                for (gx_i, gy_i) in path:
                    x,y = self.grid_to_world(gx_i, gy_i)
                    ps = PoseStamped()
                    ps.header = path_msg.header
                    ps.pose.position.x = x
                    ps.pose.position.y = y
                    path_msg.poses.append(ps)
                    self.path_world.append((x,y))
                
                self.path_pub.publish(path_msg)
                self.get_logger().info(f"A* planned path with {len(self.path_world)} waypoints.")
                return True

            # A* failed → mark this frontier failed
            self.failed_frontiers[frontier] = self.failed_frontiers.get(frontier, 0) + 1
            self._failed_timestamps[frontier] = now
            self.get_logger().info("No path")

        self.get_logger().info("Reached end of planning")
        self.failed_frontiers = {} 
        self._failed_timestamps = {} 
        self.visited_mask = np.zeros_like(self.map_data, dtype=bool)
        self.visited_reset = time.time()

        return True
    

    def reset_path(self):
        self.current_path = []
        self.current_path_world = []
        if self.path_goal is not None:
            self.failed_frontiers[self.path_goal] = self.failed_frontiers.get(self.path_goal, 0) + 1
            self._failed_timestamps[self.path_goal] = time.time()
        self.path_goal = None
        self.last_path_idx = 0
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)


    def path_has_clearance(self, path, min_clear_cells=1):
        for gx, gy in path:
            y0 = max(0, gy - min_clear_cells)
            y1 = min(self.map_height, gy + min_clear_cells + 1)
            x0 = max(0, gx - min_clear_cells)
            x1 = min(self.map_width, gx + min_clear_cells + 1)
            if np.any(self.inflated_map[y0:y1, x0:x1] == 1):
                return False
        return True
    

    def find_nearest_free(self, gx, gy, max_r=6):
        for r in range(max_r+1):
            for dx in range(-r, r+1):
                for dy in (-r, r):
                    x, y = gx+dx, gy+dy
                    if 0 <= x < self.map_width and 0 <= y < self.map_height:
                        if self.inflated_map[y][x] == 0:
                            return (x, y)
            for dy in range(-r+1, r):
                for dx in (-r, r):
                    x, y = gx+dx, gy+dy
                    if 0 <= x < self.map_width and 0 <= y < self.map_height:
                        if self.inflated_map[y][x] == 0:
                            return (x, y)
        return (gx, gy)
    

    def find_safe_target_near_frontier(self, fx, fy, max_radius=25):
        if self.inflated_map is None:
            return None

        H, W = self.map_height, self.map_width

        # collect candidates in spiral out to max_radius
        candidates = []
        if self.is_cell_free_inflated(fx, fy):
            candidates.append((fx, fy))

        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    x, y = fx + dx, fy + dy
                    if 0 <= x < W and 0 <= y < H and self.is_cell_free_inflated(x, y):
                        candidates.append((x, y))
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    x, y = fx + dx, fy + dy
                    if 0 <= x < W and 0 <= y < H and self.is_cell_free_inflated(x, y):
                        candidates.append((x, y))

        if not candidates:
            self.get_logger().info(f"No canditates withing radius {max_radius}")
            return None

        # clearance function (small window)
        def clearance_at(cx, cy, search_r=6):
            x0 = max(0, cx - search_r); x1 = min(W, cx + search_r + 1)
            y0 = max(0, cy - search_r); y1 = min(H, cy + search_r + 1)
            sub = self.inflated_map[y0:y1, x0:x1]
            ys, xs = np.where(sub == 1)
            if len(xs) == 0:
                return float(search_r + 1)
            ys_global = ys + y0
            xs_global = xs + x0
            dxs = xs_global - cx
            dys = ys_global - cy
            dists = np.hypot(dxs, dys)
            return float(np.min(dists))

        # choose candidate using clearance then closeness
        best = None
        best_clr = -1.0
        best_dist = 1e9
        for c in candidates:
            c_clr = clearance_at(c[0], c[1], search_r=6)
            d_to_front = math.hypot(c[0]-fx, c[1]-fy)

            if best is None or d_to_front < best_dist:

                best_clr = c_clr
                best_dist = d_to_front
                best = c
            elif abs(c_clr - best_clr) <= 0.2:
                if d_to_front < best_dist:
                    best_dist = d_to_front
                    best = c

        return best
    

    # -----------------------------------------------------
    #               A* ON GRID (4-connected)
    # -----------------------------------------------------
    def astar_grid(self, start, goal):
        # start, goal are (gx,gy) tuples
        sx, sy = start
        gx, gy = goal
        H, W = self.map_height, self.map_width

        # bounds
        if not (0 <= gx < W and 0 <= gy < H):
            return None

        # Use inflated_map for occupancy checks (1 == occupied)
        M_infl = self.inflated_map
        # if goal is occupied in inflated map -> fail
        if M_infl[gy, gx] == 1:
            return None
        if M_infl[sy, sx] == 1:
            # start inside obstacle; fail gracefully
            return None

        open_heap = []
        heapq.heappush(open_heap, (0 + self.heuristic(start, goal), 0, start, None))
        came_from = {}
        gscore = {start: 0}
        closed = set()

        neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

        while open_heap:
            f, g, current, parent = heapq.heappop(open_heap)
            if current in closed:
                continue
            came_from[current] = parent
            if current == goal:
                # reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path  # list of (gx,gy)
            closed.add(current)

            cx, cy = current
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if M_infl[ny, nx] == 1:
                    continue
                neighbor = (nx, ny)

                if dx != 0 and dy != 0:
                    if M_infl[cy, nx] == 1 or M_infl[ny, cx] == 1:
                        continue

                penalty = self.near_wall_penalty(nx, ny)
                step_cost = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
                tentative_g = g + step_cost + penalty
                if tentative_g < gscore.get(neighbor, 1e9):
                    gscore[neighbor] = tentative_g
                    HEURISTIC_WEIGHT = 0.8
                    fscore = tentative_g + HEURISTIC_WEIGHT * self.heuristic(neighbor, goal)

                    heapq.heappush(open_heap, (fscore, tentative_g, neighbor, current))

        return None


    def heuristic(self, a, b):
        # Euclidean distance in grid cells
        return math.hypot(a[0]-b[0], a[1]-b[1])
    

    def near_wall_penalty(self, gx, gy):
        y0 = max(0, gy - 1)
        y1 = min(self.map_height, gy + 2)
        x0 = max(0, gx - 1)
        x1 = min(self.map_width, gx + 2)

        if np.any(self.inflated_map[y0:y1, x0:x1] == 1):
            return 1.2   # tune: 0.5–1.0 works well
        return 0.0
    
    # -----------------------------------------------------
    #      WALL FOLLOWER
    # -----------------------------------------------------
    def wall_follower(self):
        if self.front is None or self.left is None or self.right is None:
            return

        cmd = Twist()


        DESIRED = 0.60

        KP_WALL  = 0.8 
        KP_ANGLE = 1.4     
        KP_FRONT = 0.8

        MAX_W = 0.45
        DEADBAND = 0.06

        BASE_SPEED = 0.18
        MIN_SPEED  = 0.05

        WALL_LOST_DIST = 1.2      # no wall nearby
        WALL_LOST_ANGLE = 0.5    # front opens much more than back

        wall_dist = self.right
        wall_angle_err = self.right_front - self.right_back
        # + = pointing toward wall
        # - = pointing away

        error = DESIRED - wall_dist
        if abs(error) < DEADBAND:
            error = 0.0

        wall_lost = (self.right > WALL_LOST_DIST and self.right_front > WALL_LOST_DIST and
            wall_angle_err > WALL_LOST_ANGLE)

        if wall_lost:
            self.get_logger().info("Wall lost")
            cmd.linear.x = 0.12
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        w_wall  = KP_WALL * error
        w_angle = -KP_ANGLE * wall_angle_err
        w_front = 0.0

        if self.front < 1.0:
            w_front = KP_FRONT * (1.0 - self.front)

        w = w_wall + w_angle + w_front
        w = max(-MAX_W, min(MAX_W, w))
        cmd.angular.z = w

        speed = BASE_SPEED * max(0.3, 1.0 - abs(w) / MAX_W)

        if self.front < 0.7:
            speed *= self.front / 0.7

        cmd.linear.x = max(MIN_SPEED, speed)

        self.get_logger().info(f"WF: r={wall_dist:.2f} e={error:.2f} ang={wall_angle_err:.2f} w={w:.2f}")

        self.cmd_pub.publish(cmd)


    def start_wall_follow(self, duration=4.0):
        if not self.wall_following:
            self.get_logger().warn("Entering wall-follow recovery")
            self.wall_following = True
            self.wall_follow_end_time = time.time() + duration
            self.reset_path()


    # -----------------------------------------------------
    #                PATH SMOOTHING / FOLLOWING
    # -----------------------------------------------------
    def smooth_path(self, world_path, window=3):
        if not world_path:
            return []

        n = len(world_path)
        smoothed = []
        for i in range(n):
            x_sum = 0.0
            y_sum = 0.0
            count = 0
            for j in range(max(0, i-window), min(n, i+window+1)):
                x_sum += world_path[j][0]
                y_sum += world_path[j][1]
                count += 1
            smoothed.append((x_sum / count, y_sum / count))
        return smoothed


    def get_path_idx(self, path_world, last_idx=0):
        if not path_world:
            return 0
        vx = self.robot_x
        vy = self.robot_y
        lookahead = 0.3  # meters
        n = len(path_world)
        for i in range(last_idx, n):
            px, py = path_world[i]
            dist = math.hypot(px - vx, py - vy)
            if dist > lookahead:
                return i
        return n - 1


    def quat_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)


    def path_follower_from_xy(self, goal_x, goal_y):
        vx = self.robot_x
        vy = self.robot_y
        gx = goal_x
        gy = goal_y

        yaw = self.robot_yaw

        desired_yaw = math.atan2(gy - vy, gx - vx)

        # Compute heading error
        heading_error = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        # distance to goal
        dist = math.hypot(gx - vx, gy - vy)

        # Proportional control variables
        Kp_ang = 1.2   
        Kd_ang = 0.15  
        Kp_lin = 1.0   

        # heading control
        heading = Kp_ang * heading_error
        # clamp angular
        heading = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, heading))

        # reduce speed when heading error is large
        cos_term = math.cos(heading_error)
        # ensure non-negative
        if cos_term < 0:
            cos_term = 0.0
        speed = max(0.03, min(MAX_LINEAR_SPEED, Kp_lin * dist * cos_term))
        
        # If obstacle close in front, slow more
        if hasattr(self, "front") and self.front < 1.0:
            speed = speed * self.front

        return speed, heading, dist
   

    # -----------------------------------------------------
    #                CONTROL ROBOT
    # -----------------------------------------------------
    def move_ttbot_safe(self, speed, heading):
        cmd = Twist()

        # ---------------- EMERGENCY 180 ----------------
        if self.front_check is not None and self.front_check < 0.4:

            if not self.turning_around:
                self.get_logger().info("Tlose: turning 180°")
                self.turning_around = True
                self.turn_end_time = time.time() + 3.1  # 3.1s ~180° at 0.6 rad/s
                self.reset_path()

            # rotate in place
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6
            self.cmd_pub.publish(cmd)
            return

        # finish rotation
        if self.turning_around:
            if time.time() >= self.turn_end_time:
                self.turning_around = False
                self.get_logger().info("✅ Finished 180° turn")
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.6
                self.cmd_pub.publish(cmd)
                return
            
        """if self.right is not None and self.right < side_dist:
            self.get_logger().info("Too close on right!")
            self.reset_path()
            speed = 0.0
            heading = turning_speed
        if self.left is not None and self.left < side_dist:
            self.get_logger().info("Too close on left!")
            self.reset_path()
            speed = 0.0
            heading = -1 * turning_speed"""

        # ---------------- NORMAL SAFE CLAMP ----------------
        cmd.linear.x = max(0.0, min(speed, MAX_LINEAR_SPEED))
        cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(heading, MAX_ANGULAR_SPEED))
        self.cmd_pub.publish(cmd)


    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    # -----------------------------------------------------
    #      PUBLISH MAP
    # -----------------------------------------------------
    def publish_working_map(self, map):
        if map is None:
            return

        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.info.resolution = self.map_res
        msg.info.width  = map.shape[1]
        msg.info.height = map.shape[0]

        msg.info.origin.position.x = self.map_origin[0]
        msg.info.origin.position.y = self.map_origin[1]
        msg.info.origin.orientation.w = 1.0

        msg.data = (map.flatten() * 100).astype(np.int8).tolist()

        self.work_map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Task1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()