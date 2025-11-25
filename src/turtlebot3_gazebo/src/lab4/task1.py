#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import time
import heapq
from collections import deque


# ---------- TUNABLE PARAMETERS ----------
OCCUPIED_THRESH = 50          # occupancy grid value >= this is considered an obstacle
UNKNOWN_VAL = -1
INFLATE_RADIUS = 4.2            # smaller by default; adjust if robot footprint requires
GOAL_NEAR_DIST_M = 0.15       # when within this world distance to goal waypoint, pop waypoint
ALIGN_ANGLE = 0.20            # radians: how close to aligned before driving forward
MAX_LINEAR_SPEED = 0.4
MAX_ANGULAR_SPEED = 0.6
OBSTACLE_FRONT_THRESH = 0.40  # meters
REPLAN_INTERVAL = 1.0         # seconds between forced replans
STUCK_TIME = 4.0              # seconds to consider stuck
STUCK_MOVE_THRESH = 0.05      # meters moved to consider not stuck
BACKUP_TIME = 0.6             # seconds to back up on recovery
RECOVERY_ROTATE_TIME = 1.0    # seconds to rotate during recovery
MIN_FRONTIER_GRID_DIST = 6    # minimum grid cells away from robot to accept frontier
# ----------------------------------------


class Task1(Node):

    def __init__(self):
        super().__init__('task1_algorithm')

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
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
        self.left = 10.0
        self.right = 10.0
        self.ranges_full = None

        # Planner state
        self.current_path = []          # list of waypoints in grid (gx,gy)
        self.current_path_world = []    # list of waypoints in world coords (wx,wy)
        self.path_goal = None           # grid goal (gx,gy)
        self.last_plan_time = 0.0

        # Stuck detection
        self.last_pos = (0.0, 0.0)
        self.last_movement_time = time.time()
        self.last_cmd = Twist()

        # oscillation detection (simple sign-flip detector)
        self.last_turn_sign = 0
        self.osc_count = 0
        self.OSCILLATION_THRESHOLD = 4
        self.OSCILLATION_COOLDOWN = 3.0
        self.osc_last_time = 0.0
        self.oscillating = False
        self.osc_end = 0.0

        # Recovery
        self.recovering = False
        self.recovery_end_time = 0.0
        self.recovery_stage = None  # "backup" or "rotate"

        # Exploration bias: persistent exploration direction in world coords (unit vector)
        # Set to None initially; once we pick a first frontier we store the direction to keep going.
        self.exploration_dir = None
        self.EXPLORE_ANGLE_LIMIT = math.radians(60)   # cone +/- 60 degrees

        # frontier visit bookkeeping
        self.visited_mask = None           # numpy bool array same shape as map
        self.VISITED_RADIUS = 10            # radius in grid cells to mark visited around goals (tune)
        self.failed_frontiers = {}         # dict {(gx,gy): fail_count}
        self.FAILED_LIMIT = 3              # after this many A* fails, temporarily ignore frontier
        self.FAILED_CLEAR_TIME = 15.0      # seconds after which failed_frontiers entry may be dropped
        self._failed_timestamps = {}       # {(gx,gy): last_fail_time}

        # path-following helper
        self.last_path_idx = 0
        self.last_follow_time = time.time()
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
            return

        # ---------------- STUCK CHECK ----------------
        if self._check_stuck(now):
            return

        # ---------------- FOLLOW CURRENT PATH ----------------
        if self.current_path and self.current_path_world:
            # Use the improved follower from the Navigation lab
            idx = self.get_path_idx(self.current_path_world, self.last_path_idx)
            idx = max(0, min(idx, len(self.current_path_world)-1))
            current_goal = self.current_path_world[idx]
            speed, heading, dist = self.path_follower_from_xy(current_goal[0], current_goal[1])
            # publish safe command
            self.move_ttbot_safe(speed, heading)
            self.last_cmd = Twist(); self.last_cmd.linear.x = speed; self.last_cmd.angular.z = heading
            self.last_path_idx = idx

            # If at end and close enough -> finish
            if idx >= len(self.current_path_world) - 1 and dist < GOAL_NEAR_DIST_M:
                self.get_logger().info("✅ Reached path goal grid={}, clearing path".format(self.path_goal))
                gx, gy = self.path_goal if self.path_goal is not None else (None, None)
                self.path_goal = None

                # mark visited area on the visited_mask to avoid reselecting nearby frontiers
                if self.visited_mask is not None and gx is not None:
                    rr = self.VISITED_RADIUS
                    H, W = self.map_height, self.map_width
                    x0 = max(0, gx - rr); x1 = min(W, gx + rr + 1)
                    y0 = max(0, gy - rr); y1 = min(H, gy + rr + 1)
                    xs = np.arange(x0, x1)
                    ys = np.arange(y0, y1)
                    # efficient disk mask
                    dx = xs.reshape(1, -1) - gx
                    dy = ys.reshape(-1, 1) - gy
                    d2 = dx*dx + dy*dy
                    disk = d2 <= (rr*rr)
                    # assign into visited_mask (note indexing visited_mask[y,x])
                    self.visited_mask[y0:y1, x0:x1] |= disk

                # reset exploration_dir (keep this)
                self.exploration_dir = None

                # clear path lists
                self.current_path = []
                self.current_path_world = []
                self.last_path_idx = 0

            return

        # ---------------- PLAN NEW PATH ----------------
        if now - self.last_plan_time > REPLAN_INTERVAL or not self.current_path:
            if not self.plan_to_frontier(farthest=False):
                self.get_logger().info("🎉 Map complete (or no reachable frontiers). Stopping.")
                self.cmd_pub.publish(Twist())
                return
            # when we set current_path_world in plan_to_frontier we will smooth it there
            self.last_plan_time = now

        # Default: no motion
        self.cmd_pub.publish(Twist())

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
        self.get_logger().info("🔁 Recovery finished, replanning.")
        self._reset_path()
        return False
    
    def _handle_oscillation(self, now):
        if not self.oscillating:
            return False

        if now < self.osc_end:
            cmd = Twist(); cmd.linear.x = 0.12
            self.cmd_pub.publish(cmd)
            return True

        self.oscillating = False
        return False
    
    def _publish_rotate(self):
        cmd = Twist()
        cmd.angular.z = 0.4
        self.cmd_pub.publish(cmd)

    def _check_stuck(self, now):
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
            self.get_logger().warning("🛑 Stuck detected -> performing recovery")
            self.start_recovery()
            rc = self.recovery_cmd()
            self.cmd_pub.publish(rc)
            self.last_cmd = rc
            return True

        return False
    
    def _reset_path(self):
        self.current_path = []
        self.current_path_world = []
        self.path_goal = None
        self.last_path_idx = 0

    # ---------------- LIDAR ----------------
    def scan_callback_old(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        self.ranges_full = ranges
        N = len(ranges)
        # front: central ±10-12% (robust to different lidar indexing)
        i0 = int(N * 0.44); i1 = int(N * 0.56)
        sector = ranges[i0:i1]
        self.front = float(np.percentile(sector, 5))  # ignores single outliers
        """if i0 <= i1:
            self.front = float(np.min(ranges[i0:i1]))
        else:
            self.front = float(min(np.min(ranges[i0:]), np.min(ranges[:i1])))"""
        # left/right sectors (20%-35% and 65%-80%)
        self.left = float(np.min(ranges[int(N*0.20):int(N*0.35)]))
        self.right = float(np.min(ranges[int(N*0.65):int(N*0.80)]))
    
    # ---------------- LIDAR ----------------
    def scan_callback_old2(self, msg):
        ranges = np.array(msg.ranges, dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        self.ranges_full = ranges
        N = len(ranges)

        # FRONT sector: central 12% (good for TB3)
        i0 = int(0.44 * N)
        i1 = int(0.56 * N)
        front_sector = ranges[i0:i1]
        # ignore outliers → more stable
        self.front = float(np.percentile(front_sector, 5))

        # SIDE sectors (use percentiles for stability)
        left_sec  = ranges[int(0.20*N):int(0.35*N)]
        right_sec = ranges[int(0.65*N):int(0.80*N)]

        self.left  = float(np.percentile(left_sec,  10))
        self.right = float(np.percentile(right_sec, 10))

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

        # Define cardinal angles
        FRONT = 0.0
        LEFT  = math.pi/2
        RIGHT = -math.pi/2

        w = math.radians(12)   # ±12° window

        def median_in_sector(center_angle):
            idx0 = angle_to_idx(center_angle - w)
            idx1 = angle_to_idx(center_angle + w)

            # clip indices
            idx0 %= N
            idx1 %= N

            if idx0 <= idx1:
                sector = ranges[idx0:idx1]
            else:
                # wrap-around case (e.g., crossing the 0° boundary)
                sector = np.concatenate((ranges[idx0:], ranges[:idx1]))

            return float(np.median(sector))

        self.front = median_in_sector(FRONT)
        self.left  = median_in_sector(LEFT)
        self.right = median_in_sector(RIGHT)



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
        # msg.data is a flat list row-major. reshape to [height, width]
        self.map_data = np.array(msg.data, dtype=int).reshape(msg.info.height, msg.info.width)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_res = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        # bump seq to notice map changes
        self.map_seq += 1

        # build inflated map (binary grid) whenever map updates
        self.inflated_map = self.build_inflated_map(self.map_data, INFLATE_RADIUS)

        # initialize visited_mask to False with same shape as map
        if self.visited_mask is None:
            self.visited_mask = np.zeros_like(self.map_data, dtype=bool)
        else:
            # Resize mask if map grows (SLAM expanding map)
            if self.visited_mask.shape != self.map_data.shape:
                new_mask = np.zeros_like(self.map_data, dtype=bool)
                H_old, W_old = self.visited_mask.shape
                H_new, W_new = new_mask.shape
                h = min(H_old, H_new)
                w = min(W_old, W_new)
                new_mask[:h, :w] = self.visited_mask[:h, :w]
                self.visited_mask = new_mask



    # ---------------- INFLATION ----------------
    def build_inflated_map(self, M, radius_cells):
        H, W = M.shape
        binmap = np.zeros((H, W), dtype=np.uint8)
        binmap[M >= OCCUPIED_THRESH] = 1

        if radius_cells <= 0:
            return binmap

        # Simple symmetric BFS distance-based inflation
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


    # ---------------- FRONTIER / PLANNING HELPERS ----------------
    def get_frontier_cells(self):
        """
        Return list of free cells (x,y) that border unknown cells.
        free cell: map_data[y,x] == 0
        unknown cell: map_data[y,x] == UNKNOWN_VAL
        """
        if self.map_data is None:
            return []

        frontiers = []
        M = self.map_data
        H, W = M.shape

        # don't check border cells (1..H-2, 1..W-2) for safety
        for y in range(1, H-1):
            for x in range(1, W-1):
                # treat 0..30 as free-ish for SLAM maps
                if not (0 <= M[y, x] <= 30):
                    continue
                # check 8-neighborhood for unknown
                neigh = M[y-1:y+2, x-1:x+2]
                if np.any(neigh == UNKNOWN_VAL):
                    # avoid frontiers touching inflated obstacle
                    if self.inflated_map is not None and np.any(self.inflated_map[y-1:y+2, x-1:x+2] == 1):
                        continue
                    frontiers.append((x, y))
        return frontiers

    def is_cell_free_inflated(self, gx, gy):
        """Return True if inflated_map considers this cell free (0)."""
        if self.inflated_map is None:
            return False
        if not (0 <= gx < self.map_width and 0 <= gy < self.map_height):
            return False
        return self.inflated_map[gy, gx] == 0

    # -----------------------------------------------------
    #      PLANNING: choose reachable frontier -> A*
    # -----------------------------------------------------
    def plan_to_frontier(self, farthest=False):
        M = self.map_data
        H, W = M.shape
        if self.map_data is None or self.inflated_map is None:
            return False

        frontiers = self.get_frontier_cells()
        self.get_logger().info(f"Frontiers found: {len(frontiers)}")
        if not frontiers:
            return False
        
        # Filter visited & failed frontiers
        filtered_frontiers = []
        now = time.time()

        for fx, fy in frontiers:
            # skip if visited
            if self.visited_mask is not None and self.visited_mask[fy, fx]:
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
                    continue

            filtered_frontiers.append((fx, fy))

        if not filtered_frontiers:
            return False

        # Robot grid
        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)
        F = np.array(filtered_frontiers, dtype=float)
        dx = F[:, 0] - rx
        dy = F[:, 1] - ry
        dist = np.hypot(dx, dy)

        # Direction preference
        if self.exploration_dir is None:
            ex, ey = (math.cos(self.robot_yaw), math.sin(self.robot_yaw))
        else:
            ex, ey = self.exploration_dir

        with np.errstate(divide='ignore', invalid='ignore'):
            ux = np.where(dist > 0, dx / dist, 0.0)
            uy = np.where(dist > 0, dy / dist, 0.0)

        dots = ux * ex + uy * ey  # directional alignment score

        # -----------------------
        # NEW SCORING: prefer aligned & nearer frontiers
        # -----------------------
        # parameters — tune if necessary
        MAX_FRONTIER_DIST_M = 10.0     # ignore frontiers farther than this (meters)
        LAMBDA_DIST = 0.6             # distance penalty weight [0..1]

        # distance in meters
        dist_m = dist * float(self.map_res)

        # normalized distance [0..1]
        dist_norm = np.clip(dist_m / (MAX_FRONTIER_DIST_M + 1e-6), 0.0, 1.0)

        # score: favor alignment, penalize distance.
        # Range roughly in [-1 - lambda, 1]
        scores = dots - LAMBDA_DIST * dist_norm

        # Hard reject: drop frontiers that are extremely far (optional)
        far_mask = dist_m <= MAX_FRONTIER_DIST_M

        # Keep ahead/weak logic based on alignment (dots)
        cos_limit = math.cos(self.EXPLORE_ANGLE_LIMIT)
        ahead_idxs = np.where(dots >= cos_limit)[0]
        weak_idxs  = np.where(dots >= 0.0)[0]

        # Candidate pools intersect with far_mask to remove extremely far ones
        ahead_idxs = np.array([i for i in ahead_idxs if far_mask[i]], dtype=int)
        weak_idxs  = np.array([i for i in weak_idxs  if far_mask[i]], dtype=int)

        # rebuild sets AFTER filtering
        ahead_set = set(ahead_idxs.tolist())
        weak_set  = set(weak_idxs.tolist())

        other_idxs = np.array(
            [i for i in range(len(F)) if far_mask[i] and i not in ahead_set and i not in weak_set],
            dtype=int
        )

        # If everything is empty, no reachable frontiers remain
        if ahead_idxs.size == 0 and weak_idxs.size == 0 and other_idxs.size == 0:
            self.get_logger().warn("No valid frontier indices remain after filtering.")
            return False

        # choose ordering by score (descending)
        if ahead_idxs.size > 0:
            candidate_order = ahead_idxs[np.argsort(scores[ahead_idxs])[::-1]]
        elif weak_idxs.size > 0:
            candidate_order = weak_idxs[np.argsort(scores[weak_idxs])[::-1]]
        else:
            candidate_order = other_idxs[np.argsort(scores[other_idxs])[::-1]]

            

        # Try each candidate in ranked order
        for idx in candidate_order:
            fx, fy = int(F[idx, 0]), int(F[idx, 1])
            frontier = (fx, fy)

            # FIND SAFE TARGET
            safe_target = self.find_safe_target_near_frontier(fx, fy)
            if safe_target is None:
                # mark failure
                self.failed_frontiers[frontier] = self.failed_frontiers.get(frontier, 0) + 1
                self._failed_timestamps[frontier] = now
                continue

            # Now run A*
            start = (rx, ry)
            goal = safe_target
            path = self.astar_grid(start, goal)

            if path:
                # success
                self.current_path = path
                # convert to world coords
                self.current_path_world = [self.grid_to_world(x, y) for x, y in path]
                # smooth path for better following (use small window)
                self.current_path_world = self.smooth_path(self.current_path_world, window=1)

                # store frontier (not safe target)
                self.path_goal = frontier

                # reset progress tracking
                self.last_progress_dist = None
                self.last_progress_time = now
                self.last_progress_path_seq = self.map_seq

                self.get_logger().info(f"📍 Path to frontier={frontier} via safe={safe_target}, len={len(path)}")
                # reset path following index
                self.last_path_idx = 0
                return True

            # A* failed → mark this frontier failed
            self.failed_frontiers[frontier] = self.failed_frontiers.get(frontier, 0) + 1
            self._failed_timestamps[frontier] = now

        # Fallback: try all by distance
        order2 = np.argsort(dist)
        for idx in order2:
            fx, fy = int(F[idx, 0]), int(F[idx, 1])
            frontier = (fx, fy)

            safe_target = self.find_safe_target_near_frontier(fx, fy)
            if safe_target is None:
                continue

            path = self.astar_grid((rx, ry), safe_target)
            if not path:
                continue

            self.current_path = path
            self.current_path_world = [self.grid_to_world(x, y) for x, y in path]
            self.current_path_world = self.smooth_path(self.current_path_world, window=3)
            self.path_goal = frontier
            self.last_progress_dist = None
            self.last_progress_time = now
            self.last_progress_path_seq = self.map_seq

            self.get_logger().info(f"📍 Fallback path to frontier={frontier} safe={safe_target}")
            self.last_path_idx = 0
            return True

        return False


    def find_safe_target_near_frontier(self, fx, fy, max_radius=6):
        """
        Search the inflated map around the frontier (fx, fy)
        for the nearest safe navigable cell.
        This prevents the robot from trying to drive into narrow or unsafe spots.
        Preference is given to candidates with larger clearance (but ties break to closeness).
        """

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
            # choose if significantly better clearance
            if c_clr > best_clr + 0.2:
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

        neighbors = [(1,0), (-1,0), (0,1), (0,-1)]

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
                tentative_g = g + 1
                if tentative_g < gscore.get(neighbor, 1e9):
                    gscore[neighbor] = tentative_g
                    fscore = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (fscore, tentative_g, neighbor, current))

        return None

    def heuristic(self, a, b):
        # Euclidean distance in grid cells
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # -----------------------------------------------------
    #                PATH SMOOTHING / FOLLOWING
    # -----------------------------------------------------
    def smooth_path(self, world_path, window=3):
        """
        world_path: list of (x, y) tuples in world coords
        returns a new list of (x,y) smoothed by sliding-average
        """
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
        """
        Return next path index based on lookahead distance.
        path_world: list of (x,y)
        """
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
        """Convert quaternion to yaw (safe for ROS standard axes)."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def path_follower_from_xy(self, goal_x, goal_y):
        """
        Simple PID-like path follower adapted from Navigation.path_follower.
        Returns (speed, heading, dist_to_goal)
        """
        vx = self.robot_x
        vy = self.robot_y
        gx = goal_x
        gy = goal_y

        yaw = self.robot_yaw

        desired_yaw = math.atan2(gy - vy, gx - vx)

        # Compute heading error safely
        heading_error = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        # distance to goal
        dist = math.hypot(gx - vx, gy - vy)

        # PID-like control (tuned)
        Kp_ang = 1.2   # proportional gain for heading
        Kd_ang = 0.15  # derivative gain (unused here, but available)
        Kp_lin = 0.7   # linear scaling

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
        if hasattr(self, "front") and self.front < 0.35:
            speed = min(speed, 0.06)

        return speed, heading, dist

    def move_ttbot(self, speed, heading):
        """Publish Twist using speed and heading (heading here is angular.z command)."""
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = heading
        self.cmd_pub.publish(cmd)

    def move_ttbot_safe(self, speed, heading):
        """Publish Twist but clamp to safe ranges."""
        cmd = Twist()
        cmd.linear.x = max(0.0, min(speed, MAX_LINEAR_SPEED))
        cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(heading, MAX_ANGULAR_SPEED))
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


    # -----------------------------------------------------
    #                RECOVERY BEHAVIOR
    # -----------------------------------------------------
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


    # -----------------------------------------------------
    #             MAP <-> WORLD helpers
    # -----------------------------------------------------
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