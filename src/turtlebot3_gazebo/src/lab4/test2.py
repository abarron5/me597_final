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
OCCUPIED_THRESH = 50
UNKNOWN_VAL = -1
INFLATE_RADIUS = 4
GOAL_NEAR_DIST_M = 0.15
ALIGN_ANGLE = 0.20
MAX_LINEAR_SPEED = 0.4
MAX_ANGULAR_SPEED = 0.6
OBSTACLE_FRONT_THRESH = 0.40
REPLAN_INTERVAL = 1.0
STUCK_TIME = 4.0
STUCK_MOVE_THRESH = 0.02
BACKUP_TIME = 0.0
RECOVERY_ROTATE_TIME = 1.0
MIN_FRONTIER_GRID_DIST = 0
# ----------------------------------------

class Task1(Node):
    def __init__(self):
        super().__init__('task1_node')

        # subscriptions & pubs
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_cb)

        # robot state
        self.robot_x = 0.0; self.robot_y = 0.0; self.robot_yaw = 0.0
        self.map_data = None
        self.map_width = 0; self.map_height = 0; self.map_res = 0.05
        self.map_origin = (0.0, 0.0)
        self.map_seq = 0
        self.inflated_map = None

        # lidar summary values (updated by scan_callback)
        self.front = 10.0; self.left = 10.0; self.right = 10.0
        self.ranges_full = None

        # planner / path
        self.current_path = []
        self.current_path_world = []
        self.path_goal = None
        self.last_plan_time = 0.0

        # visited / failed bookkeeping
        self.visited_mask = None
        self.VISITED_RADIUS = 10
        self.failed_frontiers = {}
        self._failed_timestamps = {}
        self.FAILED_LIMIT = 3
        self.FAILED_CLEAR_TIME = 60.0

        # recovery & stuck
        self.recovering = False
        self.recovery_end_time = 0.0
        self.recovery_stage = None
        self.last_cmd = Twist()
        self.last_pos = (0.0, 0.0)
        self.last_movement_time = time.time()

        # path-progress stuck tracking
        self.last_progress_dist = None
        self.last_progress_time = None
        self.last_progress_path_seq = -1

        # exploration bias
        self.exploration_dir = None
        self.EXPLORE_ANGLE_LIMIT = math.radians(60)

        # oscillation detection (simple sign-flip detector)
        self.last_turn_sign = 0
        self.osc_count = 0
        self.OSCILLATION_THRESHOLD = 4
        self.OSCILLATION_COOLDOWN = 3.0
        self.osc_last_time = 0.0
        self.oscillating = False
        self.osc_end = 0.0

        # short push timer used for doorway/hallway pushes
        self.push_timer = None

        self.get_logger().info("Task 1 node started (cleaned).")

    # ---------------- TIMER LOOP ----------------
    def timer_cb(self):
        now = time.time()

        # if in recovery, run recovery
        if self.recovering:
            if now < self.recovery_end_time:
                self.cmd_pub.publish(self.recovery_cmd())
                return
            else:
                self.recovering = False
                self.recovery_stage = None
                self.get_logger().info("🔁 Recovery finished, replanning.")
                self.current_path = []
                self.current_path_world = []
                self.path_goal = None
                self.exploration_dir = None

        # oscillation breakout (if previously triggered)
        if self.oscillating:
            if time.time() < self.osc_end:
                cmd = Twist(); cmd.linear.x = 0.12
                self.cmd_pub.publish(cmd); return
            else:
                self.oscillating = False

        # if no map, rotate slowly to help SLAM
        if self.map_data is None:
            cmd = Twist(); cmd.angular.z = 0.4
            self.cmd_pub.publish(cmd); return

        # PATH-PROGRESS BASED STUCK CHECK
        if self.current_path_world and len(self.current_path_world) > 0:
            wx, wy = self.current_path_world[0]
            dist_to_wp = math.hypot(wx - self.robot_x, wy - self.robot_y)

            # initialize trackers when path or map changes
            if self.last_progress_path_seq != self.map_seq or self.last_progress_dist is None:
                self.last_progress_dist = dist_to_wp
                self.last_progress_time = now
                self.last_progress_path_seq = self.map_seq

            # if we are commanding forward, require progress in STUCK_TIME
            if getattr(self.last_cmd, "linear", None) and self.last_cmd.linear.x > 0.03:
                if dist_to_wp + 1e-6 < (self.last_progress_dist - STUCK_MOVE_THRESH):
                    self.last_progress_dist = dist_to_wp
                    self.last_progress_time = now
                else:
                    if now - self.last_progress_time > STUCK_TIME:
                        self.get_logger().warning(f"🛑 Path-progress stuck (dist_to_wp={dist_to_wp:.3f}) -> recovery")
                        # mark current goal as failed to avoid immediate replan to same spot
                        if self.path_goal is not None:
                            key = (int(self.path_goal[0]), int(self.path_goal[1]))
                            self.failed_frontiers[key] = self.failed_frontiers.get(key, 0) + 1
                            self._failed_timestamps[key] = now
                            self.get_logger().info(f"Marked frontier {key} failed count={self.failed_frontiers[key]}")
                        self.start_recovery()
                        self.cmd_pub.publish(self.recovery_cmd())
                        self.last_cmd = self.recovery_cmd()
                        # reset progress trackers
                        self.last_progress_time = now
                        return

        # if following a path, compute and publish follow command
        if self.current_path and len(self.current_path_world) > 0:
            cmd = self.follow_path()
            # debug log for diagnostics (useful when running)
            try:
                self.get_logger().debug(f"FOLLOW cmd lin={cmd.linear.x:.3f} ang={cmd.angular.z:.3f} front={self.front:.2f} left={self.left:.2f} right={self.right:.2f} goal={self.path_goal} path_len={len(self.current_path)}")
            except Exception:
                pass
            self.cmd_pub.publish(cmd)
            self.last_cmd = cmd
            return

        # otherwise plan to frontier
        if time.time() - self.last_plan_time > REPLAN_INTERVAL or not self.current_path:
            planned = self.plan_to_frontier(farthest=False)
            self.last_plan_time = time.time()
            if not planned:
                self.get_logger().info("🎉 Map complete (or no reachable frontiers). Stopping.")
                self.cmd_pub.publish(Twist())
                return

        # default: stop
        self.cmd_pub.publish(Twist())

    # ---------------- LIDAR ----------------
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        self.ranges_full = ranges
        N = len(ranges)
        # front: central ±10-12% (robust to different lidar indexing)
        i0 = int(N * 0.44); i1 = int(N * 0.56)
        if i0 <= i1:
            self.front = float(np.min(ranges[i0:i1]))
        else:
            self.front = float(min(np.min(ranges[i0:]), np.min(ranges[:i1])))
        # left/right sectors (20%-35% and 65%-80%)
        self.left = float(np.min(ranges[int(N*0.20):int(N*0.35)]))
        self.right = float(np.min(ranges[int(N*0.65):int(N*0.80)]))

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
        # reshape to [height, width]
        self.map_data = np.array(msg.data, dtype=int).reshape(msg.info.height, msg.info.width)
        self.map_width = msg.info.width; self.map_height = msg.info.height
        self.map_res = msg.info.resolution; self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_seq += 1
        self.inflated_map = self.build_inflated_map(self.map_data, INFLATE_RADIUS)

        # init/resize visited_mask
        if self.visited_mask is None:
            self.visited_mask = np.zeros_like(self.map_data, dtype=bool)
        else:
            if self.visited_mask.shape != self.map_data.shape:
                new_mask = np.zeros_like(self.map_data, dtype=bool)
                h = min(new_mask.shape[0], self.visited_mask.shape[0])
                w = min(new_mask.shape[1], self.visited_mask.shape[1])
                new_mask[:h, :w] = self.visited_mask[:h, :w]
                self.visited_mask = new_mask

    # ---------------- INFLATION ----------------
    def build_inflated_map(self, M, radius_cells):
        H, W = M.shape
        binmap = np.zeros((H, W), dtype=np.uint8)
        binmap[M >= OCCUPIED_THRESH] = 1

        if radius_cells <= 0:
            return binmap

        # corridor detection: cell is in narrow corridor when few free neighbors
        corridor_mask = np.zeros_like(binmap, dtype=bool)
        for y in range(1, H-1):
            for x in range(1, W-1):
                if binmap[y,x] == 0:
                    # count free in 4-neighbors
                    free_count = np.sum(binmap[y-1:y+2, x-1:x+2] == 0)
                    if free_count <= 5:
                        corridor_mask[y,x] = True

        inflated = binmap.copy()
        dist = np.full((H, W), -1, dtype=int)
        q = deque()
        ys, xs = np.where(binmap == 1)
        for y, x in zip(ys, xs):
            q.append((y, x)); dist[y, x] = 0

        offs = [(1,0), (-1,0), (0,1), (0,-1), (-1,-1), (-1,1), (1,-1), (1,1)]
        while q:
            y, x = q.popleft()
            d = dist[y, x]
            if d >= radius_cells:
                continue
            for dy, dx in offs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] == -1:
                    dist[ny, nx] = d + 1
                    # reduce inflation inside corridor cells (allow narrow passage)
                    if not corridor_mask[ny, nx]:
                        inflated[ny, nx] = 1
                    q.append((ny, nx))
        return inflated

    # ---------------- FRONTIER HELPERS ----------------
    def get_frontier_cells(self):
        if self.map_data is None:
            return []
        frontiers = []
        M = self.map_data; H, W = M.shape
        for y in range(1, H-1):
            for x in range(1, W-1):
                if M[y,x] != 0:
                    continue
                neigh = M[y-1:y+2, x-1:x+2]
                if np.any(neigh == UNKNOWN_VAL):
                    frontiers.append((x, y))
        return frontiers

    def is_cell_free_inflated(self, gx, gy):
        if self.inflated_map is None:
            return False
        if not (0 <= gx < self.map_width and 0 <= gy < self.map_height):
            return False
        return self.inflated_map[gy, gx] == 0

    def detect_oscillation(self, ang_z):
        # count sign flips ignoring small turns
        if abs(ang_z) < 0.12:
            self.osc_count = 0
            self.last_turn_sign = 0
            return False
        sign = 1 if ang_z > 0 else -1
        if self.last_turn_sign != 0 and sign != self.last_turn_sign:
            self.osc_count += 1
        else:
            self.osc_count = 0
        self.last_turn_sign = sign
        if self.osc_count >= self.OSCILLATION_THRESHOLD:
            now = time.time()
            if now - self.osc_last_time > self.OSCILLATION_COOLDOWN:
                self.osc_last_time = now
                self.osc_count = 0
                return True
        return False

    # -----------------------------------------------------
    # PLANNING: choose reachable frontier -> A*
    # -----------------------------------------------------
    def plan_to_frontier(self, farthest=False):
        if self.map_data is None or self.inflated_map is None:
            return False

        frontiers = self.get_frontier_cells()
        if not frontiers:
            return False

        # filter visited and failed frontiers
        filtered = []
        for fx, fy in frontiers:
            if self.visited_mask is not None and self.visited_mask[fy, fx]:
                continue
            fail_count = self.failed_frontiers.get((fx,fy), 0)
            if fail_count >= self.FAILED_LIMIT:
                # allow clearing stale entries
                ts = self._failed_timestamps.get((fx,fy), 0)
                if time.time() - ts > self.FAILED_CLEAR_TIME:
                    self.failed_frontiers.pop((fx,fy), None)
                    self._failed_timestamps.pop((fx,fy), None)
                else:
                    continue
            filtered.append((fx,fy))
        frontiers = filtered
        if not frontiers:
            return False

        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)
        self.get_logger().info(f"Found frontiers: {len(frontiers)}  Inflated occupied: {np.sum(self.inflated_map==1)}  Robot grid={(rx,ry)}")

        # vectorized arrays
        F = np.array(frontiers, dtype=float)
        dx = F[:,0] - rx; dy = F[:,1] - ry
        dist = np.hypot(dx, dy)
        with np.errstate(invalid='ignore'):
            ux = np.where(dist>0, dx/dist, 0.0)
            uy = np.where(dist>0, dy/dist, 0.0)

        # exploration direction: prefer persistent dir, otherwise heading
        if self.exploration_dir is None:
            ex, ey = (math.cos(self.robot_yaw), math.sin(self.robot_yaw))
        else:
            ex, ey = self.exploration_dir

        dots = ux*ex + uy*ey
        maxd = max(1.0, float(np.max(dist))) if len(dist)>0 else 1.0
        dist_norm = dist / maxd
        scores = dots * (1.0 + dist_norm)

        cos_angle_limit = math.cos(self.EXPLORE_ANGLE_LIMIT)
        ahead_idxs = np.where(dots >= cos_angle_limit)[0]
        weak_idxs = np.where(dots >= 0.0)[0]

        # If the robot faces a near wall (front small), relax the angular requirement:
        if hasattr(self, "front") and self.front < 0.6:
            # include more candidates (avoid thinking wall means done)
            candidate_idxs = np.argsort(scores)[::-1]
        else:
            if ahead_idxs.size > 0:
                candidate_idxs = ahead_idxs[np.argsort(scores[ahead_idxs])[::-1]]
            elif weak_idxs.size > 0:
                candidate_idxs = weak_idxs[np.argsort(scores[weak_idxs])[::-1]]
            else:
                candidate_idxs = np.argsort(scores)[::-1]

        # Try candidates in order
        tried = 0
        for idx in candidate_idxs:
            fx, fy = int(F[idx,0]), int(F[idx,1])
            tried += 1

            # dynamic min distance: allow closer target when in corridor (front small)
            min_dist = MIN_FRONTIER_GRID_DIST
            if hasattr(self, "front") and self.front < 1.0:
                min_dist = max(1, MIN_FRONTIER_GRID_DIST // 2)

            if dist[idx] < min_dist:
                continue

            # skip if inflated occupancy blocks it
            # Allow frontier if ANY of its 4-neighbors is free.
            if not (self.is_cell_free_inflated(fx, fy) or
                    self.is_cell_free_inflated(fx+1, fy) or
                    self.is_cell_free_inflated(fx-1, fy) or
                    self.is_cell_free_inflated(fx, fy+1) or
                    self.is_cell_free_inflated(fx, fy-1)):
                continue


            start = (rx, ry); goal = (fx, fy)
            path = self.astar_grid(start, goal)
            if path:
                self.current_path = path
                self.current_path_world = [self.grid_to_world(gx, gy) for gx, gy in path]
                self.path_goal = goal
                self.get_logger().info(f"📍 Planned path to frontier grid={goal} path_len={len(path)} tried={tried}")
                # DO NOT set exploration_dir here; set when goal reached or after a while
                # reset progress trackers
                self.last_progress_dist = None
                self.last_progress_time = time.time()
                self.last_progress_path_seq = self.map_seq
                return True
            else:
                # mark failure
                key = (fx, fy)
                self.failed_frontiers[key] = self.failed_frontiers.get(key, 0) + 1
                self._failed_timestamps[key] = time.time()
                continue

        # fallback: try all frontiers by distance ascending
        order2 = np.argsort(dist)
        for idx in order2:
            fx, fy = int(F[idx,0]), int(F[idx,1])
            if dist[idx] < 1: continue
            if not self.is_cell_free_inflated(fx, fy): continue
            start = (rx, ry); goal = (fx, fy)
            path = self.astar_grid(start, goal)
            if path:
                self.current_path = path
                self.current_path_world = [self.grid_to_world(gx, gy) for gx, gy in path]
                self.path_goal = goal
                self.exploration_dir = ((fx - rx)/ (dist[idx] + 1e-9), (fy - ry)/ (dist[idx] + 1e-9))
                self.last_progress_dist = None
                self.last_progress_time = time.time()
                self.last_progress_path_seq = self.map_seq
                self.get_logger().info(f"📍 Fallback planned path to frontier grid={goal} path_len={len(path)}")
                return True

        self.get_logger().info("No reachable frontier found after trying candidates.")
        return False

    # -----------------------------------------------------
    # A* ON GRID (4-connected)
    # -----------------------------------------------------
    def astar_grid(self, start, goal):
        sx, sy = start; gx, gy = goal
        H, W = self.map_height, self.map_width
        if not (0 <= gx < W and 0 <= gy < H): return None
        M_infl = self.inflated_map
        if M_infl[gy, gx] == 1: return None
        if M_infl[sy, sx] == 1: return None

        open_heap = []
        heapq.heappush(open_heap, (self.heuristic(start,goal), 0, start, None))
        came_from = {}
        gscore = {start: 0}
        closed = set()
        neighbors = [(1,0),(-1,0),(0,1),(0,-1)]

        while open_heap:
            f,g,current,parent = heapq.heappop(open_heap)
            if current in closed: continue
            came_from[current] = parent
            if current == goal:
                # reconstruct
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path
            closed.add(current)
            cx, cy = current
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H): continue
                if M_infl[ny, nx] == 1: continue
                neighbor = (nx, ny)
                tentative_g = g + 1
                if tentative_g < gscore.get(neighbor, 1e9):
                    gscore[neighbor] = tentative_g
                    fscore = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (fscore, tentative_g, neighbor, current))
        return None

    def heuristic(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # -----------------------------------------------------
    # PATH FOLLOWER
    # -----------------------------------------------------
    def follow_path(self):
        if not self.current_path_world:
            return Twist()

        wx, wy = self.current_path_world[0]
        dx = wx - self.robot_x; dy = wy - self.robot_y
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_angle - self.robot_yaw), math.cos(target_angle - self.robot_yaw))

        cmd = Twist()

        # pop waypoint if close
        if dist < GOAL_NEAR_DIST_M:
            self.current_path_world.pop(0)
            self.current_path.pop(0)
            if not self.current_path_world:
                self.get_logger().info(f"✅ Reached path goal grid={self.path_goal}, clearing path")
                gx, gy = self.path_goal if self.path_goal is not None else (None, None)
                self.path_goal = None
                # mark visited area
                if self.visited_mask is not None and gx is not None:
                    rr = self.VISITED_RADIUS
                    H, W = self.map_height, self.map_width
                    x0 = max(0, gx - rr); x1 = min(W, gx + rr + 1)
                    y0 = max(0, gy - rr); y1 = min(H, gy + rr + 1)
                    xs = np.arange(x0, x1); ys = np.arange(y0, y1)
                    dxs = xs.reshape(1, -1) - gx
                    dys = ys.reshape(-1, 1) - gy
                    d2 = dxs*dxs + dys*dys
                    disk = d2 <= (rr*rr)
                    self.visited_mask[y0:y1, x0:x1] |= disk
                self.exploration_dir = None
                return Twist()

        # HALLWAY PUSH: only if both sides close, front clear, and roughly aligned
        left, right, front = self.left, self.right, self.front
        if left < 0.7 and right < 0.7 and front > 0.5 and abs(yaw_error) < 0.3:
            cmd.linear.x = 0.2  # faster push
            cmd.angular.z = yaw_error * 0.5
            self.push_timer = time.time()
            return cmd

        # continue short push duration
        if self.push_timer and (time.time() - self.push_timer < 0.35):
            cmd.linear.x = 0.10; cmd.angular.z = 0.0
            return cmd
        else:
            self.push_timer = None

        # Safety: if obstacle very close, do small backup + turn (not a pure spin)
        if front < 0.28:
            cmd.linear.x = -0.03
            cmd.angular.z = -0.35 if left < right else 0.35
            return cmd

        # If big yaw error, rotate first (but detect oscillation on intended ang)
        if abs(yaw_error) > ALIGN_ANGLE:
            ang = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, 1.2 * yaw_error))
            if self.detect_oscillation(ang):
                self.get_logger().warn("🔄 Oscillation detected (rotate) -> forward nudge")
                self.oscillating = True; self.osc_end = time.time() + 0.35
                cmd2 = Twist(); cmd2.linear.x = 0.08; self.cmd_pub.publish(cmd2); return cmd2
            cmd.angular.z = ang; cmd.linear.x = 0.0
            return cmd

        # nominal forward drive with small angular correction; reduce speed near obstacles
        base_speed = min(MAX_LINEAR_SPEED, 0.6 * dist + 0.02)
        if front < 0.5:
            # strong slowdown when front gets small (crawl)
            base_speed = min(base_speed, max(0.05, (front - 0.2) * 0.5))
        yaw_scale = max(0.3, math.exp(-3.0 * abs(yaw_error)))  # keeps min speed at 0.3
        cmd.linear.x = base_speed * yaw_scale
        cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, yaw_error))
        # oscillation detection on commanded ang
        if self.detect_oscillation(cmd.angular.z):
            self.get_logger().warn("🔄 Oscillation detected (drive) -> forward nudge")
            self.oscillating = True; self.osc_end = time.time() + 0.35
            ncmd = Twist(); ncmd.linear.x = 0.08; self.cmd_pub.publish(ncmd); return ncmd

        return cmd

    # -----------------------------------------------------
    # RECOVERY
    # -----------------------------------------------------
    def start_recovery(self):
        # mark current path goal as failed
        if self.path_goal is not None:
            key = (int(self.path_goal[0]), int(self.path_goal[1]))
            self.failed_frontiers[key] = self.failed_frontiers.get(key, 0) + 1
            self._failed_timestamps[key] = time.time()
            self.get_logger().info(f"start_recovery: marking frontier {key} failed (count={self.failed_frontiers[key]})")
        # begin recovery
        self.recovering = True
        self.recovery_stage = "backup"
        self.recovery_end_time = time.time() + BACKUP_TIME
        # clear current path to force replanning
        self.current_path = []; self.current_path_world = []; self.path_goal = None

    def recovery_cmd(self):
        cmd = Twist()
        if self.recovery_stage == "backup":
            cmd.linear.x = -0.06; cmd.angular.z = 0.0
            if time.time() >= self.recovery_end_time:
                self.recovery_stage = "rotate"; self.recovery_end_time = time.time() + RECOVERY_ROTATE_TIME
            return cmd
        elif self.recovery_stage == "rotate":
            cmd.linear.x = 0.0; cmd.angular.z = 0.5
            return cmd
        else:
            return Twist()

    # ---------------- map/world helpers ----------------
    def world_to_grid(self, wx, wy):
        gx = int((wx - self.map_origin[0]) / self.map_res)
        gy = int((wy - self.map_origin[1]) / self.map_res)
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
