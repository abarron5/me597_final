#!/usr/bin/env python3
"""
Explorer node — full rewrite for TurtleBot3 Waffle
Features:
 - FSM-based explorer
 - Frontier queue + clustering + scoring
 - Reachability filtering on inflated map
 - Pure Pursuit local controller
 - Reactive vector-field obstacle avoidance + hard safety override
 - Robust lidar handling (angle-aware)
 - Conservative defaults tuned for TB3 Waffle
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import time
import heapq
from collections import deque, defaultdict, namedtuple
from typing import List, Tuple, Optional

# ----------------- TUNABLE CONSTANTS (tune for robot) -----------------
ROBOT = "turtlebot3_waffle"

OCCUPIED_THRESH = 70        # occupancy grid >= this considered occupied
UNKNOWN_VAL = -1
INFLATE_RADIUS = 4          # grid cells (you said 4 works)
MAP_RES_DEFAULT = 0.05

# Pure Pursuit controller
LOOKAHEAD_M = 0.25          # lookahead distance in meters
MAX_LINEAR_SPEED = 0.25     # conservative for TB3 Waffle
MAX_ANGULAR_SPEED = 1.2
GOAL_NEAR_DIST_M = 0.15
MIN_GOAL_DIST = 4

# Planner tuning
REPLAN_INTERVAL = 1.0
MIN_FRONTIER_GRID_DIST = 6  # cells
FAILED_FRONTIER_LIMIT = 3
FAILED_CLEAR_TIME = 20.0

# Reactive safety
STOP_DIST = 0.30            # absolute hard-stop forward (m)
SLOW_DIST = 0.6             # start steering & slow (m)
AVOID_SPEED = 0.06
THIN_OVERRIDE = 0.18        # treat very near single rays as thin obstacles

# Stuck detection / recovery
STUCK_MOVE_THRESH = 0.05
STUCK_TIME = 4.0
BACKUP_TIME = 0.6
RECOVERY_ROTATE_TIME = 1.0
REAR_BLOCK_DIST = 0.20

# Misc
TIMER_PERIOD = 0.1          # seconds
SCAN_MAX_RANGE_DEFAULT = 10.0
SCAN_ANGLE_OFFSET = 0.0     # if your lidar 0 is not forward, set offset (radians)
# ----------------------------------------------------------------------

Frontier = namedtuple("Frontier", ["cells", "centroid", "score", "visited", "failed_count"])

class Explorer(Node):
    def __init__(self):
        super().__init__('explorer_node')

        # Subscribers & publishers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_cb)

        # pose / odom
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False

        # map
        self.map_data: Optional[np.ndarray] = None
        self.inflated_map: Optional[np.ndarray] = None
        self.map_width = 0
        self.map_height = 0
        self.map_res = MAP_RES_DEFAULT
        self.map_origin = (0.0, 0.0)
        self.map_seq = 0

        # lidar
        self.ranges_full: Optional[np.ndarray] = None
        self.front = SCAN_MAX_RANGE_DEFAULT
        self.left = SCAN_MAX_RANGE_DEFAULT
        self.right = SCAN_MAX_RANGE_DEFAULT
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0
        self.scan_n = 0

        # planner state
        self.current_path: List[Tuple[int,int]] = []
        self.current_path_world: List[Tuple[float,float]] = []
        self.path_goal: Optional[Tuple[int,int]] = None
        self.last_plan_time = 0.0

        # frontier management
        self.frontier_queue: List[Tuple[float, Tuple[int,int]]] = []  # (neg_score, centroid_grid)
        self.frontier_map = {}  # centroid -> Frontier
        self.frontier_seq = None

        # visited frontiers and failed counters
        self.visited_mask: Optional[np.ndarray] = None
        self.VISITED_RADIUS = int(0.5 / self.map_res) if self.map_res else 10
        self.failed_frontiers = defaultdict(int)
        self.failed_timestamps = {}

        # stuck and oscillation
        self.last_pos = None
        self.last_movement_time = time.time()
        self.last_cmd = Twist()

        # FSM state
        self.state = "IDLE"  # IDLE, PLAN, FOLLOW_PATH, AVOID, RECOVERY, DONE
        self.get_logger().info("Explorer node initialized (TB3 Waffle defaults).")

    # ------------------------ Main Timer / FSM ------------------------
    def timer_cb(self):
        now = time.time()

        # Top-level hard safety: if lidar missing or map missing behave reasonably
        if self.ranges_full is None:
            # can't make decisions without lidar -> stop and wait
            self._publish_stop()
            return

        # Hard safety override (instant) - prevents any forward motion if obstacle too close
        hard = self.hard_safety_override()
        if hard is not None:
            self.cmd_pub.publish(hard)
            self.last_cmd = hard
            return

        # State transitions
        if self.state == "IDLE":
            if self.map_data is not None:
                self.state = "PLAN"
        elif self.state == "PLAN":
            # only replan occasionally
            if time.time() - self.last_plan_time > REPLAN_INTERVAL or not self.current_path:
                planned = self._plan_once()
                self.last_plan_time = time.time()
                if planned:
                    self.state = "FOLLOW_PATH"
                else:
                    # if no frontiers -> maybe map incomplete (wait) or done
                    # check if any unknown remains reachable
                    if self._any_reachable_unknown():
                        # wait a bit and replan
                        self.get_logger().info("No plan found now — will retry.")
                        self.state = "PLAN"
                    else:
                        self.get_logger().info("🎉 No reachable unknowns found -> DONE")
                        self.state = "DONE"
        elif self.state == "FOLLOW_PATH":
            # reactive avoidance check; if avoidance returns override, execute that
            reactive = self.reactive_vector_override()
            if reactive is not None:
                self.cmd_pub.publish(reactive)
                self.last_cmd = reactive
                # Switch to AVOID briefly until obstacle cleared
                self.state = "AVOID"
                return

            if not self.current_path_world:
                self.get_logger().info("Path exhausted -> PLAN")
                self.state = "PLAN"
                return

            cmd = self.pure_pursuit_control()
            self.cmd_pub.publish(cmd)
            self.last_cmd = cmd

            # Stuck detection
            if self._check_stuck(now):
                self.state = "RECOVERY"
                return

        elif self.state == "AVOID":
            # run reactive velocity until clear
            reactive = self.reactive_vector_override()
            if reactive is not None:
                self.cmd_pub.publish(reactive)
                self.last_cmd = reactive
            else:
                # cleared -> resume following if path exists
                if self.current_path_world:
                    self.state = "FOLLOW_PATH"
                else:
                    self.state = "PLAN"

        elif self.state == "RECOVERY":
            cmd = self.recovery_cmd()
            # always check reactive during recovery
            reactive = self.reactive_vector_override()
            if reactive is not None:
                self.cmd_pub.publish(reactive)
                self.last_cmd = reactive
            else:
                self.cmd_pub.publish(cmd)
                self.last_cmd = cmd
            # If recovery finished (recovering flag false), go to PLAN
            if not self.recovering:
                self.state = "PLAN"

        elif self.state == "DONE":
            # stop and stay
            self._publish_stop()
            return

    # ------------------------ LIDAR processing ------------------------
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        ranges = np.where(np.isfinite(ranges), ranges, SCAN_MAX_RANGE_DEFAULT)
        self.ranges_full = ranges
        self.scan_angle_min = msg.angle_min + SCAN_ANGLE_OFFSET
        self.scan_angle_inc = msg.angle_increment
        self.scan_n = len(ranges)

        # get statistics for front/left/right using ±12° windows
        def angle_to_idx(angle):
            return int(round((angle - self.scan_angle_min) / self.scan_angle_inc))

        N = self.scan_n
        if N == 0 or self.scan_angle_inc == 0:
            return

        FRONT = 0.0; LEFT = math.pi/2; RIGHT = -math.pi/2
        w = math.radians(12)

        def sector_stats(center):
            i0 = angle_to_idx(center - w); i1 = angle_to_idx(center + w)
            i0 %= N; i1 %= N
            if i0 <= i1:
                sec = ranges[i0:i1+1]
            else:
                sec = np.concatenate((ranges[i0:], ranges[:i1+1]))
            if sec.size == 0:
                return SCAN_MAX_RANGE_DEFAULT, SCAN_MAX_RANGE_DEFAULT
            return float(np.median(sec)), float(np.min(sec))

        f_med, f_min = sector_stats(FRONT)
        l_med, l_min = sector_stats(LEFT)
        r_med, r_min = sector_stats(RIGHT)

        self.front = f_min if f_min < THIN_OVERRIDE else f_med
        self.left = l_min if l_min < THIN_OVERRIDE else l_med
        self.right = r_min if r_min < THIN_OVERRIDE else r_med

    # ------------------------ ODOM ------------------------
    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2*(q.w * q.z + q.x * q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self.robot_yaw = math.atan2(siny, cosy)
        if not self.odom_received:
            self.odom_received = True
            self.last_pos = (self.robot_x, self.robot_y)
            self.last_movement_time = time.time()

    # ------------------------ MAP ------------------------
    def map_callback(self, msg: OccupancyGrid):
        # robust parsing
        try:
            H = msg.info.height
            W = msg.info.width
            arr = np.array(msg.data, dtype=int)
            if arr.size != H * W:
                self.get_logger().warn(f"Map data size mismatch {arr.size} != {H}*{W}")
                return
            self.map_data = arr.reshape(H, W)
        except Exception as e:
            self.get_logger().error(f"Failed to parse map: {e}")
            return

        self.map_width = W
        self.map_height = H
        if msg.info.resolution and msg.info.resolution > 0:
            self.map_res = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        self.map_seq += 1
        self.inflated_map = self.build_inflated_map(self.map_data, INFLATE_RADIUS)

        # (re)initialize visited mask if needed
        if self.visited_mask is None or self.visited_mask.shape != self.map_data.shape:
            self.visited_mask = np.zeros_like(self.map_data, dtype=bool)

        # invalidate frontier cache so plan will recompute clusters
        self.frontier_seq = None

    # ------------------------ MAP INFLATION ------------------------
    def build_inflated_map(self, M: np.ndarray, radius_cells: int) -> np.ndarray:
        H, W = M.shape
        binmap = (M >= OCCUPIED_THRESH).astype(np.uint8)
        if radius_cells <= 0 or H < 3 or W < 3:
            return binmap.copy()

        dist = np.full((H, W), -1, dtype=int)
        inflated = binmap.copy()
        q = deque()
        ys, xs = np.nonzero(binmap == 1)
        for y, x in zip(ys, xs):
            q.append((y, x)); dist[y, x] = 0

        neighbors8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        while q:
            y, x = q.popleft()
            d = dist[y, x]
            if d >= radius_cells:
                continue
            for dy, dx in neighbors8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] == -1:
                    dist[ny, nx] = d + 1
                    inflated[ny, nx] = 1
                    q.append((ny, nx))

        return inflated

    # ------------------------ FRONTIER CLUSTERING ------------------------
    def _compute_frontiers(self) -> List[Frontier]:
        """
        Compute frontier clusters only when map_seq changed; returns list of Frontier
        A frontier cell = free cell with any unknown neighbor.
        Cluster adjacent frontier cells into connected components and compute centroid and cluster size.
        """
        if self.map_data is None:
            return []

        # cache guard
        if self.frontier_seq == self.map_seq and self.frontier_map:
            return list(self.frontier_map.values())

        M = self.map_data
        H, W = M.shape
        unknown = (M == UNKNOWN_VAL)
        free = (M == 0)

        # compute frontier mask
        neigh_unknown = np.zeros_like(M, dtype=bool)
        shifts = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1),(0,0)]
        for dy, dx in shifts:
            neigh_unknown |= np.roll(np.roll(unknown, dy, axis=0), dx, axis=1)
        frontier_mask = free & neigh_unknown

        # clear boundary
        frontier_mask[0,:] = False; frontier_mask[-1,:] = False; frontier_mask[:,0] = False; frontier_mask[:,-1] = False

        ys, xs = np.nonzero(frontier_mask)
        cells = list(zip(xs.tolist(), ys.tolist()))  # grid (x,y)

        # cluster via BFS over 8-connected frontier mask
        frontier_map = {}
        visited = set()
        for (x0, y0) in cells:
            if (x0, y0) in visited:
                continue
            # BFS
            stack = [(x0, y0)]
            comp = []
            visited.add((x0, y0))
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and frontier_mask[ny, nx] and (nx, ny) not in visited:
                        visited.add((nx, ny)); stack.append((nx, ny))
            # compute centroid
            xs_comp = [c[0] for c in comp]; ys_comp = [c[1] for c in comp]
            cx = int(round(sum(xs_comp) / len(xs_comp)))
            cy = int(round(sum(ys_comp) / len(ys_comp)))
            f = Frontier(cells=comp, centroid=(cx, cy), score=0.0, visited=False, failed_count=0)
            frontier_map[(cx, cy)] = f

        # replace cache
        self.frontier_map = frontier_map
        self.frontier_seq = self.map_seq
        return list(frontier_map.values())

    # ------------------------ REACHABILITY FILTER ------------------------
    def _reachable_mask(self) -> Optional[np.ndarray]:
        if self.inflated_map is None or self.map_data is None:
            return None
        H, W = self.inflated_map.shape
        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)
        if not (0 <= rx < W and 0 <= ry < H):
            return None
        if self.inflated_map[ry, rx] == 1:
            # robot inside an obstacle in inflated map -> unreachable
            return None
        reachable = np.zeros_like(self.inflated_map, dtype=bool)
        q = deque()
        q.append((rx, ry))
        reachable[ry, rx] = True
        while q:
            x, y = q.popleft()
            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if 0 <= nx < W and 0 <= ny < H and not reachable[ny, nx] and self.inflated_map[ny, nx] == 0:
                    reachable[ny, nx] = True
                    q.append((nx, ny))
        return reachable

    # ------------------------ FRONTIER SCORING ------------------------
    def _score_frontier(self, frontier: Frontier, reachable_mask: np.ndarray) -> float:
        """
        Score = information_gain (cluster size) * alignment + distance_penalty
        alignment = dot of unit vector to centroid and robot heading
        distance_penalty = 1/(1 + dist)
        """
        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)
        cx, cy = frontier.centroid
        # skip unreachable centroid
        H, W = self.map_height, self.map_width
        if not (0 <= cx < W and 0 <= cy < H) or not reachable_mask[cy, cx]:
            return -1e9

        dx = cx - rx; dy = cy - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            dist = 1e-6
        # alignment
        ex = math.cos(self.robot_yaw); ey = math.sin(self.robot_yaw)
        ux = dx / dist; uy = dy / dist
        alignment = max(-1.0, min(1.0, ux*ex + uy*ey))
        info_gain = len(frontier.cells)
        score = info_gain * (0.5 + 0.5 * (alignment + 1) / 2.0) * (1.0 / (1.0 + dist))
        return score

    # ------------------------ PLAN ONCE (choose next frontier) ------------------------
    def _plan_once(self) -> bool:
        """
        Selects the best frontier cluster, finds a safe target near its centroid,
        ensures the goal is not trivial (too close to robot), and runs A*.
        """

        # 1. Compute frontier clusters
        frontiers = self._compute_frontiers()
        if not frontiers:
            return False

        # 2. Compute reachability mask
        reachable = self._reachable_mask()
        if reachable is None:
            self.get_logger().warn("Robot not in free space (inflated map) — cannot plan.")
            return False

        # 3. Prepare priority queue of candidate frontiers
        pq = []
        for f in frontiers:
            cx, cy = f.centroid

            # skip visited
            if (
                self.visited_mask is not None
                and 0 <= cx < self.map_width
                and 0 <= cy < self.map_height
                and self.visited_mask[cy, cx]
            ):
                continue

            score = self._score_frontier(f, reachable)
            if score < -1e8:
                continue

            heapq.heappush(pq, (-score, (cx, cy)))  # max-heap via negative

        if not pq:
            return False

        # ---- Robot grid position (compute once up front!) ----
        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)
        rx = int(rx)
        ry = int(ry)

        # ---- Planning loop: try candidates in ranked order ----
        while pq:
            negscore, (cx, cy) = heapq.heappop(pq)
            score = -negscore

            f = self.frontier_map.get((cx, cy))
            if f is None:
                continue

            # Skip recently-failed frontiers
            fail_count = self.failed_frontiers.get((cx, cy), 0)
            if fail_count >= FAILED_FRONTIER_LIMIT:
                ts = self.failed_timestamps.get((cx, cy), 0)
                if time.time() - ts < FAILED_CLEAR_TIME:
                    continue
                else:
                    self.failed_frontiers.pop((cx, cy), None)
                    self.failed_timestamps.pop((cx, cy), None)

            # 4. Find a safe target near frontier centroid
            safe = self.find_safe_target_near_frontier(cx, cy, max_radius=8)
            if safe is None:
                self.failed_frontiers[(cx, cy)] += 1
                self.failed_timestamps[(cx, cy)] = time.time()
                continue

            gx, gy = safe

            # --- PATCH: skip trivial goals (too close to robot) ---
            if abs(gx - rx) + abs(gy - ry) < 3:
                # skip this frontier; pick a deeper target instead
                continue
            # ------------------------------------------------------

            # 5. Run A* to the safe target
            path = self.astar_grid((rx, ry), (gx, gy))
            if path:
                self.current_path = path
                self.current_path_world = [self.grid_to_world(x, y) for x, y in path]
                self.path_goal = (cx, cy)
                self.get_logger().info(
                    f"Planned path to frontier centroid {(cx, cy)} "
                    f"(len={len(path)}, score={score:.3f})"
                )
                return True

            # A* failed → mark as failed
            self.failed_frontiers[(cx, cy)] += 1
            self.failed_timestamps[(cx, cy)] = time.time()

        return False

    # ------------------------ SAFE TARGET NEAR FRONTIER ------------------------
    def find_safe_target_near_frontier(self, fx: int, fy: int, max_radius: int = 8) -> Optional[Tuple[int,int]]:
        """
        Search inflated_map near (fx, fy) for a free cell to use as A* goal.
        Prefer a cell slightly back from the frontier towards the robot (so robot doesn't drive into unknown).
        """
        if self.inflated_map is None:
            return None
        H, W = self.inflated_map.shape
        # direction back toward robot
        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)
        vx, vy = rx - fx, ry - fy
        norm = math.hypot(vx, vy) or 1.0
        cand_x = int(round(fx + (vx / norm) * MIN_GOAL_DIST))
        cand_y = int(round(fy + (vy / norm) * MIN_GOAL_DIST))
        # check candidate then spiral   
        def is_free(x, y):
            if 0 <= x < W and 0 <= y < H:
                return self.inflated_map[y, x] == 0
            return False

        if is_free(cand_x, cand_y):
            return (cand_x, cand_y)

        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    x, y = fx + dx, fy + dy
                    if is_free(x, y):
                        return (x, y)
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    x, y = fx + dx, fy + dy
                    if is_free(x, y):
                        return (x, y)
        return None

    # ------------------------ A* (4-connected) ------------------------
    def astar_grid(self, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        if self.inflated_map is None:
            return None
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        H, W = self.inflated_map.shape
        if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
            return None
        if self.inflated_map[sy, sx] == 1 or self.inflated_map[gy, gx] == 1:
            return None

        open_heap = []
        heapq.heappush(open_heap, (0 + self.heuristic((sx,sy),(gx,gy)), 0, (sx,sy), None))
        came_from = {}
        gscore = { (sx,sy): 0 }
        closed = set()
        neighbors = [(1,0),(-1,0),(0,1),(0,-1)]

        while open_heap:
            f, g, current, parent = heapq.heappop(open_heap)
            if current in closed:
                continue
            came_from[current] = parent
            if current == (gx, gy):
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
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if self.inflated_map[ny, nx] == 1:
                    continue
                neighbor = (nx, ny)
                tentative_g = g + 1
                if tentative_g < gscore.get(neighbor, 1e9):
                    gscore[neighbor] = tentative_g
                    fscore = tentative_g + self.heuristic(neighbor, (gx, gy))
                    heapq.heappush(open_heap, (fscore, tentative_g, neighbor, current))
        return None

    def heuristic(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ------------------------ WORLD/GRID helpers ------------------------
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int,int]:
        gx = int((wx - self.map_origin[0]) / self.map_res)
        gy = int((wy - self.map_origin[1]) / self.map_res)
        gx = max(0, min(self.map_width - 1, gx))
        gy = max(0, min(self.map_height - 1, gy))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float,float]:
        wx = self.map_origin[0] + gx * self.map_res + 0.5*self.map_res
        wy = self.map_origin[1] + gy * self.map_res + 0.5*self.map_res
        return wx, wy

    # ------------------------ PURE PURSUIT local controller ------------------------
    def pure_pursuit_control(self) -> Twist:
        """
        Lookahead-based control: find first point on path beyond LOOKAHEAD_M, compute curvature
        and convert curvature to v, omega. Smooth and clamp velocities.
        """
        if not self.current_path_world:
            return Twist()

        # find lookahead point
        lx = None; ly = None
        for (wx, wy) in self.current_path_world:
            dx = wx - self.robot_x; dy = wy - self.robot_y
            d = math.hypot(dx, dy)
            if d >= LOOKAHEAD_M:
                lx, ly = wx, wy
                break
        if lx is None:
            # no lookahead found -> use last waypoint
            lx, ly = self.current_path_world[-1]

        # transform to robot frame
        dx = lx - self.robot_x; dy = ly - self.robot_y
        # rotate by -robot_yaw
        x_r = math.cos(-self.robot_yaw) * dx - math.sin(-self.robot_yaw) * dy
        y_r = math.sin(-self.robot_yaw) * dx + math.cos(-self.robot_yaw) * dy

        # curvature kappa = 2*y_r / L^2 where L is lookahead distance
        L = math.hypot(x_r, y_r)
        if L < 1e-6:
            return Twist()

        kappa = (2.0 * y_r) / (L * L)
        # target linear speed scaled by how much curvature required
        v = MAX_LINEAR_SPEED * max(0.15, (1.0 - min(abs(kappa)*2.5, 0.9)))
        # if close to final waypoint, slow
        # if final waypoint near, pop it
        # also if path point within GOAL_NEAR_DIST_M, pop it
        wx0, wy0 = self.current_path_world[0]
        if math.hypot(wx0 - self.robot_x, wy0 - self.robot_y) < GOAL_NEAR_DIST_M:
            # pop first waypoint
            self.current_path_world.pop(0)
            if self.current_path:
                self.current_path.pop(0)
            # if path finished, mark visited area and request plan
            if not self.current_path_world:
                self._mark_frontier_visited()
                self.current_path = []
                self.path_goal = None
                return Twist()

        # compute omega = v * kappa (approx)
        omega = v * kappa
        # clamp
        v = max(0.0, min(MAX_LINEAR_SPEED, v))
        omega = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, omega))

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        return cmd

    # ------------------------ REACTIVE VECTOR FIELD ------------------------
    def reactive_vector_override(self) -> Optional[Twist]:
        """
        Compute repulsive vector from lidar points in front half-plane and combine
        with attractive vector toward current lookahead point (if any).
        If combined forward component <= 0 -> return rotate-only cmd.
        Return None if no reactive action necessary (no near obstacles).
        """
        if self.ranges_full is None:
            return None
        # quick checks using precomputed front/left/right
        if self.front < STOP_DIST:
            # immediate rotate away
            cmd = Twist(); cmd.linear.x = 0.0
            cmd.angular.z = 0.8 if self.left > self.right else -0.8
            return cmd
        if self.front > SLOW_DIST:
            return None  # no avoidance needed

        # Build repulsive vector in robot frame by sampling rays in frontal wedge +-60°
        N = self.scan_n
        if N == 0 or self.scan_angle_inc == 0:
            return None
        half_w = math.radians(60)
        angs = np.arange(self.scan_angle_min, self.scan_angle_min + N*self.scan_angle_inc, self.scan_angle_inc)
        # choose angles in [-60, +60]
        indices = np.where(np.logical_and(angs >= -half_w, angs <= half_w))[0]
        if indices.size == 0:
            # fallback to using left/right
            ang = 0.6 if self.left > self.right else -0.6
            cmd = Twist(); cmd.linear.x = AVOID_SPEED; cmd.angular.z = ang
            return cmd

        # repulsion sum
        rx = 0.0; ry = 0.0
        for i in indices:
            r = float(self.ranges_full[i])
            if r <= 0.001 or r > SLOW_DIST:
                continue
            a = angs[i]
            # weight inverse-square, clamp
            w = 1.0 / max(0.01, r*r)
            rx += -w * math.cos(a)
            ry += -w * math.sin(a)

        # attractive vector = heading toward lookahead (robot frame)
        attr_x = 0.0; attr_y = 0.0
        if self.current_path_world:
            # take first point as attraction
            lx, ly = self.current_path_world[0]
            dx = lx - self.robot_x; dy = ly - self.robot_y
            # robot-frame transform
            attr_x = math.cos(-self.robot_yaw) * dx - math.sin(-self.robot_yaw) * dy
            attr_y = math.sin(-self.robot_yaw) * dx + math.cos(-self.robot_yaw) * dy

        # combine repulsion + attraction (weights)
        vx = 0.8 * attr_x + 1.5 * rx
        vy = 0.8 * attr_y + 1.5 * ry

        # compute desired heading and forward
        desired_angle = math.atan2(vy, vx) if abs(vx) > 1e-6 or abs(vy) > 1e-6 else 0.0
        forward = vx * math.cos(desired_angle) + vy * math.sin(desired_angle)
        # clamp forward to small positive speed while avoiding
        if forward <= 0.0:
            # no forward component -> spin in place toward more free side
            cmd = Twist(); cmd.linear.x = 0.0
            cmd.angular.z = 0.6 if self.left > self.right else -0.6
            return cmd
        else:
            cmd = Twist()
            cmd.linear.x = min(AVOID_SPEED, forward)
            cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, desired_angle * 1.0))
            return cmd

    # ------------------------ HARD SAFETY OVERRIDE ------------------------
    def hard_safety_override(self) -> Optional[Twist]:
        """
        Top-level hard safety: if ANY ray in a narrow frontal wedge closer than STOP_DIST,
        immediately stop forward motion and rotate away.
        """
        if self.ranges_full is None or self.scan_n == 0:
            return None
        # take central wedge ±20% of scan (approx ±72° on 360 -> but uses angles)
        N = self.scan_n
        if N == 0:
            return None
        # compute front arc indices around forward angle (angle 0 corresponds to self.scan_angle_min -> need mapping)
        # faster heuristic: use self.front computed earlier (median/min style)
        if self.front < STOP_DIST:
            cmd = Twist(); cmd.linear.x = 0.0
            cmd.angular.z = 0.8 if self.left > self.right else -0.8
            return cmd
        return None

    # ------------------------ STUCK DETECTION & RECOVERY ------------------------
    def _check_stuck(self, now: float) -> bool:
        if not self.odom_received:
            return False
        if not hasattr(self, "last_cmd") or getattr(self.last_cmd, "linear", None) is None:
            return False
        if self.last_cmd.linear.x <= 0.03:
            # not actively moving forward, don't consider stuck
            if self.last_pos is None:
                self.last_pos = (self.robot_x, self.robot_y)
                self.last_movement_time = now
            return False

        if self.last_pos is None:
            self.last_pos = (self.robot_x, self.robot_y)
            self.last_movement_time = now
            return False

        dx = self.robot_x - self.last_pos[0]; dy = self.robot_y - self.last_pos[1]
        moved = math.hypot(dx, dy)
        if moved > STUCK_MOVE_THRESH:
            self.last_pos = (self.robot_x, self.robot_y); self.last_movement_time = now
            return False

        if now - self.last_movement_time > STUCK_TIME:
            self.get_logger().warn("Stuck detected — starting recovery")
            self.start_recovery()
            return True
        return False

    def start_recovery(self):
        # if boxed in both front & rear -> clear path & replan
        front_blocked = self.front < STOP_DIST if hasattr(self, "front") else False
        rear_blocked = (self.left < REAR_BLOCK_DIST or self.right < REAR_BLOCK_DIST) if hasattr(self, "left") else False
        if front_blocked and rear_blocked:
            self.get_logger().warn("Boxed in — clearing path & replanning")
            self._reset_path()
            self.recovering = False
            return
        # start backup then rotate
        self.recovering = True
        self.recovery_stage = "backup"
        self.recovery_end_time = time.time() + BACKUP_TIME

    def recovery_cmd(self) -> Twist:
        cmd = Twist()
        if not getattr(self, "recovering", False):
            return cmd
        # check rear safety
        rear_blocked = (self.left < REAR_BLOCK_DIST or self.right < REAR_BLOCK_DIST)
        if self.recovery_stage == "backup" and rear_blocked:
            # skip backup -> rotate
            self.recovery_stage = "rotate"
            self.recovery_end_time = time.time() + RECOVERY_ROTATE_TIME
        if self.recovery_stage == "backup":
            cmd.linear.x = -0.04; cmd.angular.z = 0.0
            if time.time() >= self.recovery_end_time:
                self.recovery_stage = "rotate"; self.recovery_end_time = time.time() + RECOVERY_ROTATE_TIME
            return cmd
        if self.recovery_stage == "rotate":
            # if rotate unsafe, give up and clear path
            if (self.left < 0.12 and self.right < 0.12):
                self.get_logger().warn("Rotate unsafe — clearing path")
                self._reset_path()
                self.recovering = False
                return Twist()
            cmd.linear.x = 0.0; cmd.angular.z = 0.6
            # finish recovery after rotate time passes
            if time.time() >= self.recovery_end_time:
                self.recovering = False
            return cmd
        return cmd

    # ------------------------ VISITED / FRONTIER MARKING ------------------------
    def _mark_frontier_visited(self):
        """
        When the robot reaches the end of a path, mark frontier area visited by checking
        whether unknown cells behind frontier became observed. We conservatively mark a disk
        in visited_mask around the centroid, but only if unknowns were resolved.
        """
        if self.path_goal is None or self.visited_mask is None or self.map_data is None:
            return
        cx, cy = self.path_goal
        # check whether near-frontier unknown cells are now known (not UNKNOWN_VAL)
        M = self.map_data
        H, W = M.shape
        rr = max(2, int(0.5 / self.map_res))
        x0 = max(0, cx - rr); x1 = min(W, cx + rr + 1)
        y0 = max(0, cy - rr); y1 = min(H, cy + rr + 1)
        region = M[y0:y1, x0:x1]
        # if region has no unknowns -> mark visited
        if np.all(region != UNKNOWN_VAL):
            xs = np.arange(x0, x1); ys = np.arange(y0, y1)
            dxs = xs.reshape(1, -1) - cx; dys = ys.reshape(-1, 1) - cy
            d2 = dxs*dxs + dys*dys
            disk = d2 <= (rr*rr)
            self.visited_mask[y0:y1, x0:x1] |= disk
            self.get_logger().info(f"Marked frontier at {(cx,cy)} visited (radius {rr}).")
        else:
            # don't mark visited if unknown remains — ensures not prematurely declaring done
            self.get_logger().info(f"Frontier at {(cx,cy)} not yet observed behind; not marking visited.")

    # ------------------------ HELPERS ------------------------
    def _any_reachable_unknown(self) -> bool:
        """
        Check whether any unknown cells are reachable from the robot's free workspace.
        If none -> exploration is done.
        """
        if self.map_data is None or self.inflated_map is None:
            return False
        reachable = self._reachable_mask()
        if reachable is None:
            return False
        unknown = (self.map_data == UNKNOWN_VAL)
        # reachable unknown if any unknown cell has a free neighbor that is reachable
        H, W = self.map_data.shape
        for y in range(1, H-1):
            for x in range(1, W-1):
                if not unknown[y, x]:
                    continue
                # if any neighbor free & reachable
                neigh = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                for nx, ny in neigh:
                    if 0 <= nx < W and 0 <= ny < H and reachable[ny, nx] and self.inflated_map[ny, nx] == 0:
                        return True
        return False

    def _reset_path(self):
        self.current_path = []; self.current_path_world = []; self.path_goal = None

    def _publish_stop(self):
        cmd = Twist(); cmd.linear.x = 0.0; cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.last_cmd = cmd

# ----------------------------- main ---------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = Explorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
