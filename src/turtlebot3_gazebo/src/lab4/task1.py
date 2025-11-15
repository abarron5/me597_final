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
INFLATE_RADIUS = 5            # smaller by default; adjust if robot footprint requires
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
        super().__init__('task1_node')

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
        self.map_width = 0
        self.map_height = 0
        self.map_res = 0.05
        self.map_origin = (0.0, 0.0)
        self.map_seq = 0  # increment on map 
        self.inflated_map = None

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
        self.VISITED_RADIUS = 20            # radius in grid cells to mark visited around goals (tune)
        self.failed_frontiers = {}         # dict {(gx,gy): fail_count}
        self.FAILED_LIMIT = 3              # after this many A* fails, temporarily ignore frontier
        self.FAILED_CLEAR_TIME = 60.0      # seconds after which failed_frontiers entry may be dropped
        self._failed_timestamps = {}       # {(gx,gy): last_fail_time}

        self.get_logger().info("Task 1 node started.")

    # ---------------- TIMER LOOP ----------------
    def timer_cb(self):
        # If we are in recovery, run recovery behavior
        now = time.time()
        if self.recovering:
            if now < self.recovery_end_time:
                self.cmd_pub.publish(self.recovery_cmd())
                return
            else:
                # finish recovery
                self.recovering = False
                self.recovery_stage = None
                self.get_logger().info("🔁 Recovery finished, replanning.")
                # clear path to force replan
                self.current_path = []
                self.current_path_world = []
                self.path_goal = None
                # optionally keep exploration_dir or reset it:
                self.exploration_dir = None


        # if no map yet, rotate slowly to get SLAM started
        if self.map_data is None:
            cmd = Twist()
            cmd.angular.z = 0.4
            self.cmd_pub.publish(cmd)
            return

        # Check stuck while driving forward
        if hasattr(self, "last_cmd") and getattr(self.last_cmd, "linear", None) is not None:
            if self.last_cmd.linear.x > 0.03:
                dx = self.robot_x - self.last_pos[0]
                dy = self.robot_y - self.last_pos[1]
                moved = math.hypot(dx, dy)
                if moved > STUCK_MOVE_THRESH:
                    self.last_movement_time = now
                    self.last_pos = (self.robot_x, self.robot_y)
                else:
                    if now - self.last_movement_time > STUCK_TIME:
                        self.get_logger().warning("🛑 Stuck detected -> performing recovery")
                        self.start_recovery()
                        self.cmd_pub.publish(self.recovery_cmd())
                        self.last_cmd = self.recovery_cmd()
                        return

        # If there is a current valid path, follow it
        if self.current_path and len(self.current_path_world) > 0:
            # emergency obstacle check while following path
            """if hasattr(self, "front") and self.front < OBSTACLE_FRONT_THRESH:
                self.get_logger().info("🚧 Obstacle detected while following path -> replan")
                self.current_path = []
                self.current_path_world = []
                self.path_goal = None
                # publish a stop to help SLAM and lidar update
                self.cmd_pub.publish(Twist())
                return"""

            cmd = self.follow_path()
            self.cmd_pub.publish(cmd)
            self.last_cmd = cmd
            return

        # otherwise plan a new path to a reachable frontier
        if time.time() - self.last_plan_time > REPLAN_INTERVAL or not self.current_path:
            planned = self.plan_to_frontier(farthest=False)
            self.last_plan_time = time.time()
            if not planned:
                # nothing to plan (no reachable frontiers)
                self.get_logger().info("🎉 Map complete (or no reachable frontiers). Stopping.")
                self.cmd_pub.publish(Twist())
                return

        # In case planning returned a path but we didn't publish yet, publish stop as default
        self.cmd_pub.publish(Twist())

    # ---------------- LIDAR ----------------
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        self.ranges_full = ranges
        N = len(ranges)
        self.front = np.min(ranges[int(N*0.45):int(N*0.55)])
        self.left  = np.min(ranges[int(N*0.25):int(N*0.45)])
        self.right = np.min(ranges[int(N*0.55):int(N*0.75)])

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

        # Optionally reduce inflation near narrow hallways
        # Example: detect narrow corridors by checking free neighbors
        corridor_mask = np.zeros_like(binmap, dtype=bool)
        for y in range(1,H-1):
            for x in range(1,W-1):
                if binmap[y,x] == 0:
                    # Count free neighbors (4-connectivity)
                    free_count = np.sum(binmap[y-1:y+2, x-1:x+2]==0)
                    if free_count <= 3:
                        corridor_mask[y,x] = True

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
                    # reduce inflation in corridor cells
                    if not corridor_mask[ny, nx]:
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
                if M[y, x] != 0:
                    continue
                # check 8-neighborhood for unknown
                neigh = M[y-1:y+2, x-1:x+2]
                if np.any(neigh == UNKNOWN_VAL):
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
        if self.map_data is None or self.inflated_map is None:
            return False

        H, W = self.map_height, self.map_width

        # gather frontier cells (free cells adjacent to unknown)
        frontiers = self.get_frontier_cells()
        
        # filter out frontiers that are already marked visited
        if self.visited_mask is not None:
            filtered = []
            for (fx, fy) in frontiers:
                # skip visited
                if 0 <= fy < self.map_height and 0 <= fx < self.map_width and self.visited_mask[fy, fx]:
                    continue
                # skip temporarily-blacklisted failed frontiers
                if (fx, fy) in self.failed_frontiers and self.failed_frontiers[(fx, fy)] >= self.FAILED_LIMIT:
                    # optionally drop stale entries
                    ts = self._failed_timestamps.get((fx,fy), 0)
                    if time.time() - ts > self.FAILED_CLEAR_TIME:
                        self.failed_frontiers.pop((fx,fy), None)
                        self._failed_timestamps.pop((fx,fy), None)
                    else:
                        continue
                filtered.append((fx, fy))
            frontiers = filtered

        if not frontiers:
            return False

        self.get_logger().info(f"Found frontiers: {len(frontiers)}  Inflated occupied: {np.sum(self.inflated_map==1)}")

        if not frontiers:
            return False

        # Robot position in grid (gx,gy)
        rx, ry = self.world_to_grid(self.robot_x, self.robot_y)
        # sanity: ensure robot cell is free in inflated map
        robot_inflated = self.inflated_map[ry, rx] if self.inflated_map is not None else 1
        self.get_logger().info(f"Robot grid = {(rx, ry)}  inflated_occupied={robot_inflated}")

        # compute heading / exploration direction in grid units
        heading_world = (math.cos(self.robot_yaw), math.sin(self.robot_yaw))
        # convert heading to grid-scale (same axis, only scale matters for dot products)
        heading_grid = heading_world

        # If we have a persistent exploration_dir, use it; otherwise use current heading.
        if self.exploration_dir is not None:
            explore_dir = self.exploration_dir
        else:
            explore_dir = heading_grid

        # Build arrays for vectorized scoring
        F = np.array(frontiers, dtype=float)   # columns [x, y] grid coords
        dx = F[:,0] - rx
        dy = F[:,1] - ry
        dist = np.hypot(dx, dy)
        # avoid division by zero
        with np.errstate(invalid='ignore'):
            ux = np.where(dist > 0, dx / dist, 0.0)
            uy = np.where(dist > 0, dy / dist, 0.0)

        # dot product between exploration direction and frontier direction
        ex, ey = explore_dir
        dots = ux * ex + uy * ey  # cos(theta) in [-1,1]

        # prefer frontiers that are in front (dot large) and reasonably far (so we make progress)
        # score = dot * (1 + dist_norm)  where dist_norm is normalized [0,1] to emphasize farther frontiers
        if len(dist) > 0:
            maxd = max(1.0, float(np.max(dist)))
            dist_norm = dist / maxd
        else:
            dist_norm = dist

        scores = dots * (1.0 + dist_norm)  # tune this formula if needed

        # First attempt: candidates ahead of robot within angle cone
        cos_angle_limit = math.cos(self.EXPLORE_ANGLE_LIMIT)
        ahead_idxs = np.where(dots >= cos_angle_limit)[0]
        weak_idxs = np.where(dots >= 0.0)[0]   # <-- NEW: forward hemisphere

        order = None
        if ahead_idxs.size > 0:
            # 1) strong forward cone
            order = ahead_idxs[np.argsort(scores[ahead_idxs])[::-1]]
        elif weak_idxs.size > 0:
            # 2) weak forward hemisphere — prevents corner oscillation
            order = weak_idxs[np.argsort(scores[weak_idxs])[::-1]]
        else:
            # 3) ONLY NOW allow backward frontiers
            self.exploration_dir = None   # <-- CRITICAL: reset
            order = np.argsort(scores)[::-1]

        tried = 0
        for idx in order:
            fx, fy = int(F[idx,0]), int(F[idx,1])
            tried += 1

            # skip frontiers that are too close to robot (same or adjacent cells)
            if dist[idx] < MIN_FRONTIER_GRID_DIST:
                if tried <= 10:
                    self.get_logger().debug(f"Skipping near frontier {(fx,fy)} dist={dist[idx]}")
                continue

            # skip if the frontier cell itself is not free in inflated map (safety)
            if not self.is_cell_free_inflated(fx, fy):
                self.get_logger().debug(f"Frontier {(fx,fy)} is blocked by inflation.")
                continue

            # Attempt A* directly to the frontier cell (free cell)
            start = (rx, ry)
            goal = (fx, fy)
            path = self.astar_grid(start, goal)
            if path:
                # store path (grid coords)
                self.current_path = path
                self.current_path_world = [self.grid_to_world(gx, gy) for gx, gy in path]
                self.path_goal = goal
                self.get_logger().info(f"📍 Planned path to frontier goal grid={goal} path_len={len(path)} tried={tried}")

                # set persistent exploration direction if not yet set OR if this frontier strongly aligns with current heading
                dir_vec = (fx - rx, fy - ry)
                dnorm = math.hypot(dir_vec[0], dir_vec[1])
                if dnorm > 0:
                    dir_unit = (dir_vec[0] / dnorm, dir_vec[1] / dnorm)
                    # If no exploration_dir yet, set it to this direction.
                    """if self.exploration_dir is None:
                        self.exploration_dir = dir_unit
                        self.get_logger().info(f"➡️ Setting exploration_dir to {self.exploration_dir}")
                    else:
                        # optionally refresh exploration_dir to keep roughly same direction:
                        # if new chosen frontier is reasonably aligned, update exploration_dir slightly (low-pass)
                        if (dir_unit[0]*self.exploration_dir[0] + dir_unit[1]*self.exploration_dir[1]) > 0.5:
                            # low-pass update to smooth drift (alpha small)
                            alpha = 0.2
                            ex_u = alpha*dir_unit[0] + (1-alpha)*self.exploration_dir[0]
                            ey_u = alpha*dir_unit[1] + (1-alpha)*self.exploration_dir[1]
                            norm = math.hypot(ex_u, ey_u)
                            if norm > 0:
                                self.exploration_dir = (ex_u / norm, ey_u / norm)"""
                    # DO NOT update exploration_dir here.
                    # It will be set when the robot actually arrives at the frontier target.
                    pass

                # reset stuck detection trackers
                self.last_pos = (self.robot_x, self.robot_y)
                self.last_movement_time = time.time()
                return True
            else:
                self.get_logger().debug(f"A* failed for frontier {(fx,fy)} (tried {tried})")
                # track failures
                key = (fx, fy)
                self.failed_frontiers[key] = self.failed_frontiers.get(key, 0) + 1
                self._failed_timestamps[key] = time.time()
                continue

        # If we reach here we found no reachable frontier in ahead cone: try any frontier by increasing relax fallback
        # Fallback 2: try all frontiers sorted by distance ascending (local)
        order2 = np.argsort(dist)
        for idx in order2:
            fx, fy = int(F[idx,0]), int(F[idx,1])
            if dist[idx] < MIN_FRONTIER_GRID_DIST:
                continue
            if not self.is_cell_free_inflated(fx, fy):
                continue
            start = (rx, ry)
            goal = (fx, fy)
            path = self.astar_grid(start, goal)
            if path:
                self.current_path = path
                self.current_path_world = [self.grid_to_world(gx, gy) for gx, gy in path]
                self.path_goal = goal
                self.get_logger().info(f"📍 Fallback planned path to frontier grid={goal} path_len={len(path)}")
                # reset exploration_dir (we might want to change direction when forced)
                dir_vec = (fx - rx, fy - ry)
                dnorm = math.hypot(dir_vec[0], dir_vec[1])
                if dnorm > 0:
                    self.exploration_dir = (dir_vec[0]/dnorm, dir_vec[1]/dnorm)
                self.last_pos = (self.robot_x, self.robot_y)
                self.last_movement_time = time.time()
                return True

        # no reachable frontier found
        self.get_logger().info("No reachable frontier found after trying candidates.")
        return False


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
    #                PATH FOLLOWER
    # -----------------------------------------------------
    def follow_path(self):
        """
        Follow current_path_world waypoints. Returns Twist command.
        """
        if not self.current_path_world:
            return Twist()

        # next waypoint in world coords
        wx, wy = self.current_path_world[0]

        dx = wx - self.robot_x
        dy = wy - self.robot_y
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        yaw_error = target_angle - self.robot_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        cmd = Twist()

        # If close to waypoint, pop it
        if dist < GOAL_NEAR_DIST_M:
            self.current_path_world.pop(0)
            self.current_path.pop(0)
            # if path finished
            if not self.current_path_world:
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

                # optionally clear failed_frontiers entries near the goal (they might now be reachable/irrelevant)
                to_delete = []
                if gx is not None:
                    for f in list(self.failed_frontiers.keys()):
                        fx, fy = f
                        if (fx - gx)**2 + (fy - gy)**2 <= (rr*rr):
                            to_delete.append(f)
                for f in to_delete:
                    self.failed_frontiers.pop(f, None)
                    self._failed_timestamps.pop(f, None)

                return Twist()



        # Safety override: if obstacle directly in front, rotate in place
        if hasattr(self, "front") and self.front < 0.28:
            cmd.linear.x = 0.0
            # rotate away from obstacle using side readings
            if hasattr(self, "left") and hasattr(self, "right"):
                if self.left < self.right:
                    cmd.angular.z = -0.5
                else:
                    cmd.angular.z = 0.5
            else:
                cmd.angular.z = 0.5
            return cmd

        # rotate to face waypoint first if yaw error large
        if abs(yaw_error) > ALIGN_ANGLE:
            # rotate proportionally, no forward
            ang = 1.2 * yaw_error
            ang = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, ang))
            cmd.angular.z = ang
            cmd.linear.x = 0.0
            return cmd

        # else drive forward with small angular correction
        cmd.linear.x = min(MAX_LINEAR_SPEED, 0.5 * dist + 0.02)
        # damp speed when near obstacles
        if hasattr(self, "front") and self.front < 0.6:
            cmd.linear.x = min(cmd.linear.x, 0.06)
        cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, 1.0*yaw_error))
        return cmd

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
