#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Point
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Header

import numpy as np
import math
from collections import deque
import heapq
from cv_bridge import CvBridge
import cv2 as cv

# tuning
INFLATE_RADIUS_CELLS = 5        # inflate obstacles by this many cells
OCCUPIED_THRESH = 65            # map values >= this are occupied
LOOKAHEAD_M = 0.30
FRONT_OBSTACLE_THRESHOLD = 0.6  # m -> treat as obstacle blocking
FRONT_ANGLE_DEG = 25             # degrees on each side for front sector
MAX_LINEAR_SPEED = 0.16
MAX_ANGULAR_SPEED = 0.8
GOAL_NEAR_DIST_M = 0.25
LIDAR_FORWARD_OFFSET = math.radians(0)

# Safety limit to prevent A* running forever
ASTAR_MAX_EXPANSIONS = 400000


class AStarGridSolver:
    """Simple A* solver on an occupancy grid (0 free, 1 occupied).
    Uses 8-connected neighbors by default but can be set to 4-connected.
    Returns list of (gx,gy) nodes (gx=col, gy=row) or None if not found.
    """
    def __init__(self, map_bin, use_diagonals=True):
        self.map = map_bin
        self.H, self.W = self.map.shape
        if use_diagonals:
            self.neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        else:
            self.neighbors = [(1,0),(-1,0),(0,1),(0,-1)]

    def heuristic(self, a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def solve(self, start, goal):
        # start,goal are (gx,gy) where gx is column, gy is row
        sx, sy = start
        gx, gy = goal
        if not (0 <= sx < self.W and 0 <= sy < self.H and 0 <= gx < self.W and 0 <= gy < self.H):
            return None
        if self.map[sy, sx] == 1 or self.map[gy, gx] == 1:
            return None

        open_heap = []
        gscore = {}
        came_from = {}
        start_key = (sx, sy)
        goal_key = (gx, gy)

        gscore[start_key] = 0.0
        heapq.heappush(open_heap, (self.heuristic(start_key, goal_key), 0.0, start_key))
        closed = set()
        expansions = 0

        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_key:
                # reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start_key)
                path.reverse()
                return path

            closed.add(current)
            expansions += 1
            if expansions > ASTAR_MAX_EXPANSIONS:
                # give up to avoid freezing
                return None

            cx, cy = current
            for dx, dy in self.neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.W and 0 <= ny < self.H):
                    continue
                if self.map[ny, nx] == 1:
                    continue
                tentative_g = g + math.hypot(dx, dy)
                key = (nx, ny)
                if tentative_g < gscore.get(key, 1e9):
                    gscore[key] = tentative_g
                    came_from[key] = current
                    heapq.heappush(open_heap, (tentative_g + self.heuristic(key, goal_key), tentative_g, key))

        return None


class Task3(Node):
    def __init__(self):
        super().__init__('task3_algorithm')

        # QoS
        qos = QoSProfile(depth=10)

        # Map must be subscribed with TRANSIENT_LOCAL durability to receive the latched map
        from rclpy.qos import ReliabilityPolicy, DurabilityPolicy
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, qos)
        # Keep RViz goal subscription (manual override still allowed)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, qos_profile=map_qos)

        # Camera image for detection
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image_raw', self.cam_cb, cam_qos)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        self.path_pub = self.create_publisher(Path, 'global_plan', qos)

        self.red_pub   = self.create_publisher(Point, "/red_pos", 10)
        self.green_pub = self.create_publisher(Point, "/green_pos", 10)
        self.blue_pub  = self.create_publisher(Point, "/blue_pos", 10)

        # State
        self.map_msg = None
        self.map_arr = None            # numpy array [H, W] flipped (row0 top)
        self.map_loaded = False
        self.inf_map = None            # inflated binary map: 1->obstacle, 0->free
        self.map_res = None
        self.map_origin = (0.0, 0.0)
        self.map_width = 0
        self.map_height = 0

        self.pose = None               # geometry_msgs/Pose
        self.goal = None               # PoseStamped
        self.global_path = None        # nav_msgs/Path
        self.path_world = []           # list of (x,y)
        self.last_path_idx = 0

        self.lidar = None              # last LaserScan
        self.ranges = None

        # search-specific
        self.search_points = deque()
        self.search_generated = False
        self.current_search = None

        self.frame_skip = 3
        self.frame_count = 0


        self.detected_balls = {
            "red": None,
            "green": None,
            "blue": None
        }

        # detection
        self.found = {}   # {"red": (x,y), "green": (x,y), "blue": (x,y)}
        # HSV ranges (low, high)
        self.color_ranges = {
            "red":   [(0, 170, 120), (10, 255, 255)],
            "green": [(40, 120, 120), (80, 255, 255)],
            "blue":  [(90, 170, 120), (130, 255, 255)]
        }

        # timer loops
        self.timer = self.create_timer(0.1, self.timer_cb)
        self.status_timer = self.create_timer(1.0, self.print_ball_status)

        self.get_logger().info("Task3 node started (search + localize).")

    # ---------------- callbacks ----------------
    def map_cb(self, msg: OccupancyGrid):
        try:
            # defensive reshape: check sizes
            self.map_msg = msg
            H = int(msg.info.height)
            W = int(msg.info.width)
            flat = np.array(msg.data, dtype=int)
            if flat.size != H * W:
                self.get_logger().error(f"Map data length mismatch: data={flat.size}, expected={H*W}")
                return
            arr = flat.reshape((H, W))

            # flip up-down so row0 = top of image (consistent with previous code)
            arr = np.flipud(arr)

            self.map_arr = arr.copy()
            self.map_res = float(msg.info.resolution)
            # origin: note map origin in OccupancyGrid is the real-world coord of the map cell (0,0)
            self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            self.map_width = W
            self.map_height = H

            # build inflated binary map (1 occupied, 0 free)
            binmap = np.zeros_like(arr, dtype=np.uint8)
            binmap[arr >= OCCUPIED_THRESH] = 1
            self.inf_map = self.inflate(binmap, INFLATE_RADIUS_CELLS)
            self.map_loaded = True
            self.get_logger().info(f"Map received: size=({H},{W}), res={self.map_res:.3f}")

            # generate search points ONCE
            if not self.search_generated:
                self.generate_search_points()
                self.search_generated = True

        except Exception as e:
            self.get_logger().error(f"Exception in map_cb: {e}")

    def amcl_cb(self, msg: PoseWithCovarianceStamped):
        # store Pose (geometry_msgs/Pose)
        self.pose = msg.pose.pose

    def goal_cb(self, msg: PoseStamped):
        # manual RViz goal still supported — it will override automatic search goal
        self.get_logger().info("New goal received from RViz.")
        self.goal = msg
        self.get_logger().info(f"Goal (world): {msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}")
        # attempt an immediate plan
        ok = self.plan_path()
        self.get_logger().info(f"Plan result: {ok}")

    def scan_cb(self, msg: LaserScan):
        self.lidar = msg
        self.ranges = np.array(msg.ranges, dtype=float)
        self.ranges = np.where(np.isfinite(self.ranges), self.ranges, np.inf)

    """def cam_cb(self, msg: Image):
        # Camera callback: detect colored balls and localize using area-based distance
        if self.pose is None or self.map_arr is None or len(self.found) >= 3:
            return
        
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        cx_frame = w // 2

        # Camera constants
        FOV = math.radians(62.2)
        F = (w / 2) / math.tan(FOV / 2)     # focal length in pixels
        BALL_RADIUS_M = 0.07                # 7 cm radius

        for color, (lo, hi) in self.color_ranges.items():
            if color in self.found:
                continue

            # Create mask
            mask = cv.inRange(hsv, np.array(lo), np.array(hi))

            # Clean noise
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((5, 5), np.uint8))

            cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            # Largest blob
            c = max(cnts, key=cv.contourArea)
            area = cv.contourArea(c)
            if area < 120:      # smaller threshold now that we're detecting at distance
                continue

            # Bounding box
            x, y, bw, bh = cv.boundingRect(c)
            cx = x + bw / 2
            cy = y + bh / 2

            # Pixel radius (approx)
            radius_pixels = bw / 2
            if radius_pixels < 4:
                continue

            # --- ANGLE from center of camera ---
            angle = (cx - cx_frame) / cx_frame * (FOV / 2)

            # --- DISTANCE using camera projection ---
            distance = (BALL_RADIUS_M * F) / radius_pixels

            # Robot pose
            rx = self.pose.position.x
            ry = self.pose.position.y
            yaw = self.quat_to_yaw(self.pose.orientation)

            # Ball world coordinates
            bx = rx + distance * math.cos(yaw + angle)
            by = ry + distance * math.sin(yaw + angle)

            # Store it
            self.found[color] = (bx, by)
            self.detected_balls[color] = (bx, by)
            self.get_logger().info(
                f"FOUND {color.upper()} ball at ({bx:.2f}, {by:.2f}) | dist={distance:.2f} m"
            )

            if len(self.found) >= 3:
                self.get_logger().info("All balls located — stopping robot.")
                self.stop_robot()
                self.global_path = None
                self.path_world = []
                self.goal = None

            # Detect only one color per frame
            break"""
    
    def cam_cb(self, msg: Image):
        """FAST + SIMPLE + RELIABLE camera callback based on working RedBallTracker logic."""

        # Early exit
        if self.pose is None or self.map_arr is None or len(self.found) >= 3:
            return

        # Frame skipping BEFORE expensive work
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return

        # Convert ROS → CV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return

        h, w = frame.shape[:2]
        cx_frame = w // 2

        # Convert to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # ---------- SAME LOGIC YOU KNOW WORKS ----------
        for color, (lo, hi) in self.color_ranges.items():
            if color in self.found:
                continue

            # Mask
            mask = cv.inRange(hsv, np.array(lo), np.array(hi))

            # Clean-up mask (your working logic)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((5, 5), np.uint8))

            # Find contours
            cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            # Largest contour
            c = max(cnts, key=cv.contourArea)
            area = cv.contourArea(c)
            if area < 500:  # conservative threshold
                continue

            x, y, w_box, h_box = cv.boundingRect(c)
            cx = x + w_box / 2
            cy = y + h_box / 2

            # Pixel radius approximation
            radius_pixels = w_box / 2
            if radius_pixels < 5:
                continue

            # ---------- Distance + angle ----------
            FOV = math.radians(62.2)
            F = (w / 2) / math.tan(FOV / 2)
            BALL_RADIUS_M = 0.07

            angle = (cx - cx_frame) / cx_frame * (FOV / 2)
            distance = (BALL_RADIUS_M * F) / radius_pixels

            # ---------- Convert to world coordinates ----------
            rx = self.pose.position.x
            ry = self.pose.position.y
            yaw = self.quat_to_yaw(self.pose.orientation)

            bx = rx + distance * math.cos(yaw + angle)
            by = ry + distance * math.sin(yaw + angle)

            # Store result
            self.found[color] = (bx, by)
            self.detected_balls[color] = (bx, by)
            self.get_logger().info(
                f"FOUND {color.upper()} ball at ({bx:.2f}, {by:.2f}), dist={distance:.2f}m"
            )

            # Stop on all 3
            if len(self.found) == 3:
                self.stop_robot()
                self.global_path = None
                self.path_world = []
                self.goal = None

            break  # only detect one color per frame



    # ---------------- main loop ----------------
    def timer_cb(self):
        # need a pose to do anything
        if self.pose is None:
            return

        # If map not yet loaded, wait
        if not self.map_loaded:
            return

        # If we've found all balls, ensure robot stopped and do nothing
        if len(self.found) >= 3:
            return

        # If no manual goal & no automatic goal assigned, pick next search point
        if self.goal is None:
            self.assign_next_search_goal()
            return

        # If we have no planned path, attempt to plan
        if self.global_path is None:
            ok = self.plan_path()
            if not ok:
                self.get_logger().warn("Initial planning failed; will retry in timer loop.")
            return

        # Obstacle detection
        if self.is_obstacle_blocking_next_waypoint():
            self.get_logger().warn("Obstacle blocking path — replanning")
            # temporary injection and replan handled inside plan_path()
            self.plan_path(local_replan=True)
            return

        # follow path
        self.follow_path()

     # ---------------- map helpers ----------------
    def inflate(self, binmap, radius_cells):
        H, W = binmap.shape
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
        """Convert world (x,y) to grid indices (gx,gy) where gx=col, gy=row"""
        res = self.map_res
        ox, oy = self.map_origin
        j = int((wx - ox) / res)   # column (gx)
        i = self.map_height - int((wy - oy) / res) - 1  # row (gy)
        return j, i

    def grid_to_world(self, gx, gy):
        ox, oy = self.map_origin
        res = self.map_res
        x = gx * res + ox + res/2.0
        y = (self.map_height - gy - 1) * res + oy + res/2.0
        return x, y

    def find_nearest_free(self, g):
        """BFS to nearest free cell on self.inf_map. Input and output are (gx,gy)."""
        from collections import deque
        if self.inf_map is None:
            return g
        start_gx, start_gy = g
        W = self.map_width
        H = self.map_height
        if 0 <= start_gx < W and 0 <= start_gy < H and self.inf_map[start_gy, start_gx] == 0:
            return (start_gx, start_gy)
        q = deque([(start_gx, start_gy)])
        visited = set([(start_gx, start_gy)])
        while q:
            gx, gy = q.popleft()
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    nx, ny = gx + dx, gy + dy
                    if not (0 <= nx < W and 0 <= ny < H):
                        continue
                    if (nx, ny) in visited:
                        continue
                    visited.add((nx, ny))
                    if self.inf_map[ny, nx] == 0:
                        return (nx, ny)
                    q.append((nx, ny))
        return (start_gx, start_gy)

    # ---------------- A* planner (grid) ----------------
    def astar_grid(self, start_g, goal_g, map_bin=None):
        if map_bin is None:
            map_bin = self.inf_map
        if map_bin is None:
            return None
        solver = AStarGridSolver(map_bin, use_diagonals=True)
        path = solver.solve(start_g, goal_g)
        return path

    # ---------------- planning / replanning ----------------
    def plan_path(self, local_replan=False):
        if not self.map_loaded or self.pose is None or self.goal is None:
            missing = []
            if not self.map_loaded:
                missing.append("map_loaded")
            if self.pose is None:
                missing.append("pose")
            if self.goal is None:
                missing.append("goal")
            self.get_logger().warn(f"plan_path() missing: {', '.join(missing)} -> deferring planning")
            return False

        # convert poses to grid
        sx, sy = self.pose.position.x, self.pose.position.y
        gx_w, gy_w = self.goal.pose.position.x, self.goal.pose.position.y

        start_g = self.world_to_grid(sx, sy)
        goal_g = self.world_to_grid(gx_w, gy_w)

        # snap to nearest free if necessary
        start_g = self.find_nearest_free(start_g)
        goal_g  = self.find_nearest_free(goal_g)

        self.get_logger().info(f"start_g={start_g}, goal_g={goal_g}")
        try:
            self.get_logger().info(f"start_cell_value={self.inf_map[start_g[1], start_g[0]]}")
            self.get_logger().info(f"goal_cell_value={self.inf_map[goal_g[1], goal_g[0]]}")
        except Exception:
            self.get_logger().warn("start/goal outside inf_map bounds")

        # working map for local replan
        working_map = self.inf_map.copy()

        if local_replan and self.is_obstacle_blocking_next_waypoint():
            ox, oy = self.find_closest_front_point()
            if ox is not None:
                cgx, cgy = self.world_to_grid(ox, oy)
                r = max(1, int(0.55 / self.map_res))   # radius in cells
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        # circle mask
                        if dx*dx + dy*dy > r*r:
                            continue
                        nx = cgx + dx
                        ny = cgy + dy
                        if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                            working_map[ny, nx] = 1

        # run A*
        path = self.astar_grid(start_g, goal_g, map_bin=working_map)

        if path is None:
            self.get_logger().warn("A* failed to find a path.")
            self.global_path = None
            self.path_world = []
            """if self.find_closest_front_point > 2.0:
                speed = 0.1
                heading = 0.0
                self.move_ttbot_safe(speed, heading)"""
            return False

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

        self.global_path = path_msg
        self.last_path_idx = 0
        self.path_pub.publish(self.global_path)
        self.get_logger().info(f"A* planned path with {len(self.path_world)} waypoints.")
        return True

    # ---------------- obstacle helpers ----------------
    def is_obstacle_blocking_next_waypoint(self):
        if self.lidar is None or not self.path_world:
            return False

        idx = max(0, min(self.last_path_idx, len(self.path_world)-1))
        next_idx = min(idx + 2, len(self.path_world)-1)
        wx, wy = self.path_world[next_idx]

        rx, ry = self.pose.position.x, self.pose.position.y
        vx = wx - rx
        vy = wy - ry
        dist_to_way = math.hypot(vx, vy)
        """if dist_to_way < 0.1:
            return False"""

        desired_yaw = math.atan2(vy, vx)
        if self.lidar is None:
            return False
        angle_min = self.lidar.angle_min
        angle_inc = self.lidar.angle_increment
        N = len(self.ranges)

        w = math.radians(FRONT_ANGLE_DEG)
        center = LIDAR_FORWARD_OFFSET
        idx0 = int((center - w - angle_min) / angle_inc)
        idx1 = int((center + w - angle_min) / angle_inc)
        idx0 = max(0, min(N-1, idx0))
        idx1 = max(0, min(N-1, idx1))
        sector = self.ranges[idx0:idx1+1]

        if sector.size == 0:
            return False
        min_front = float(np.min(sector))
        #self.get_logger().info(f"Minimum distance: {min_front}")
        if min_front < FRONT_OBSTACLE_THRESHOLD:
            #return True
            rel_idx = int(np.argmin(sector))
            overall_idx = idx0 + rel_idx
            angle_lidar = angle_min + overall_idx * angle_inc
            angle_robot = angle_lidar + LIDAR_FORWARD_OFFSET
            angle_robot = (angle_robot + math.pi) % (2*math.pi) - math.pi

            robot_yaw = self.quat_to_yaw(self.pose.orientation)
            bearing_to_way = desired_yaw - robot_yaw
            bearing_to_way = (bearing_to_way + math.pi) % (2*math.pi) - math.pi
            if abs(angle_robot - bearing_to_way) <= math.radians(70):
                return True
        return False

    def find_closest_front_point(self):
        if self.lidar is None:
            return (None, None)
        angle_min = self.lidar.angle_min
        angle_inc = self.lidar.angle_increment
        N = len(self.ranges)
        w = math.radians(FRONT_ANGLE_DEG)

        center = LIDAR_FORWARD_OFFSET
        idx0 = int((center - w - angle_min) / angle_inc)
        idx1 = int((center + w - angle_min) / angle_inc)
        idx0 = max(0, min(N-1, idx0))
        idx1 = max(0, min(N-1, idx1))
        sector = self.ranges[idx0:idx1+1]
        

        if sector.size == 0:
            return (None, None)
        rel_idx = int(np.argmin(sector))
        overall_idx = idx0 + rel_idx
        r = float(self.ranges[overall_idx])
        if not math.isfinite(r) or r > 5.0:
            return (None, None)
        angle_lidar = angle_min + overall_idx * angle_inc
        angle_corrected = angle_lidar + LIDAR_FORWARD_OFFSET

        rx = r * math.cos(angle_corrected)
        ry = r * math.sin(angle_corrected)

        robot_x = self.pose.position.x
        robot_y = self.pose.position.y
        robot_yaw = self.quat_to_yaw(self.pose.orientation)
        wx = robot_x + (rx * math.cos(robot_yaw) - ry * math.sin(robot_yaw))
        wy = robot_y + (rx * math.sin(robot_yaw) + ry * math.cos(robot_yaw))
        return (wx, wy)

    # ---------------- path following ----------------
    def follow_path(self):
        if not self.path_world:
            return
        idx = self.get_path_idx(self.path_world, self.last_path_idx)
        idx = max(0, min(idx, len(self.path_world)-1))
        gx, gy = self.path_world[idx]
        speed, heading, dist = self.path_follower_from_xy(gx, gy)
        self.move_ttbot_safe(speed, heading)
        self.last_path_idx = idx

        if idx >= len(self.path_world)-1 and dist < GOAL_NEAR_DIST_M:
            self.get_logger().info("Reached goal.")
            # clear current navigation goal so timer_cb can assign another
            self.global_path = None
            self.path_world = []
            self.goal = None
            self.stop_robot()

    def get_path_idx(self, path_world, last_idx=0):
        """
        Return next path index based on lookahead distance.
        path_world: list of (x,y)
        """
        if not path_world:
            return 0
        vx = self.pose.position.x
        vy = self.pose.position.y
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
        cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def path_follower_from_xy(self, goal_x, goal_y):
        # Current pose
        rx = self.pose.position.x
        ry = self.pose.position.y
        yaw = self.quat_to_yaw(self.pose.orientation)

        # Predict forward in time to compensate for latency
        # (0.15–0.30 seconds works great)
        latency_comp = 0.18
        rx_pred = rx + math.cos(yaw) * latency_comp * 0.12
        ry_pred = ry + math.sin(yaw) * latency_comp * 0.12

        # Compute heading to waypoint using predicted pose
        dx = goal_x - rx_pred
        dy = goal_y - ry_pred
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)

        # Heading error
        err = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        # Smooth heading error (tolerant to delayed pose)
        alpha = 0.35
        self.err_smoothed = alpha * err + (1 - alpha) * getattr(self, "err_smoothed", 0.0)

        # Derivative (to prevent oscillation)
        d_err = self.err_smoothed - getattr(self, "err_prev", 0.0)
        self.err_prev = self.err_smoothed

        # Gains tuned for latency environments
        Kp = 1.6
        Kd = 0.3

        ang = Kp * self.err_smoothed + Kd * d_err

        # Deadzone for micro-errors
        if abs(ang) < 0.05:
            ang = 0.0

        # Clamp
        ang = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, ang))

        # Auto-slowing: robot travels slower when turning
        speed_factor = 1.0 - min(0.8, abs(ang) * 1.2)
        lin = MAX_LINEAR_SPEED * speed_factor

        # Apply obstacle-based limit
        """obs_lim = self.compute_obstacle_speed_limit()
        lin = min(lin, obs_lim)"""

        # Safety slowdown on close path points
        if dist < 0.20:
            lin = min(lin, 0.06)

        # Final clamp
        lin = max(0.02, min(MAX_LINEAR_SPEED, lin))

        return lin, ang, dist
    
    def compute_obstacle_speed_limit(self):
        if self.ranges is None:
            return MAX_LINEAR_SPEED

        angle_min = self.lidar.angle_min
        angle_inc = self.lidar.angle_increment
        N = len(self.ranges)

        # Use a much wider arc than obstacle detection
        left = math.radians(70)
        right = math.radians(-70)

        idx0 = int((right - angle_min) / angle_inc)
        idx1 = int((left  - angle_min) / angle_inc)

        idx0 = max(0, min(N-1, idx0))
        idx1 = max(0, min(N-1, idx1))

        sector = self.ranges[idx0:idx1+1]
        if sector.size == 0:
            return MAX_LINEAR_SPEED

        d = float(np.min(sector))
        self.get_logger().info(f"Minimum distance: {d}")

        # Convert distance to speed limits
        if d > 1.5:
            return MAX_LINEAR_SPEED
        elif d > 1.0:
            return 0.12
        elif d > 0.7:
            return 0.08
        elif d > 0.4:
            return 0.04
        else:
            return 0.0

    def move_ttbot_safe(self, speed, heading):
        cmd = Twist()
        cmd.linear.x = max(0.0, min(speed, MAX_LINEAR_SPEED))
        cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(heading, MAX_ANGULAR_SPEED))
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    # ---------------- search helpers ---------------- 
    def generate_search_points_work(self):
        
        #Robust room segmentation with fallback.
        #1. Close narrow halls -> segment rooms.
        #2. Extract large connected regions.
        #3. Use region centroid as a search target.
        #4. If segmentation fails, fall back to grid sampling.
        
        if self.inf_map is None:
            return

        H, W = self.inf_map.shape
        free = (self.inf_map == 0).astype(np.uint8)

        # ---- STEP 1: CLOSE DOORS & HALLWAYS ----
        # This kernel closes ~0.3m gaps (tunable)
        kernel = np.ones((7, 7), np.uint8)
        closed = cv.morphologyEx(free, cv.MORPH_CLOSE, kernel, iterations=2)

        # ---- STEP 2: CONNECTED COMPONENTS ----
        num, labels = cv.connectedComponents(closed)

        pts = []

        for label in range(1, num):
            ys, xs = np.where(labels == label)
            size = len(xs)

            # Ignore tiny hallway fragments
            if size < 500:
                continue

            # Compute region centroid
            cgx = int(np.mean(xs))
            cgy = int(np.mean(ys))

            # Convert to world
            wx, wy = self.grid_to_world(cgx, cgy)

            # Double check it's valid in the *real* free map
            if self.inf_map[cgy, cgx] == 0:
                pts.append((wx, wy))

        # ---- STEP 3: GUARANTEE NON-EMPTY RESULT ----
        if len(pts) == 0:
            self.get_logger().warn("Room segmentation failed → falling back to grid sampling.")
            pts = []
            step = max(1, int(2.0 / self.map_res))  # about 2 m spacing
            for gy in range(0, H, step):
                for gx in range(0, W, step):
                    if self.inf_map[gy, gx] == 0:
                        xw, yw = self.grid_to_world(gx, gy)
                        pts.append((xw, yw))

        # ---- STEP 4: FINAL GUARANTEE ----
        if len(pts) == 0:
            self.get_logger().error("NO SEARCH POINTS FOUND. Map may be invalid.")
            pts = [(0, 0)]  # won't happen in practice but avoids crashing

        np.random.shuffle(pts)
        self.search_points = deque(pts)

        self.get_logger().info(f"Generated {len(pts)} search points.")
    
    def generate_search_points(self,
                            spacing_m=1.2,      # how dense in free space
                            safe_margin_m=0.15  # keep robot + inflation away from walls/balls
                            ):
        """
        SIMPLE + RELIABLE GLOBAL SWEEP SEARCH
        ------------------------------------
        - Samples the RAW MAP (self.map_arr) so doorways stay open.
        - Uses INFLATED MAP (self.inf_map) for safety margin checks.
        - Produces a deterministic boustrophedon sweep (zig-zag).
        - Guarantees full coverage of all rooms.
        - Rejects points too close to:
            * walls
            * doorframes
            * balls / obstacles (if your inflation layer marks them)
        - Always returns a point set.
        """

        from collections import deque

        # ---- safety checks ----
        if self.map_arr is None or self.inf_map is None or self.map_res is None:
            self.get_logger().warn("generate_search_points(): map not ready")
            return []

        raw = self.map_arr          # RAW occupancy (keeps doorways open)
        inf = self.inf_map          # INFLATED map (keeps us safe)
        H, W = raw.shape

        # parameters → cells
        step = max(1, int(round(spacing_m / self.map_res)))
        margin = max(1, int(round(safe_margin_m / self.map_res)))

        self.get_logger().info(
            f"[SEARCH] spacing={spacing_m}m({step} cells), margin={safe_margin_m}m({margin} cells)"
        )

        # safety check helper (inflated map)
        def is_safe(gx, gy):
            """Reject points too close to walls/balls/obstacles."""
            if gx < margin or gy < margin or gx >= W - margin or gy >= H - margin:
                return False

            # must be free in inflated map
            patch = inf[gy - margin: gy + margin + 1,
                        gx - margin: gx + margin + 1]
            return np.all(patch == 0)

        # choose sweep direction based on robot position
        start_from_top = False
        try:
            rx, ry = self.pose.position.x, self.pose.position.y
            rgx, rgy = self.world_to_grid(rx, ry)
            start_from_top = (rgy < H // 2)
            self.get_logger().info(f"[SEARCH] robot at grid=({rgx},{rgy}) start_from_top={start_from_top}")
        except:
            pass

        # -------- build sampling grid --------
        pts = []
        rows = range(margin, H - margin, step)
        cols = range(margin, W - margin, step)

        for col_i, gx in enumerate(cols):

            # zig-zag direction
            if (col_i % 2 == 0) ^ start_from_top:
                sweep_rows = rows
            else:
                sweep_rows = reversed(list(rows))

            for gy in sweep_rows:
                # must be free in RAW map
                if raw[gy, gx] >= OCCUPIED_THRESH:
                    continue

                if not is_safe(gx, gy):
                    continue

                # convert to world
                wx, wy = self.grid_to_world(gx, gy)
                pts.append((wx, wy))

        # -------- fallback if empty --------
        if not pts:
            self.get_logger().warn("[SEARCH] No points found — using coarse fallback")
            for gy in range(0, H, max(1, step)):
                for gx in range(0, W, max(1, step)):
                    if raw[gy, gx] < 50:
                        if is_safe(gx, gy):
                            pts.append(self.grid_to_world(gx, gy))

        if not pts:
            pts = [(0.0, 0.0)]
            self.get_logger().error("[SEARCH] STILL empty — fallback to (0,0)")

        # assign queue
        self.search_points = deque(pts)

        self.get_logger().info(f"[SEARCH] Generated {len(pts)} safe global sweep points")
        return pts

    
    def generate_search_points_latest(self):
        """
        Deterministic room-by-room sweeping search.
        - free = occ < 50
        - morphological closing to merge small gaps
        - connected components -> rooms
        - rooms sorted left→right→top→bottom
        - 3×3 interior safe samples in each room
        - fallback to coarse grid sampling if segmentation fails
        """
        if self.inf_map is None:
            self.get_logger().warn("generate_search_points() called but inf_map is None")
            return

        occ = self.inf_map
        H, W = occ.shape

        # ---- FREE MASK USING NEW RULE ----
        free_mask = (occ < 5).astype(np.uint8)
        self.get_logger().info(f"[DEBUG] Free mask created: free_count={np.sum(free_mask)}")

        # ---- STEP 1: CLOSE SMALL DOORS/HALLWAYS ----
        kernel = np.ones((10, 10), np.uint8)
        closed = cv.morphologyEx(free_mask, cv.MORPH_CLOSE, kernel, iterations=2)
        self.get_logger().info("[DEBUG] Morphological closing complete")

        # ---- STEP 2: CONNECTED COMPONENTS (ROOM FINDING) ----
        num_labels, labels = cv.connectedComponents(closed)
        self.get_logger().info(f"[DEBUG] Connected components found: {num_labels - 1}")

        rooms = []
        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            size = len(xs)

            self.get_logger().info(f"[DEBUG] Region {label}: size={size}")

            # skip tiny rooms/hallway slivers
            if size < 500:
                continue

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            rooms.append({
                "id": label,
                "cx": float(xs.mean()),
                "cy": float(ys.mean()),
                "bbox": (x_min, x_max, y_min, y_max),
            })

        if not rooms:
            self.get_logger().warn("No rooms detected! Falling back immediately.")
            return self._fallback_grid_sampling(occ)

        # ---- SORT ROOMS LEFT→RIGHT THEN TOP→BOTTOM ----
        rooms.sort(key=lambda r: (r["cx"], r["cy"]))
        self.get_logger().info(f"[DEBUG] Rooms sorted: {len(rooms)} rooms")

        pts = []

        def is_safe(gx, gy):
            """Check if inflated map says this pixel is safe."""
            if gx < 3 or gy < 3 or gx >= W - 3 or gy >= H - 3:
                return False
            if occ[gy, gx] >= 50:
                return False
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if occ[gy + dy, gx + dx] >= 50:
                        return False
            return True

        # ---- SAMPLE EACH ROOM WITH A 3×3 INTERIOR GRID ----
        for rm in rooms:
            x_min, x_max, y_min, y_max = rm["bbox"]
            width = x_max - x_min
            height = y_max - y_min

            # interior samples
            grid_x = np.linspace(x_min + 0.2 * width, x_max - 0.2 * width, 3)
            grid_y = np.linspace(y_min + 0.2 * height, y_max - 0.2 * height, 3)

            self.get_logger().info(
                f"[DEBUG] Room {rm['id']} bbox=({x_min},{x_max},{y_min},{y_max}) generating grid samples"
            )

            for px in grid_x:
                for py in grid_y:
                    gx, gy = int(px), int(py)
                    if is_safe(gx, gy):
                        wx, wy = self.grid_to_world(gx, gy)
                        pts.append((wx, wy))
                        self.get_logger().info(
                            f"[DEBUG] Added search point at world ({wx:.2f}, {wy:.2f})"
                        )
                    else:
                        self.get_logger().info(
                            f"[DEBUG] Point ({gx},{gy}) rejected by safety check"
                        )

        # ---- FALLBACK IF NO SAMPLES ----
        if len(pts) == 0:
            self.get_logger().warn("Room sampling failed → fallback coarse grid sampling.")
            return self._fallback_grid_sampling(occ)

        # ---- FINALIZE ----
        self.search_points = deque(pts)
        self.get_logger().info(f"Generated {len(pts)} search points across {len(rooms)} rooms.")
        return pts


    # ----------------- FALLBACK FUNCTION -----------------
    def _fallback_grid_sampling(self, occ):
        """Used when room segmentation fails."""
        H, W = occ.shape
        pts = []
        step = max(1, int(2.0 / self.map_res))

        self.get_logger().warn("[DEBUG] Fallback grid sampling triggered")

        for gy in range(0, H, step):
            for gx in range(0, W, step):
                if occ[gy, gx] < 50:
                    wx, wy = self.grid_to_world(gx, gy)
                    pts.append((wx, wy))

        if len(pts) == 0:
            self.get_logger().error("NO SEARCH POINTS FOUND IN FALLBACK. USING (0,0).")
            pts = [(0, 0)]

        self.search_points = deque(pts)
        self.get_logger().info(f"[DEBUG] Fallback generated {len(pts)} points.")
        return pts
    
    def generate_search_points_slow(self):
        """
        Deterministic room-by-room sweeping search with hallway merging.
        - Uses RAW occupancy map (not inflated) for proper segmentation.
        - free = occ < 50
        - connected components → rooms
        - hallways merged into nearest large room
        - 3×3 interior grid samples per room
        """

        if self.map_arr is None:
            self.get_logger().warn("generate_search_points(): map_arr not ready")
            return

        occ = self.map_arr  # H × W, raw occupancy
        H, W = occ.shape

        # --------------------------
        # 1. Build free-space mask
        # --------------------------
        free_mask = ((occ >= 0) & (occ < 50)).astype(np.uint8)
        free_count = int(free_mask.sum())
        self.get_logger().info(f"[DEBUG] free_mask ready: free_count={free_count}")

        # --------------------------
        # 2. Connected components
        # --------------------------
        num_labels, labels = cv.connectedComponents(free_mask)
        self.get_logger().info(f"[DEBUG] Connected components detected: {num_labels}")

        regions = []
        hallway_regions = []

        # Threshold for "hallway" width or size
        HALLWAY_SIZE = 3000        # smaller → hallway-like
        HALLWAY_ASPECT_RATIO = 0.20  # narrow shapes

        # Collect regions
        for lid in range(1, num_labels):
            ys, xs = np.where(labels == lid)
            size = len(xs)

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            width = x_max - x_min
            height = y_max - y_min

            # aspect ratio for shape filtering
            aspect = min(width, height) / max(width, height)

            self.get_logger().info(
                f"[DEBUG] Region {lid}: size={size}, aspect={aspect:.2f}, bbox=({x_min},{x_max},{y_min},{y_max})"
            )

            # Hallway if small OR extremely elongated
            if size < HALLWAY_SIZE or aspect < HALLWAY_ASPECT_RATIO:
                hallway_regions.append((lid, xs, ys))
            else:
                regions.append((lid, xs, ys))

        self.get_logger().info(f"[DEBUG] Large regions (rooms): {len(regions)}")
        self.get_logger().info(f"[DEBUG] Hallway-like regions to merge: {len(hallway_regions)}")

        if len(regions) == 0:
            self.get_logger().error("No valid rooms detected. Map may be too open.")
            return

        # --------------------------
        # 3. Merge hallways into nearest room
        # --------------------------
        merged_regions = []

        # Convert regions to dicts for merging
        room_data = []
        for lid, xs, ys in regions:
            cx = xs.mean()
            cy = ys.mean()
            room_data.append({
                "id": lid,
                "xs": xs,
                "ys": ys,
                "cx": cx,
                "cy": cy,
                "bbox": (xs.min(), xs.max(), ys.min(), ys.max()),
            })

        for lid, xs, ys in hallway_regions:
            cx = xs.mean()
            cy = ys.mean()

            # find nearest real room
            dists = [(abs(r["cx"] - cx) + abs(r["cy"] - cy), r) for r in room_data]
            _, nearest = min(dists, key=lambda x: x[0])

            self.get_logger().info(
                f"[DEBUG] Hallway region {lid} merged → room {nearest['id']}"
            )

            # append hallway pixels to the room
            nearest["xs"] = np.concatenate((nearest["xs"], xs))
            nearest["ys"] = np.concatenate((nearest["ys"], ys))

        # final rooms list
        final_rooms = room_data

        # --------------------------
        # 4. Sort rooms left → right
        # --------------------------
        final_rooms.sort(key=lambda r: r["cx"])
        self.get_logger().info(f"[DEBUG] Final room count after merge: {len(final_rooms)}")

        # --------------------------
        # 5. Generate search points
        #     3×3 interior grid per room
        # --------------------------
        search_points = []

        for r in final_rooms:
            xs = r["xs"]
            ys = r["ys"]

            x_min, x_max, y_min, y_max = r["bbox"]
            width = x_max - x_min
            height = y_max - y_min

            self.get_logger().info(
                f"[DEBUG] Room {r['id']} final bbox=({x_min},{x_max},{y_min},{y_max}) generating samples…"
            )

            # interior sample grid
            grid_x = np.linspace(x_min + 0.25 * width, x_max - 0.25 * width, 3)
            grid_y = np.linspace(y_min + 0.25 * height, y_max - 0.25 * height, 3)

            for px in grid_x:
                for py in grid_y:
                    px0 = int(px)
                    py0 = int(py)

                    if occ[py0, px0] < 50:  # must actually be free
                        wx = px * self.map_res + self.map_origin[0]
                        wy = (H - py) * self.map_res + self.map_origin[1]  # careful with Y flip

                        search_points.append((wx, wy))
                        self.get_logger().info(
                            f"[DEBUG] Added search point world ({wx:.2f}, {wy:.2f})"
                        )

        # store into queue
        self.search_points = deque(search_points)
        self.get_logger().info(
            f"Generated {len(search_points)} deterministic search points across {len(final_rooms)} rooms (hallways merged)."
        )

    

    def assign_next_search_goal(self):
        # If all balls found, stop search
        if len(self.found) >= 3:
            self.stop_robot()
            self.get_logger().info("All balls found — stopping search.")
            return

        # If we have a manual RViz goal waiting, let it run
        # (we prefer manual goal if set)
        if self.goal is not None:
            return

        # Pop next search point
        while self.search_points:
            x, y = self.search_points.popleft()
            # Skip if point is too far or same as other found locations
            # set as new goal
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            self.goal = ps
            self.get_logger().info(f"New search goal: {x:.2f}, {y:.2f}")
            # attempt to plan
            ok = self.plan_path()
            if ok:
                return
            else:
                # failed to plan to this search point, try next
                self.goal = None
                continue

        self.get_logger().warn("Search points exhausted. No more points to try.")

    # ---------------- raycast ----------------
    def raycast_on_map(self, angle_cam):
        """
        Raycast on occupancy grid along robot_yaw + angle_cam.
        Returns distance in meters to first occupied cell; None if none hit.
        """
        if self.map_arr is None or self.pose is None:
            return None
        gx, gy = self.world_to_grid(self.pose.position.x, self.pose.position.y)
        yaw = self.quat_to_yaw(self.pose.orientation)
        ang = yaw + angle_cam

        # step in cells (integer ray march). use small step for better accuracy if needed
        max_steps = int(12.0 / self.map_res)  # up to 12 meters
        for i in range(1, max_steps):
            rx = gx + int(round(i * math.cos(ang)))
            ry = gy + int(round(i * math.sin(ang)))
            if rx < 0 or ry < 0 or ry >= self.map_height or rx >= self.map_width:
                return None
            if self.map_arr[ry, rx] >= OCCUPIED_THRESH:
                return i * self.map_res
        return None
    
    def print_ball_status(self):
        self.get_logger().info("==== BALL LOCALIZATION STATUS ====")
        for color, pos in self.detected_balls.items():
            if pos is None:
                self.get_logger().info(f"{color.upper()}: not detected yet")
            else:
                x, y = pos
                self.get_logger().info(f"{color.upper()}: ({x:.2f}, {y:.2f})")
                pt = Point(x, y, z=0.0)
                if color == "red":
                    self.red_pub.publish(pt)
                elif color == "green":
                    self.green_pub.publish(pt)
                elif color == "blue":
                    self.blue_pub.publish(pt)


def main(args=None):
    rclpy.init(args=args)
    node = Task3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
