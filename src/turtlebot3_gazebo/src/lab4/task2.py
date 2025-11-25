#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

import numpy as np
import math
from collections import deque
import heapq

# tuning
INFLATE_RADIUS_CELLS = 4        # smaller for debugging; increase if safe
OCCUPIED_THRESH = 65            # map values >= this are occupied
LOOKAHEAD_M = 0.30
FRONT_OBSTACLE_THRESHOLD = 0.7  # m -> treat as dynamic blocking
FRONT_ANGLE_DEG = 15             # degrees on each side for front sector
MAX_LINEAR_SPEED = 0.20
MAX_ANGULAR_SPEED = 0.8
GOAL_NEAR_DIST_M = 0.25

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
            if expansions % 50000 == 0:
                # periodic warning
                # note: Node logger isn't available here; we'll return a special signal by raising
                pass
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


class Task2(Node):
    def __init__(self):
        super().__init__('task2_algorithm')

        # QoS
        qos = QoSProfile(depth=10)

        # Map must be subscribed with TRANSIENT_LOCAL durability to receive the latched map
        from rclpy.qos import ReliabilityPolicy, DurabilityPolicy
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Subscriptions
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, qos)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, qos_profile=map_qos)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        self.path_pub = self.create_publisher(Path, 'global_plan', qos)

        # State
        self.map_msg = None
        self.map_arr = None            # numpy array [H, W]
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

        # timer loop
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info("Task2 node started.")

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
        except Exception as e:
            self.get_logger().error(f"Exception in map_cb: {e}")

    def amcl_cb(self, msg: PoseWithCovarianceStamped):
        # store Pose (geometry_msgs/Pose)
        self.pose = msg.pose.pose

    def goal_cb(self, msg: PoseStamped):
        self.get_logger().info("New goal received from RViz.")
        self.goal = msg
        self.get_logger().info(f"Goal (world): {msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}")
        # attempt an immediate plan, plan_path internally checks readiness
        ok = self.plan_path()
        self.get_logger().info(f"Plan result: {ok}")

    def scan_cb(self, msg: LaserScan):
        self.lidar = msg
        self.ranges = np.array(msg.ranges, dtype=float)
        self.ranges = np.where(np.isfinite(self.ranges), self.ranges, np.inf)

    # ---------------- main loop ----------------
    def timer_cb(self):
        # keep AMCL happy: if paused, continue publishing zero cmd vel externally
        if self.pose is None:
            # nothing to do until we have a pose
            return

        if self.goal is None:
            return

        # If we have no map yet, wait (map_cb will set map_loaded)
        if not self.map_loaded:
            return

        # If we don't yet have a planned path, attempt to plan
        if self.global_path is None:
            ok = self.plan_path()
            if not ok:
                self.get_logger().warn("Initial planning failed; will retry in timer loop.")
            return

        # dynamic obstacle detection
        if self.is_dynamic_blocking_next_waypoint():
            self.get_logger().warn("Dynamic obstacle blocking path — replanning")
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
        """Convert world (x,y) to grid indices (gx,gy) where gx=col, gy=row (indexing into self.map_arr)
        This matches the conversion used in auto_navigator_fast.py"""
        res = self.map_res
        ox, oy = self.map_origin
        # column (j)
        j = int((wx - ox) / res)
        # row (i) with flip because map array row 0 corresponds to top of image
        i = self.map_height - int((wy - oy) / res) - 1
        #i = self.mp.map.size[1] - int((wy - oy) / res)
        # clamp
        #j = max(0, min(self.map_width - 1, j))
        #i = max(0, min(self.map_height - 1, i))
        # return (gx, gy) where gx=col, gy=row
        return j, i

    def grid_to_world(self, gx, gy):
        ox, oy = self.map_origin
        res = self.map_res
        x = gx * res + ox + res/2.0
        y = (self.map_height - gy - 1) * res + oy + res/2.0

        #x = gy * res + ox 
        #y = (self.map_height-gx) * res + oy 
        return x, y

    def find_nearest_free(self, g):
        """BFS to nearest free cell on self.inf_map. Input and output are (gx,gy) where gx=col, gy=row."""
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

        if local_replan and self.is_dynamic_blocking_next_waypoint():
            ox, oy = self.find_closest_front_point()
            if ox is not None:
                cgx, cgy = self.world_to_grid(ox, oy)
                r = max(1, int(0.2 / self.map_res))
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        nx, ny = cgx + dx, cgy + dy
                        if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                            working_map[ny, nx] = 1

        # run A*
        path = self.astar_grid(start_g, goal_g, map_bin=working_map)

        if path is None:
            self.get_logger().warn("A* failed to find a path.")
            self.global_path = None
            self.path_world = []
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

    # ---------------- dynamic obstacle helpers ----------------
    def is_dynamic_blocking_next_waypoint(self):
        if self.lidar is None or not self.path_world:
            return False

        idx = max(0, min(self.last_path_idx, len(self.path_world)-1))
        next_idx = min(idx + 2, len(self.path_world)-1)
        wx, wy = self.path_world[next_idx]

        rx, ry = self.pose.position.x, self.pose.position.y
        vx = wx - rx
        vy = wy - ry
        dist_to_way = math.hypot(vx, vy)
        if dist_to_way < 0.1:
            return False

        desired_yaw = math.atan2(vy, vx)
        if self.lidar is None:
            return False
        angle_min = self.lidar.angle_min
        angle_inc = self.lidar.angle_increment
        N = len(self.ranges)

        w = math.radians(FRONT_ANGLE_DEG)
        idx0 = int((0.0 - w - angle_min) / angle_inc)
        idx1 = int((0.0 + w - angle_min) / angle_inc)
        idx0 = max(0, min(N-1, idx0))
        idx1 = max(0, min(N-1, idx1))
        sector = self.ranges[idx0:idx1+1]
        if sector.size == 0:
            return False
        min_front = float(np.min(sector))
        if min_front < FRONT_OBSTACLE_THRESHOLD:
            rel_idx = int(np.argmin(sector))
            overall_idx = idx0 + rel_idx
            angle = angle_min + overall_idx * angle_inc
            robot_yaw = self.quat_to_yaw(self.pose.orientation)
            bearing_to_way = desired_yaw - robot_yaw
            bearing_to_way = (bearing_to_way + math.pi) % (2*math.pi) - math.pi
            if abs(angle - bearing_to_way) <= math.radians(25):
                return True
        return False

    def find_closest_front_point(self):
        if self.lidar is None:
            return (None, None)
        angle_min = self.lidar.angle_min
        angle_inc = self.lidar.angle_increment
        N = len(self.ranges)
        w = math.radians(FRONT_ANGLE_DEG)
        idx0 = int((0.0 - w - angle_min) / angle_inc)
        idx1 = int((0.0 + w - angle_min) / angle_inc)
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
        angle = angle_min + overall_idx * angle_inc
        rx = r * math.cos(angle)
        ry = r * math.sin(angle)
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
            self.global_path = None
            self.path_world = []
            self.stop_robot()

    def get_path_idx(self, path_world, last_idx=0):
        if not path_world:
            return 0
        vx = self.pose.position.x
        vy = self.pose.position.y
        lookahead = LOOKAHEAD_M
        n = len(path_world)
        for i in range(last_idx, n):
            px, py = path_world[i]
            dist = math.hypot(px - vx, py - vy)
            if dist > lookahead:
                return i
        return n-1

    def quat_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def path_follower_from_xy(self, goal_x, goal_y):
        vx = self.pose.position.x
        vy = self.pose.position.y
        gx = goal_x
        gy = goal_y
        yaw = self.quat_to_yaw(self.pose.orientation)
        desired_yaw = math.atan2(gy - vy, gx - vx)
        heading_error = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        dist = math.hypot(gx - vx, gy - vy)
        Kp_ang = 1.2
        Kp_lin = 0.8
        heading = Kp_ang * heading_error
        heading = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, heading))
        cos_term = math.cos(heading_error)
        if cos_term < 0:
            cos_term = 0.0
        speed = max(0.03, min(MAX_LINEAR_SPEED, Kp_lin * dist * cos_term))
        if hasattr(self, 'ranges') and self.ranges is not None:
            N = len(self.ranges)
            i0 = max(0, int(N*0.45)); i1 = min(N-1, int(N*0.55))
            front_med = float(np.median(self.ranges[i0:i1+1]))
            if front_med < 0.35:
                speed = min(speed, 0.05)
        return speed, heading, dist

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


def main(args=None):
    rclpy.init(args=args)
    node = Task2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
