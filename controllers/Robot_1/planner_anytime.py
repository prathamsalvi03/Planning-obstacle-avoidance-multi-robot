import os
import sys
import numpy as np
from scipy.spatial import KDTree

os.environ['WEBOTS_HOME'] = '/usr/local/webots'
sys.path.append(os.path.join(os.environ['WEBOTS_HOME'], 'lib/controller/python'))

from controller import Supervisor

class AnytimeRRTStar:
    def __init__(self, start, goal, bounds, step_size=0.3, search_radius=1.0):
        self.start         = np.array(start)
        self.goal          = np.array(goal)
        self.bounds        = bounds
        self.step_size     = step_size
        self.search_radius = search_radius
        self.nodes         = [self.start]
        self.parents       = {0: None}
        self.costs         = {0: 0.0}
        self.obstacles     = []  # list of (x, y, radius) circles

    def add_obstacle(self, x, y, radius=0.4):
        """Call this from the main loop with lidar hit positions."""
        self.obstacles.append((x, y, radius))

    def is_collision(self, point):
        """Check if a point is inside any known obstacle."""
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(point - np.array([ox, oy])) < r:
                return True
        return False

    def is_edge_collision(self, p1, p2):
        """Check if the edge between two nodes passes through an obstacle."""
        steps = int(np.linalg.norm(p2 - p1) / 0.1)
        for i in range(steps + 1):
            t = i / max(steps, 1)
            point = p1 + t * (p2 - p1)
            if self.is_collision(point):
                return True
        return False

    def plan_step(self):
        rnd  = self.sample()
        tree = KDTree(self.nodes)
        _, near_idx = tree.query(rnd)
        diff = rnd - self.nodes[near_idx]
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return
        new_node = (self.nodes[near_idx] + (diff / dist) * self.step_size
                    if dist > self.step_size else rnd)

        # ── COLLISION CHECK — reject node if in obstacle ────────────────
        if self.is_collision(new_node):
            return
        if self.is_edge_collision(self.nodes[near_idx], new_node):
            return

        indices  = tree.query_ball_point(new_node, self.search_radius)
        min_cost = self.costs[near_idx] + np.linalg.norm(new_node - self.nodes[near_idx])
        best_p   = near_idx

        for i in indices:
            if self.is_edge_collision(self.nodes[i], new_node):
                continue
            cost = self.costs[i] + np.linalg.norm(new_node - self.nodes[i])
            if cost < min_cost:
                min_cost = cost
                best_p   = i

        new_idx = len(self.nodes)
        self.nodes.append(new_node)
        self.parents[new_idx] = best_p
        self.costs[new_idx]   = min_cost

        for i in indices:
            if self.is_edge_collision(self.nodes[i], new_node):
                continue
            new_c = self.costs[new_idx] + np.linalg.norm(self.nodes[i] - new_node)
            if new_c < self.costs[i]:
                self.parents[i] = new_idx
                self.costs[i]   = new_c

    def get_path(self):
        tree = KDTree(self.nodes)
        _, goal_idx = tree.query(self.goal)
        path, curr = [], goal_idx
        while curr is not None:
            path.append(self.nodes[curr])
            curr = self.parents[curr]
        return path[::-1]


def run_robot():
    robot    = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    print(f"[INIT] timestep={timestep}, name={robot.getName()}")

    # ── Motors ──────────────────────────────────────────────────────────
    left_motor  = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    for m in [left_motor, right_motor]:
        m.setPosition(float('inf'))
        m.setVelocity(0.0)
    print("[INIT] Motors ready")

    # ── Lidar ───────────────────────────────────────────────────────────
    lidar = robot.getDevice('LDS-01')
    if lidar:
        lidar.enable(timestep)
        lidar.enablePointCloud()
        print(f"[INIT] Lidar ready — {lidar.getHorizontalResolution()} points/scan")
    else:
        print("[ERROR] Lidar not found!")
        return

    # ── Supervisor — get node BEFORE first step ──────────────────────────
    robot_node = robot.getSelf()

    # ── Step once to get real spawn position ────────────────────────────
    robot.step(timestep)
    pos          = robot_node.getPosition()
    actual_start = [pos[0], pos[1]]   # TurtleBot uses X, Y
    print(f"[INIT] Actual start: {actual_start}")

    # ── Planner ─────────────────────────────────────────────────────────
    planner = AnytimeRRTStar(
        start  = actual_start,
        goal   = [-3.0, 0.0],         # open area, adjust if needed
        bounds = [-9, 9, -12, 3]
    )

    # ── Constants ───────────────────────────────────────────────────────
    MAX_SPEED          = 6.67
    WHEEL_RADIUS       = 0.033
    WHEEL_BASE         = 0.16
    K_LIN              = 0.3
    K_ANG              = 2.0
    GOAL_TOLERANCE     = 0.2
    OBSTACLE_THRESHOLD = 0.4
    PLAN_EVERY_N       = 3
    step_count         = 0
    goal_reached       = False

    def set_speeds(v_lin, v_ang):
        left  = np.clip((v_lin - v_ang * WHEEL_BASE / 2) / WHEEL_RADIUS,
                        -MAX_SPEED, MAX_SPEED)
        right = np.clip((v_lin + v_ang * WHEEL_BASE / 2) / WHEEL_RADIUS,
                        -MAX_SPEED, MAX_SPEED)
        left_motor.setVelocity(left)
        right_motor.setVelocity(right)

    current_waypoint = None
    WAYPOINT_TOLERANCE = 0.3  # reach this close before next waypoint

    while robot.step(timestep) != -1:
        step_count += 1

        if step_count % PLAN_EVERY_N == 0:
            planner.plan_step()

        pos     = robot_node.getPosition()
        curr_p  = np.array([pos[0], pos[1]])
        rot     = robot_node.getOrientation()
        heading = np.arctan2(rot[3], rot[0])

        dist_to_goal = np.linalg.norm(np.array(planner.goal) - curr_p)
        if dist_to_goal < GOAL_TOLERANCE and not goal_reached:
            print(f"[GOAL] Reached at step {step_count}!")
            goal_reached = True
        if goal_reached:
            set_speeds(0.0, 0.0)
            continue

        ranges    = lidar.getRangeImage()
        n         = len(ranges)
        front_arc = list(range(0, 30)) + list(range(n - 30, n))
        left_arc  = list(range(30, 90))
        right_arc = list(range(n - 90, n - 30))

        def min_valid(idxs):
            vals = [ranges[i] for i in idxs
                    if 0 < ranges[i] < float('inf')]
            return min(vals) if vals else float('inf')

        min_front = min_valid(front_arc)
        min_left  = min_valid(left_arc)
        min_right = min_valid(right_arc)

        if min_front < OBSTACLE_THRESHOLD:
            current_waypoint = None  # reset waypoint on obstacle
            if min_left < min_right:
                set_speeds(0.0, -2.0)
            else:
                set_speeds(0.0,  2.0)
        else:
            path = planner.get_path()
            if len(path) > 1:

                # ── Waypoint locking ──────────────────────────────────
                # Pick a waypoint further ahead to avoid jitter
                lookahead = min(3, len(path) - 1)
                candidate = path[lookahead]

                # Only switch waypoint if current one reached or none set
                if current_waypoint is None:
                    current_waypoint = candidate
                elif np.linalg.norm(current_waypoint - curr_p) < WAYPOINT_TOLERANCE:
                    current_waypoint = candidate
                    print(f"[WP] Next waypoint: ({current_waypoint[0]:.2f},{current_waypoint[1]:.2f})")

                target  = current_waypoint
                desired = np.arctan2(target[1] - curr_p[1],
                                     target[0] - curr_p[0])
                error   = (desired - heading + np.pi) % (2 * np.pi) - np.pi

                if abs(error) > 0.4:
                    set_speeds(0.0, K_ANG * error)
                else:
                    set_speeds(K_LIN, K_ANG * error)
            else:
                set_speeds(0.1, 0.0)

        if step_count % 30 == 0:
            wp_str = f"({current_waypoint[0]:.2f},{current_waypoint[1]:.2f})" \
                     if current_waypoint is not None else "None"
            print(f"[SCAN] front={min_front:.2f} left={min_left:.2f} "
                  f"right={min_right:.2f} dist_goal={dist_to_goal:.2f} "
                  f"pos=({curr_p[0]:.2f},{curr_p[1]:.2f}) "
                  f"hdg={np.degrees(heading):.1f}° wp={wp_str}")


if __name__ == "__main__":
    run_robot()