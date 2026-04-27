import os
import sys
import numpy as np
from scipy.spatial import KDTree

os.environ['WEBOTS_HOME'] = '/usr/local/webots'
sys.path.append(os.path.join(os.environ['WEBOTS_HOME'], 'lib/controller/python'))

from controller import Supervisor

class AnytimeRRTStar:
    def __init__(self, start, goal, bounds, step_size=0.2, search_radius=0.8):
        self.start      = np.array(start)
        self.goal       = np.array(goal)
        self.bounds     = bounds
        self.step_size  = step_size
        self.search_radius = search_radius
        self.nodes      = [self.start]
        self.parents    = {0: None}
        self.costs      = {0: 0.0}

    def sample(self):
        if np.random.random() < 0.1:
            return self.goal
        return np.array([
            np.random.uniform(self.bounds[0], self.bounds[1]),
            np.random.uniform(self.bounds[2], self.bounds[3])
        ])

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

        indices  = tree.query_ball_point(new_node, self.search_radius)
        min_cost = self.costs[near_idx] + np.linalg.norm(new_node - self.nodes[near_idx])
        best_p   = near_idx

        for i in indices:
            cost = self.costs[i] + np.linalg.norm(new_node - self.nodes[i])
            if cost < min_cost:
                min_cost = cost
                best_p   = i

        new_idx = len(self.nodes)
        self.nodes.append(new_node)
        self.parents[new_idx] = best_p
        self.costs[new_idx]   = min_cost

        for i in indices:
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

    # ── Motors (TurtleBot3 Burger) ──────────────────────────────────────
    left_motor  = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')

    for m in [left_motor, right_motor]:
        m.setPosition(float('inf'))
        m.setVelocity(0.0)
    print("[INIT] Motors ready")

    # ── Lidar (LDS-01 built into TurtleBot3) ───────────────────────────
    lidar = robot.getDevice('LDS-01')
    if lidar:
        lidar.enable(timestep)
        lidar.enablePointCloud()
        print(f"[INIT] Lidar ready — {lidar.getNumberOfLayers()} layers, "
              f"{lidar.getHorizontalResolution()} points/scan")
    else:
        print("[ERROR] Lidar not found!")
        return

    # ── Supervisor odometry ─────────────────────────────────────────────
    robot_node = robot.getSelf()

    # ── Planner ─────────────────────────────────────────────────────────
    # Robot starts at (0, -4.3) on the floor, goal is across the room
    planner = AnytimeRRTStar(
        start  = [0.0,  -4.3],
        goal   = [0.0,   2.0],
        bounds = [-9, 9, -12, 3]
    )

    # ── Tuning ──────────────────────────────────────────────────────────
    MAX_SPEED          = 6.67   # TurtleBot3 Burger max rad/s
    WHEEL_RADIUS       = 0.033  # metres
    WHEEL_BASE         = 0.16   # metres between wheels
    K_LIN              = 1.0
    K_ANG              = 3.0
    GOAL_TOLERANCE     = 0.2
    OBSTACLE_THRESHOLD = 0.35   # metres — LDS range
    PLAN_EVERY_N       = 3
    step_count         = 0
    goal_reached       = False

    def set_speeds(v_lin, v_ang):
        """Convert linear/angular velocity to left/right wheel speeds."""
        left  = np.clip((v_lin - v_ang * WHEEL_BASE / 2) / WHEEL_RADIUS,
                        -MAX_SPEED, MAX_SPEED)
        right = np.clip((v_lin + v_ang * WHEEL_BASE / 2) / WHEEL_RADIUS,
                        -MAX_SPEED, MAX_SPEED)
        left_motor.setVelocity(left)
        right_motor.setVelocity(right)

    while robot.step(timestep) != -1:
        step_count += 1

        # ── 1. Expand planner ──────────────────────────────────────────
        if step_count % PLAN_EVERY_N == 0:
            planner.plan_step()

        # ── 2. Odometry ────────────────────────────────────────────────
        pos     = robot_node.getPosition()
        curr_p  = np.array([pos[0], pos[2]])     # X, Z (Y-up world)
        rot     = robot_node.getOrientation()
        heading = np.arctan2(rot[3], rot[0])      # atan2(R10, R00)

        # ── 3. Goal check ──────────────────────────────────────────────
        dist_to_goal = np.linalg.norm(np.array(planner.goal) - curr_p)
        if dist_to_goal < GOAL_TOLERANCE and not goal_reached:
            print(f"[GOAL] Reached at step {step_count}!")
            goal_reached = True

        if goal_reached:
            set_speeds(0.0, 0.0)
            continue

        # ── 4. Lidar obstacle check ────────────────────────────────────
        ranges = lidar.getRangeImage()   # flat list, 360 values for LDS-01
        n      = len(ranges)

        # Front arc: indices around 0/360 (robot faces +X in lidar frame)
        front_arc  = list(range(0, 30)) + list(range(n - 30, n))
        left_arc   = list(range(30, 90))
        right_arc  = list(range(n - 90, n - 30))

        def min_valid(indices):
            vals = [ranges[i] for i in indices
                    if 0 < ranges[i] < float('inf')]
            return min(vals) if vals else float('inf')

        min_front = min_valid(front_arc)
        min_left  = min_valid(left_arc)
        min_right = min_valid(right_arc)

        if step_count % 30 == 0:
            print(f"[SCAN] front={min_front:.2f} left={min_left:.2f} "
                  f"right={min_right:.2f} dist_goal={dist_to_goal:.2f}")

        # ── 5. Control ─────────────────────────────────────────────────
        if min_front < OBSTACLE_THRESHOLD:
            # Avoid — turn away from closer side
            if min_left < min_right:
                set_speeds(0.05, -1.5)   # turn right
            else:
                set_speeds(0.05,  1.5)   # turn left
        else:
            # Follow RRT* path
            path = planner.get_path()
            if len(path) > 1:
                target = path[min(1, len(path) - 1)]
                desired = np.arctan2(target[1] - curr_p[1],
                                     target[0] - curr_p[0])
                error   = (desired - heading + np.pi) % (2 * np.pi) - np.pi

                v_lin = K_LIN * np.clip(dist_to_goal, 0, 0.5)
                v_ang = K_ANG * error
                set_speeds(v_lin, v_ang)
            else:
                set_speeds(0.1, 0.0)   # nudge forward while tree builds


if __name__ == "__main__":
    run_robot()