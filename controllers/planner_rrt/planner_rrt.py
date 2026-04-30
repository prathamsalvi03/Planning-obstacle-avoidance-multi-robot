import os
import sys
import numpy as np
from scipy.spatial import KDTree

os.environ['WEBOTS_HOME'] = '/usr/local/webots'
sys.path.append(os.path.join(os.environ['WEBOTS_HOME'], 'lib/controller/python'))

from controller import Supervisor


# ═══════════════════════════════════════════════════════════════════════════════
# RRT* PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

class AnytimeRRTStar:
    def __init__(self, start, goal, bounds, step_size=0.4, search_radius=1.2):
        self.start         = np.array(start)
        self.goal          = np.array(goal)
        self.bounds        = bounds
        self.step_size     = step_size
        self.search_radius = search_radius
        self.nodes         = [self.start]
        self.parents       = {0: None}
        self.costs         = {0: 0.0}
        self.obstacles     = []

    def add_obstacle(self, x, y, radius=0.4):
        self.obstacles.append((x, y, radius))

    def is_collision(self, point):
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(point - np.array([ox, oy])) < r:
                return True
        return False

    def is_edge_collision(self, p1, p2):
        steps = max(int(np.linalg.norm(p2 - p1) / 0.1), 1)
        for i in range(steps + 1):
            t     = i / steps
            point = p1 + t * (p2 - p1)
            if self.is_collision(point):
                return True
        return False

    def sample(self):
        if np.random.random() < 0.15:
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
            return False
        new_node = (self.nodes[near_idx] + (diff / dist) * self.step_size
                    if dist > self.step_size else rnd)

        if self.is_collision(new_node):
            return False
        if self.is_edge_collision(self.nodes[near_idx], new_node):
            return False

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
        return True

    def get_path(self):
        tree = KDTree(self.nodes)
        _, goal_idx = tree.query(self.goal)
        path, curr = [], goal_idx
        while curr is not None:
            path.append(self.nodes[curr])
            curr = self.parents[curr]
        return path[::-1]

    def goal_reachable(self):
        tree = KDTree(self.nodes)
        dist, _ = tree.query(self.goal)
        return dist < self.step_size * 2


# ═══════════════════════════════════════════════════════════════════════════════
# ORCA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_orca_velocity(my_pos, my_vel, other_pos, other_vel,
                          combined_radius=0.5, tau=3.0, max_speed=0.4):
    """
    Returns a safe velocity for this robot given one neighbor.
    Based on Reciprocal Velocity Obstacles (van den Berg et al.)
    """
    rel_pos = other_pos - my_pos
    rel_vel = my_vel - other_vel
    dist    = np.linalg.norm(rel_pos)

    # Already overlapping — emergency push apart
    if dist < combined_radius:
        push = -rel_pos / (dist + 1e-9) * max_speed
        return push

    # ── Build velocity obstacle ─────────────────────────────────────────
    # Cone apex at origin, pointing toward other robot / tau
    apex     = rel_pos / tau
    leg_len  = np.sqrt(max(dist**2 - combined_radius**2, 1e-6))
    r_scaled = combined_radius / tau

    # Normal to nearest VO boundary
    w     = rel_vel - apex
    w_len = np.linalg.norm(w)

    if w_len < 1e-9:
        return my_vel   # no conflict

    w_hat = w / w_len

    # Is rel_vel inside the VO cone?
    dot = np.dot(rel_vel - apex, rel_pos / (dist + 1e-9))
    if dot < 0:
        return my_vel   # behind cone — safe

    # Project onto nearest boundary
    # Use circular cross-section at distance dist/tau
    u = (r_scaled - np.dot(w, rel_pos / (dist + 1e-9))) * w_hat

    # Each robot takes half responsibility
    safe_vel = my_vel + 0.5 * u

    # Clamp to max speed
    speed = np.linalg.norm(safe_vel)
    if speed > max_speed:
        safe_vel = safe_vel / speed * max_speed

    return safe_vel


# ═══════════════════════════════════════════════════════════════════════════════
# LIDAR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def scan_and_map(robot, robot_node, lidar, planner, duration_ms, timestep,
                 plan_per_step=10):
    """
    Phase 1: Robot stands still, rotates 360° scanning, builds obstacle map
    and expands RRT* tree. Returns when duration_ms elapsed.
    """
    steps    = int(duration_ms / timestep)
    n_ranges = lidar.getHorizontalResolution()

    for s in range(steps):
        if robot.step(timestep) == -1:
            return

        pos     = robot_node.getPosition()
        curr_p  = np.array([pos[0], pos[1]])
        rot     = robot_node.getOrientation()
        heading = np.arctan2(rot[3], rot[0])
        ranges  = lidar.getRangeImage()
        n       = len(ranges)

        # Add ALL valid lidar hits to obstacle map
        for i, r in enumerate(ranges):
            if 0.12 < r < 5.0:          # wider range during scan phase
                angle = heading + (2 * np.pi * i / n)
                ox = curr_p[0] + r * np.cos(angle)
                oy = curr_p[1] + r * np.sin(angle)
                too_close = any(
                    np.linalg.norm(np.array([ox, oy]) - np.array([ex, ey])) < 0.2
                    for ex, ey, _ in planner.obstacles
                )
                if not too_close:
                    planner.add_obstacle(ox, oy, radius=0.4)

        # Expand tree aggressively during scan
        for _ in range(plan_per_step):
            planner.plan_step()

        if s % 20 == 0:
            print(f"  [SCAN {s}/{steps}] obs={len(planner.obstacles)} "
                  f"nodes={len(planner.nodes)} "
                  f"reachable={planner.goal_reachable()}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

def run_robot():
    robot    = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    name     = robot.getName()
    print(f"\n{'='*50}")
    print(f"[INIT] Robot={name}  timestep={timestep}ms")
    print(f"{'='*50}")

    # ── Motors ──────────────────────────────────────────────────────────
    left_motor  = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    for m in [left_motor, right_motor]:
        m.setPosition(float('inf'))
        m.setVelocity(0.0)
    print("[INIT] Motors ready")

    # ── Lidar ───────────────────────────────────────────────────────────
    lidar = robot.getDevice('LDS-01')
    if not lidar:
        print("[ERROR] Lidar not found!")
        return
    lidar.enable(timestep)
    lidar.enablePointCloud()
    print(f"[INIT] Lidar: {lidar.getHorizontalResolution()} pts/scan  "
          f"range={lidar.getMaxRange():.1f}m")

    # ── Supervisor ──────────────────────────────────────────────────────
    robot_node = robot.getSelf()
    other_name = "Robot_2" if name == "Robot_1" else "Robot_1"
    other_node = robot.getFromDef(other_name)
    print(f"[INIT] Other robot ({other_name}): "
          f"{'FOUND' if other_node else 'NOT FOUND — ORCA disabled'}")

    # ── Get real start position ──────────────────────────────────────────
    robot.step(timestep)
    pos          = robot_node.getPosition()
    actual_start = [pos[0], pos[1]]
    print(f"[INIT] Start: ({actual_start[0]:.3f}, {actual_start[1]:.3f})")

    # ── Goal assignment ──────────────────────────────────────────────────
    if name == "Robot_1":
        goal = [-7.0, -10.5]   # bottom-left
    else:
        goal = [7.0, -0.5]     # top-right

    print(f"[INIT] Goal:  ({goal[0]:.1f}, {goal[1]:.1f})")

    # ── Planner ─────────────────────────────────────────────────────────
    planner = AnytimeRRTStar(
        start         = actual_start,
        goal          = goal,
        bounds        = [-9, 9, -12, 3],
        step_size     = 0.4,
        search_radius = 1.2
    )

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1 — SCAN (robot stationary, lidar maps everything, RRT* builds)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[PHASE 1] SCANNING — {name} stationary for 5 seconds...")
    scan_and_map(robot, robot_node, lidar, planner,
                 duration_ms=5000, timestep=timestep, plan_per_step=15)

    print(f"[PHASE 1] DONE — obs={len(planner.obstacles)} "
          f"nodes={len(planner.nodes)} "
          f"reachable={planner.goal_reachable()}")

    # If tree didn't reach goal, keep planning for 3 more seconds
    if not planner.goal_reachable():
        print("[PHASE 1] Goal not reachable yet — extending planning...")
        extra_steps = int(3000 / timestep)
        for _ in range(extra_steps):
            if robot.step(timestep) == -1:
                return
            for _ in range(20):
                planner.plan_step()
        print(f"[PHASE 1] Extended — nodes={len(planner.nodes)} "
              f"reachable={planner.goal_reachable()}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2 — MOVE with ORCA + RRT* path following
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[PHASE 2] MOVING — {name} following RRT* path with ORCA")

    MAX_SPEED          = 6.67
    WHEEL_RADIUS       = 0.033
    WHEEL_BASE         = 0.16
    K_ANG              = 2.0
    GOAL_TOLERANCE     = 0.3
    OBSTACLE_THRESHOLD = 0.35
    PLAN_EVERY_N       = 5
    ORCA_RADIUS        = 0.5     # combined collision radius
    ORCA_TAU           = 3.0     # time horizon
    ORCA_MAX_SPEED     = 0.4     # m/s
    WAYPOINT_TOLERANCE = 0.4
    step_count         = 0
    goal_reached       = False
    current_waypoint   = None
    prev_pos           = np.array(actual_start)
    other_prev_pos     = None

    def set_speeds(v_lin, v_ang):
        left  = np.clip((v_lin - v_ang * WHEEL_BASE / 2) / WHEEL_RADIUS,
                        -MAX_SPEED, MAX_SPEED)
        right = np.clip((v_lin + v_ang * WHEEL_BASE / 2) / WHEEL_RADIUS,
                        -MAX_SPEED, MAX_SPEED)
        left_motor.setVelocity(left)
        right_motor.setVelocity(right)

    while robot.step(timestep) != -1:
        step_count += 1
        dt = timestep / 1000.0

        # ── 1. Keep expanding planner ───────────────────────────────────
        if step_count % PLAN_EVERY_N == 0:
            planner.plan_step()

        # ── 2. Odometry ─────────────────────────────────────────────────
        pos     = robot_node.getPosition()
        curr_p  = np.array([pos[0], pos[1]])
        rot     = robot_node.getOrientation()
        heading = np.arctan2(rot[3], rot[0])
        own_vel = (curr_p - prev_pos) / dt
        prev_pos = curr_p.copy()

        # ── 3. Goal check ───────────────────────────────────────────────
        dist_to_goal = np.linalg.norm(np.array(planner.goal) - curr_p)
        if dist_to_goal < GOAL_TOLERANCE and not goal_reached:
            print(f"\n[GOAL] {name} reached goal at step {step_count}! "
                  f"dist={dist_to_goal:.3f}m")
            goal_reached = True
        if goal_reached:
            set_speeds(0.0, 0.0)
            continue

        # ── 4. Lidar — keep updating map while moving ───────────────────
        ranges = lidar.getRangeImage()
        n      = len(ranges)

        if step_count % 8 == 0:
            for i, r in enumerate(ranges):
                if 0.12 < r < 4.0:
                    angle = heading + (2 * np.pi * i / n)
                    ox = curr_p[0] + r * np.cos(angle)
                    oy = curr_p[1] + r * np.sin(angle)
                    too_close = any(
                        np.linalg.norm(np.array([ox, oy]) - np.array([ex, ey])) < 0.2
                        for ex, ey, _ in planner.obstacles
                    )
                    if not too_close:
                        planner.add_obstacle(ox, oy, radius=0.4)

        # ── 5. Lidar arcs for reactive avoidance ───────────────────────
        front_arc = list(range(0, 30)) + list(range(n - 30, n))
        left_arc  = list(range(30, 90))
        right_arc = list(range(n - 90, n - 30))

        def min_valid(idxs):
            vals = [ranges[i] for i in idxs if 0 < ranges[i] < float('inf')]
            return min(vals) if vals else float('inf')

        min_front = min_valid(front_arc)
        min_left  = min_valid(left_arc)
        min_right = min_valid(right_arc)

        # ── 6. RRT* waypoint ────────────────────────────────────────────
        path = planner.get_path()
        if len(path) > 1:
            lookahead = min(4, len(path) - 1)
            candidate = np.array(path[lookahead])
            if current_waypoint is None:
                current_waypoint = candidate
            elif np.linalg.norm(current_waypoint - curr_p) < WAYPOINT_TOLERANCE:
                current_waypoint = candidate
                print(f"[WP] {name} -> ({current_waypoint[0]:.2f},"
                      f"{current_waypoint[1]:.2f}) dist_goal={dist_to_goal:.2f}")

            to_wp   = current_waypoint - curr_p
            wp_dist = np.linalg.norm(to_wp)
            desired_vel = (to_wp / (wp_dist + 1e-9)) * ORCA_MAX_SPEED
        else:
            desired_vel = np.array([0.0, 0.0])

        # ── 7. ORCA ─────────────────────────────────────────────────────
        safe_vel = desired_vel.copy()
        if other_node:
            op3d = other_node.getPosition()
            other_p = np.array([op3d[0], op3d[1]])
            dist_to_other = np.linalg.norm(other_p - curr_p)

            # Estimate other robot velocity
            if other_prev_pos is not None:
                other_vel = (other_p - other_prev_pos) / dt
            else:
                other_vel = np.array([0.0, 0.0])
            other_prev_pos = other_p.copy()

            if dist_to_other < 4.0:   # apply ORCA within 4m
                safe_vel = compute_orca_velocity(
                    my_pos        = curr_p,
                    my_vel        = desired_vel,
                    other_pos     = other_p,
                    other_vel     = other_vel,
                    combined_radius = ORCA_RADIUS,
                    tau           = ORCA_TAU,
                    max_speed     = ORCA_MAX_SPEED
                )
                if step_count % 60 == 0 and dist_to_other < 4.0:
                    print(f"[ORCA] {name} d={dist_to_other:.2f}m "
                          f"des=({desired_vel[0]:.2f},{desired_vel[1]:.2f}) "
                          f"safe=({safe_vel[0]:.2f},{safe_vel[1]:.2f})")

        # ── 8. Execute velocity ─────────────────────────────────────────
        if min_front < OBSTACLE_THRESHOLD:
            # Hard reactive override for static obstacles
            current_waypoint = None
            if min_left < min_right:
                set_speeds(0.0, -2.0)
            else:
                set_speeds(0.0,  2.0)

        else:
            safe_speed = np.linalg.norm(safe_vel)
            if safe_speed < 0.01:
                # ORCA says stop — nudge sideways
                set_speeds(0.0, 1.5)
            else:
                desired_heading = np.arctan2(safe_vel[1], safe_vel[0])
                error = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi
                if abs(error) > 0.4:
                    set_speeds(0.0, K_ANG * error)   # turn first
                else:
                    set_speeds(safe_speed, K_ANG * error)

        # ── 9. Debug every 60 steps ─────────────────────────────────────
        if step_count % 60 == 0:
            wp_str = (f"({current_waypoint[0]:.2f},{current_waypoint[1]:.2f})"
                      if current_waypoint is not None else "None")
            print(f"[{name}|{step_count}] "
                  f"pos=({curr_p[0]:.2f},{curr_p[1]:.2f}) "
                  f"goal_dist={dist_to_goal:.2f} "
                  f"front={min_front:.2f} "
                  f"nodes={len(planner.nodes)} "
                  f"obs={len(planner.obstacles)} "
                  f"path={len(path)} "
                  f"wp={wp_str}")


if __name__ == "__main__":
    run_robot()