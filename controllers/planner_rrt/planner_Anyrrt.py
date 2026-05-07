import os
import sys
import numpy as np
from scipy.spatial import KDTree

os.environ['WEBOTS_HOME'] = '/usr/local/webots'
sys.path.append(os.path.join(os.environ['WEBOTS_HOME'], 'lib/controller/python'))

from controller import Supervisor
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD KNOWLEDGE — extracted from Planning.wbt
# ═══════════════════════════════════════════════════════════════════════════════
# Walls:   North Y=3.82, South Y=-12.5, West X=-10, East X=10
# Obstacles (X, Y, radius):
# Replace STATIC_OBSTACLES with smaller radii
STATIC_OBSTACLES = [
    # Oil barrels (radius 0.3 not 0.5)
    (9.34,  2.90, 0.3),
    (7.30, -1.94, 0.3),
    (1.60, -1.28, 0.3),
    # Wooden boxes (radius 0.3)
    (-4.83, -8.30, 0.3),
    (-4.69, -3.82, 0.3),
    ( 5.37, -4.64, 0.3),
    ( 1.55, -6.95, 0.3),
    (-0.50,-11.77, 0.4),
    (-1.98, -0.01, 0.3),
    ( 0.92,-11.92, 0.4),
    (-3.05,-12.10, 0.4),
    ( 5.98,-11.81, 0.3),
    # Pipes cluster
    (-7.57,  3.00, 0.6),
    (-5.71,  3.00, 0.4),
    (-4.86,  3.00, 0.3),
    (-6.62,  3.00, 0.4),
]

# Wall lines seeded as obstacle points
def seed_walls():
    pts = []
    for x in np.arange(-9.5, 9.5, 0.5):
        pts.append((x,  3.6, 0.3))   # north
        pts.append((x, -12.3, 0.3))  # south
    for y in np.arange(-12.0, 3.5, 0.5):
        pts.append((-9.6, y, 0.3))   # west
        pts.append(( 9.6, y, 0.3))   # east
    return pts

# ═══════════════════════════════════════════════════════════════════════════════
# RRT* PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

class AnytimeRRTStar:
    def __init__(self, start, goal, bounds, step_size=0.8, search_radius=1.5):
        self.start         = np.array(start)
        self.goal          = np.array(goal)
        self.bounds        = bounds
        self.step_size     = step_size
        self.search_radius = search_radius
        self.nodes         = [self.start]
        self.parents       = {0: None}
        self.costs         = {0: 0.0}
        self.obstacles     = []
        self._obs_tree     = None
        self._obs_pts      = None

    def add_obstacle(self, x, y, radius=0.4):
        self.obstacles.append((x, y, radius))
        self._obs_tree = None   # invalidate KDTree cache

    def _rebuild_obs_tree(self):
        if self.obstacles:
            self._obs_pts  = np.array([[o[0], o[1]] for o in self.obstacles])
            self._obs_tree = KDTree(self._obs_pts)

    def is_collision(self, point):
        if not self.obstacles:
            return False
        if self._obs_tree is None:
            self._rebuild_obs_tree()
        max_r = 0.8   # max obstacle radius — query within this
        idxs  = self._obs_tree.query_ball_point(point, max_r)
        for i in idxs:
            ox, oy, r = self.obstacles[i]
            if np.linalg.norm(point - np.array([ox, oy])) < r:
                return True
        return False

    def is_edge_collision(self, p1, p2):
        steps = max(int(np.linalg.norm(p2 - p1) / 0.15), 1)
        for i in range(steps + 1):
            t     = i / steps
            point = p1 + t * (p2 - p1)
            if self.is_collision(point):
                return True
        return False

    def sample(self):
        if np.random.random() < 0.10:   # 10% goal bias
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

    def get_next_clear_waypoint(self, path, lookahead=5):
        """Get next waypoint that is not in collision."""
        for i in range(1, min(lookahead + 1, len(path))):
            wp = np.array(path[i])
            if not self.is_collision(wp):
                return wp
        return np.array(path[min(1, len(path)-1)])


# ═══════════════════════════════════════════════════════════════════════════════
# ORCA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_orca_velocity(my_pos, my_vel, other_pos, other_vel,
                          combined_radius=0.35, tau=2.0, max_speed=0.5):
    rel_pos = other_pos - my_pos
    rel_vel = my_vel - other_vel
    dist    = np.linalg.norm(rel_pos)

    # Already colliding
    if dist < combined_radius:
        if dist < 1e-9:
            away = np.array([1.0, 0.0])
        else:
            away = -rel_pos / dist
        return away * max_speed

    rel_pos_hat = rel_pos / dist

    # VO center in velocity space
    vo_center = rel_pos / tau
    w         = rel_vel - vo_center
    w_len     = np.linalg.norm(w)

    # Check if inside circular cap
    in_circle = np.linalg.norm(rel_vel - vo_center) < combined_radius / tau

    # Leg directions
    sin_a     = min(combined_radius / dist, 1.0)
    cos_a     = np.sqrt(max(1.0 - sin_a**2, 0.0))
    left_leg  = np.array([ cos_a * rel_pos_hat[0] - sin_a * rel_pos_hat[1],
                            sin_a * rel_pos_hat[0] + cos_a * rel_pos_hat[1]])
    right_leg = np.array([ cos_a * rel_pos_hat[0] + sin_a * rel_pos_hat[1],
                           -sin_a * rel_pos_hat[0] + cos_a * rel_pos_hat[1]])

    in_cone = (np.dot(rel_vel, rel_pos_hat) > 0 and
               np.dot(rel_vel, left_leg)    > 0 and
               np.dot(rel_vel, right_leg)   > 0)

    if not (in_circle or in_cone):
        return my_vel   # safe

    if in_circle:
        if w_len < 1e-9:
            n = np.array([-rel_pos_hat[1], rel_pos_hat[0]])
        else:
            n = w / w_len
        u = (combined_radius / tau - w_len) * n
    else:
        proj_left  = rel_vel - np.dot(rel_vel, left_leg)  * left_leg
        proj_right = rel_vel - np.dot(rel_vel, right_leg) * right_leg
        if np.linalg.norm(proj_left) < np.linalg.norm(proj_right):
            n = np.array([-left_leg[1],  left_leg[0]])
            u = np.dot(rel_vel, n) * n
        else:
            n = np.array([ right_leg[1], -right_leg[0]])
            u = np.dot(rel_vel, n) * n

    safe_vel = my_vel + 0.5 * u
    speed    = np.linalg.norm(safe_vel)
    if speed > max_speed:
        safe_vel = safe_vel / speed * max_speed

    return safe_vel


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

def run_robot():
    robot    = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    name     = robot.getName()
    print(f"\n{'='*55}")
    print(f"[INIT] Robot={name}  timestep={timestep}ms")
    print(f"{'='*55}")
    
    plt.ion() 
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    # Pre-set the view
    ax.set_xlim([-10, 10])
    ax.set_ylim([-13, 4])
    ax.set_aspect('equal')

    # ── Motors ──────────────────────────────────────────────────────────
    left_motor  = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    for m in [left_motor, right_motor]:
        m.setPosition(float('inf'))
        m.setVelocity(0.0)

    # ── Lidar ───────────────────────────────────────────────────────────
    lidar = robot.getDevice('LDS-01')
    if not lidar:
        print("[ERROR] Lidar not found!")
        return
    lidar.enable(timestep)
    lidar.enablePointCloud()
    print(f"[INIT] Lidar: {lidar.getHorizontalResolution()} pts  "
          f"max_range={lidar.getMaxRange():.1f}m")

    # ── Supervisor ──────────────────────────────────────────────────────
    robot_node = robot.getSelf()
    other_name = "Robot_2" if name == "Robot_1" else "Robot_1"
    other_node = robot.getFromDef(other_name)
    print(f"[INIT] Other robot: {'FOUND' if other_node else 'NOT FOUND'}")

    # ── Real start position ──────────────────────────────────────────────
    robot.step(timestep)
    pos          = robot_node.getPosition()
    actual_start = [pos[0], pos[1]]
    print(f"[INIT] Start: ({actual_start[0]:.3f}, {actual_start[1]:.3f})")

    # ── Goals — crossing paths to test ORCA ─────────────────────────────
    # Robot_1 starts left (-3, -2.17) → goes to right-bottom (6, -10)
    # Robot_2 starts right (3, -8.32) → goes to left-top (-6, -1)
    if name == "Robot_1":
        goal = [6.0, -10.0]
    else:
        goal = [-6.0, -1.0]

    print(f"[INIT] Goal: ({goal[0]:.1f}, {goal[1]:.1f})")

    # ── Planner ─────────────────────────────────────────────────────────
    planner = AnytimeRRTStar(
        start         = actual_start,
        goal          = goal,
        bounds        = [-9.0, 9.0, -12.0, 3.5],
        step_size     = 0.6,
        search_radius = 1.8
    )

    # ── Seed static obstacles from world knowledge ───────────────────────
    print("[INIT] Seeding known obstacles...")
    for ox, oy, r in STATIC_OBSTACLES:
        planner.add_obstacle(ox, oy, r)
    for ox, oy, r in seed_walls():
        planner.add_obstacle(ox, oy, r)
    print(f"[INIT] Pre-seeded {len(planner.obstacles)} obstacles")

    # ── PHASE 1: Scan + Plan (robot stationary) ──────────────────────────
    print(f"\n[PHASE 1] Scanning + planning for 3 seconds...")
    scan_steps = int(3000 / timestep)

    for s in range(scan_steps):
        if robot.step(timestep) == -1:
            return

        pos     = robot_node.getPosition()
        curr_p  = np.array([pos[0], pos[1]])
        rot     = robot_node.getOrientation()
        heading = np.arctan2(rot[3], rot[0])
        ranges  = lidar.getRangeImage()
        n       = len(ranges)

        # Add lidar hits to map
        # --- OPTIMIZED LIDAR PROCESSING ---
        for i, r in enumerate(ranges):
            # Only process hits within a useful range
            if 0.12 < r < lidar.getMaxRange() * 0.95:
                angle = heading + (2 * np.pi * i / n)
                ox = curr_p[0] + r * np.cos(angle)
                oy = curr_p[1] + r * np.sin(angle)
                new_obs_pt = np.array([ox, oy])

                # 1. Use the KDTree for a fast spatial lookup
                # Check if any existing obstacle is within 0.5m of this new hit
                is_redundant = False
                if planner.obstacles:
                    # Rebuild tree if it was invalidated by a previous addition
                    if planner._obs_tree is None:
                        planner._rebuild_obs_tree()
                    
                    # Find all obstacles within 0.5m
                    nearby_idxs = planner._obs_tree.query_ball_point(new_obs_pt, 0.5)
                    if len(nearby_idxs) > 0:
                        is_redundant = True

                # 2. Only add if it's a "fresh" discovery
                if not is_redundant:
                    planner.add_obstacle(ox, oy, radius=0.4)

        # Expand tree aggressively
        for _ in range(10):
            planner.plan_step()

        if s % 30 == 0:
            print(f"  [SCAN {s}/{scan_steps}] "
                  f"obs={len(planner.obstacles)} "
                  f"nodes={len(planner.nodes)} "
                  f"reachable={planner.goal_reachable()}")

    print(f"\n[PHASE 1] DONE")
    print(f"  obstacles : {len(planner.obstacles)}")
    print(f"  tree nodes: {len(planner.nodes)}")
    print(f"  reachable : {planner.goal_reachable()}")

    path = planner.get_path()
    if len(path) > 1:
        print(f"  path len  : {len(path)}")
        print(f"  path[1]   : ({path[1][0]:.2f},{path[1][1]:.2f})")
        print(f"  path[-1]  : ({path[-1][0]:.2f},{path[-1][1]:.2f})")

    # ── PHASE 2: Move ────────────────────────────────────────────────────
    print(f"\n[PHASE 2] Moving...")

    MAX_SPEED          = 6.67
    WHEEL_RADIUS       = 0.033
    WHEEL_BASE         = 0.16
    K_ANG              = 3.0
    GOAL_TOLERANCE     = 0.35
    OBSTACLE_THRESHOLD = 0.35
    PLAN_EVERY_N       = 3
    ORCA_RADIUS        = 0.4
    ORCA_TAU           = 2.0
    ORCA_MAX_SPEED     = 0.55
    WAYPOINT_TOLERANCE = 0.3
    DEADLOCK_STEPS     = 120
    DEADLOCK_DIST      = 0.1
    ESCAPE_DURATION    = 180  # NEW: How long to maintain escape behavior

    step_count       = 0
    goal_reached     = False
    current_waypoint = None
    wp_stuck_count   = 0
    prev_pos         = np.array(actual_start)
    other_prev_pos   = None
    deadlock_ref     = np.array(actual_start)
    deadlock_count   = 0
    escape_mode      = False
    escape_steps     = 0
    escape_vel       = None

    # ── Committed path — robot follows this until a replan is forced ────
    # Never call get_path() in the main loop unless needs_replan is True.
    committed_path   = planner.get_path() if planner.goal_reachable() else []
    waypoint_idx     = 1       # index in committed_path we are heading toward
    needs_replan     = False   # set True only when path is genuinely blocked

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

        # ── 1. Planner runs only when a replan is explicitly triggered ──
        # (Continuous expansion during movement causes path thrashing —
        #  the tree rewires each step so get_path() returns a different
        #  path every time, and the robot chases a moving target forever.)
        pass

        # ── 2. Odometry ─────────────────────────────────────────────────
        pos     = robot_node.getPosition()
        curr_p  = np.array([pos[0], pos[1]])
        rot     = robot_node.getOrientation()
        heading = np.arctan2(rot[3], rot[0])
        own_vel = (curr_p - prev_pos) / (dt + 1e-9)
        prev_pos = curr_p.copy()

        # ── 3. Goal check ───────────────────────────────────────────────
        dist_to_goal = np.linalg.norm(np.array(planner.goal) - curr_p)
        if dist_to_goal < GOAL_TOLERANCE and not goal_reached:
            print(f"\n{'*'*40}")
            print(f"[GOAL] {name} REACHED GOAL at step {step_count}!")
            print(f"  final pos : ({curr_p[0]:.3f},{curr_p[1]:.3f})")
            print(f"  dist      : {dist_to_goal:.3f}m")
            print(f"{'*'*40}")
            goal_reached = True
        if goal_reached:
            set_speeds(0.0, 0.0)
            continue

        # ── 4. Lidar update ─────────────────────────────────────────────
        ranges = lidar.getRangeImage()
        n      = len(ranges)
        
        other_p = None
        other_vel = np.array([0.0, 0.0])
        
        if other_node:
            op3d = other_node.getPosition()
            # This is where we define other_p for the rest of the loop
            other_p = np.array([op3d[0], op3d[1]]) 
            
            dist_to_other = np.linalg.norm(other_p - curr_p)

            if other_prev_pos is not None:
                other_vel = (other_p - other_prev_pos) / (dt + 1e-9)
            other_prev_pos = other_p.copy()

        

        # --- 3. THE LOOP (Inside while robot.step != -1) ---
        if step_count % 40 == 0:
            # Clear the previous drawing to prepare for the new frame
            ax.clear() 
            
            # Re-apply limits (ax.clear removes them)
            ax.set_xlim([-10, 10])
            ax.set_ylim([-13, 4])
            ax.set_aspect('equal')
            ax.set_title(f"Live Navigation Stack | Step: {step_count}")

            # DRAW OBSTACLES (Static & Lidar)
            obs_pts = np.array([[o[0], o[1]] for o in planner.obstacles])
            if obs_pts.size > 0:
                ax.plot(obs_pts[:, 0], obs_pts[:, 1], 'ro', markersize=1, alpha=0.3)

            # DRAW RRT* TREE
            nodes_np = np.array(planner.nodes)
            for i, p_idx in planner.parents.items():
                if p_idx is not None:
                    ax.plot([nodes_np[i,0], nodes_np[p_idx,0]], 
                            [nodes_np[i,1], nodes_np[p_idx,1]], 
                            color='lightgray', linewidth=0.5, zorder=1)

            # DRAW CURRENT PATH
            path_pts = np.array(committed_path) if len(committed_path) > 1 else np.array([])
            if path_pts.size > 1:
                ax.plot(path_pts[:, 0], path_pts[:, 1], 'g-', linewidth=2, zorder=2)

            # DRAW ROBOTS
            # 'bs' = Blue Square (Self), 'rs' = Red Square (Other)
            ax.plot(curr_p[0], curr_p[1], 'bs', markersize=8, label='Self', zorder=3)
            
            if 'other_p' in locals() and other_p is not None:
                ax.plot(other_p[0], other_p[1], 'rs', markersize=8, label='Robot_2', zorder=3)

            # DRAW GOAL
            ax.plot(planner.goal[0], planner.goal[1], 'k*', markersize=12, label='Goal')

            # Update the specific window instance
            plt.draw()
            plt.pause(0.001)
            
            
            for i, r in enumerate(ranges):
                if 0.12 < r < lidar.getMaxRange() * 0.9:
                    angle = heading + (2 * np.pi * i / n)
                    ox    = curr_p[0] + r * np.cos(angle)
                    oy    = curr_p[1] + r * np.sin(angle)
                    too_close = any(
                        np.linalg.norm(np.array([ox, oy]) - np.array([ex, ey])) < 0.25
                        for ex, ey, _ in planner.obstacles
                    )
                    if not too_close:
                        planner.add_obstacle(ox, oy, radius=0.4)
                        
            

        # ── 5. Lidar arcs ───────────────────────────────────────────────
        front_arc = list(range(0, 30)) + list(range(n - 30, n))
        left_arc  = list(range(30, 90))
        right_arc = list(range(n - 90, n - 30))

        def min_valid(idxs):
            vals = [ranges[i] for i in idxs if 0 < ranges[i] < float('inf')]
            return min(vals) if vals else float('inf')

        min_front = min_valid(front_arc)
        min_left  = min_valid(left_arc)
        min_right = min_valid(right_arc)

        # ── 6. Get other robot info ─────────────────────────────────────
        other_p = None
        other_vel = np.array([0.0, 0.0])
        dist_to_other = float('inf')
        
        if other_node:
            op3d          = other_node.getPosition()
            other_p       = np.array([op3d[0], op3d[1]])
            dist_to_other = np.linalg.norm(other_p - curr_p)

            if other_prev_pos is not None:
                other_vel = (other_p - other_prev_pos) / (dt + 1e-9)
            other_prev_pos = other_p.copy()

        # ── 7. Deadlock detection with ADAPTIVE ESCAPE ──────────────────
        deadlock_count += 1
        
        if escape_mode:
            # Continue escape maneuver
            escape_steps -= 1
            if escape_steps <= 0:
                escape_mode = False
                escape_vel = None
                needs_replan = True   # re-plan from current position after escaping
                print(f"[ESCAPE END] {name} resuming normal navigation")
        
        if deadlock_count >= DEADLOCK_STEPS and not escape_mode:
            moved = np.linalg.norm(curr_p - deadlock_ref)
            if moved < DEADLOCK_DIST:
                print(f"[DEADLOCK] {name} at ({curr_p[0]:.2f},{curr_p[1]:.2f})")
                
                # IMPROVED: Compute adaptive escape direction
                if other_p is not None:
                    # Escape perpendicular to line connecting robots
                    to_other = other_p - curr_p
                    if np.linalg.norm(to_other) > 1e-6:
                        to_other_norm = to_other / np.linalg.norm(to_other)
                        # Perpendicular vector (90 degrees rotation)
                        perp = np.array([-to_other_norm[1], to_other_norm[0]])
                        
                        # Choose direction based on goal
                        to_goal = np.array(planner.goal) - curr_p
                        if np.dot(perp, to_goal) < 0:
                            perp = -perp
                        
                        # Add forward component toward goal
                        escape_vel = 0.7 * perp + 0.3 * (to_goal / (np.linalg.norm(to_goal) + 1e-9))
                        escape_vel = escape_vel / (np.linalg.norm(escape_vel) + 1e-9) * 0.45
                    else:
                        # Fallback if too close
                        escape_vel = np.array([0.4, 0.0]) if name == "Robot_1" else np.array([-0.4, 0.0])
                else:
                    # No other robot info, use goal direction
                    to_goal = np.array(planner.goal) - curr_p
                    escape_vel = to_goal / (np.linalg.norm(to_goal) + 1e-9) * 0.45
                
                escape_mode = True
                escape_steps = ESCAPE_DURATION
                print(f"[ESCAPE START] {name} escaping with vel=({escape_vel[0]:.2f},{escape_vel[1]:.2f})")
                
            deadlock_count = 0
            deadlock_ref   = curr_p.copy()

        # ── 8. Compute desired velocity ─────────────────────────────────
        if escape_mode:
            desired_vel = escape_vel

        else:
            desired_vel = np.array([0.0, 0.0])

            # ── Replan only when explicitly requested ───────────────────
            if needs_replan or len(committed_path) < 2:
                print(f"[REPLAN] {name} replanning from ({curr_p[0]:.2f},{curr_p[1]:.2f})")
                # Give the planner a burst of expansions from current position
                planner.nodes    = [curr_p]
                planner.parents  = {0: None}
                planner.costs    = {0: 0.0}
                planner.start    = curr_p
                for _ in range(500):
                    planner.plan_step()
                committed_path = planner.get_path() if planner.goal_reachable() else []
                waypoint_idx   = 1
                needs_replan   = False
                print(f"[REPLAN] new path length={len(committed_path)}")

            # ── Advance waypoint index along the committed path ─────────
            # Skip waypoints that are already behind us (within tolerance)
            # and skip any that turned out to be in collision (new obstacle).
            while waypoint_idx < len(committed_path) - 1:
                wp = np.array(committed_path[waypoint_idx])
                if planner.is_collision(wp):
                    # This waypoint is now blocked — trigger replan next step
                    needs_replan = True
                    print(f"[BLOCKED WP] {name} waypoint {waypoint_idx} is now in collision")
                    break
                if np.linalg.norm(wp - curr_p) < WAYPOINT_TOLERANCE:
                    waypoint_idx += 1   # reached this waypoint, advance
                else:
                    break

            if len(committed_path) > 1 and not needs_replan:
                target_idx = min(waypoint_idx, len(committed_path) - 1)
                current_waypoint = np.array(committed_path[target_idx])

                to_wp   = current_waypoint - curr_p
                wp_dist = np.linalg.norm(to_wp)
                if wp_dist > 1e-6:
                    desired_vel = (to_wp / wp_dist) * ORCA_MAX_SPEED

        # ── 9. ORCA (skip during escape mode) ──────────────────────────
        if escape_mode:
            safe_vel = desired_vel  # No ORCA during escape
        else:
            safe_vel = desired_vel.copy()
            if other_node and dist_to_other < 4.0:
                safe_vel = compute_orca_velocity(
                    my_pos          = curr_p,
                    my_vel          = desired_vel,
                    other_pos       = other_p,
                    other_vel       = other_vel,
                    combined_radius = ORCA_RADIUS,
                    tau             = ORCA_TAU,
                    max_speed       = ORCA_MAX_SPEED
                )

        # ── 10. Execute ─────────────────────────────────────────────────
        if min_front < 0.25:
            # Very close — back up
            set_speeds(-0.1, -2.0 if min_left < min_right else 2.0)
            if not escape_mode:
                current_waypoint = None

        elif min_front < OBSTACLE_THRESHOLD:
            # Close — turn away
            set_speeds(0.0, -2.5 if min_left < min_right else 2.5)
            if not escape_mode:
                current_waypoint = None

        else:
            safe_speed = np.linalg.norm(safe_vel)
            if safe_speed < 0.01 and not escape_mode:
                # IMPROVED: Use deterministic turning based on relative position
                if other_p is not None:
                    # Turn away from other robot
                    to_other = other_p - curr_p
                    perp_heading = np.arctan2(to_other[1], to_other[0]) + np.pi/2
                    turn_error = (perp_heading - heading + np.pi) % (2 * np.pi) - np.pi
                    turn = 2.0 * np.sign(turn_error)
                else:
                    # Fallback
                    turn = 1.5 if name == "Robot_1" else -1.5
                set_speeds(0.0, turn)
            else:
                desired_heading = np.arctan2(safe_vel[1], safe_vel[0])
                error = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi
                if abs(error) > 0.5:
                    set_speeds(0.0, K_ANG * error)
                else:
                    set_speeds(min(safe_speed, 0.55), K_ANG * error)

        # ── 11. Debug every 60 steps ────────────────────────────────────
        if step_count % 60 == 0:
            wp_str = (f"({current_waypoint[0]:.2f},{current_waypoint[1]:.2f})"
                      if current_waypoint is not None else "None")
            other_dist = "N/A"
            if other_node:
                op = other_node.getPosition()
                other_dist = f"{np.linalg.norm(np.array([op[0],op[1]])-curr_p):.2f}m"
            escape_str = f" ESCAPE={escape_steps}" if escape_mode else ""
            print(f"[{name}|{step_count}] "
                  f"pos=({curr_p[0]:.2f},{curr_p[1]:.2f}) "
                  f"goal={dist_to_goal:.2f} "
                  f"front={min_front:.2f} "
                  f"other={other_dist} "
                  f"nodes={len(planner.nodes)} "
                  f"path={len(committed_path) if not escape_mode else 'N/A'} "
                  f"wp_idx={waypoint_idx}{escape_str}")


if __name__ == "__main__":
    run_robot()