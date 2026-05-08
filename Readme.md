# Multi-Robot Planning & Obstacle Avoidance

[![Webots](https://img.shields.io/badge/Simulator-Webots-blue)](https://cyberbotics.com/)
[![Python](https://img.shields.io/badge/Language-Python%203.10+-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-red)](LICENSE)

An advanced autonomous navigation stack implementing **Anytime RRT*** for global path refinement and **ORCA (Optimal Reciprocal Collision Avoidance)** for multi-robot coordination. Developed in the Webots simulator, this framework allows multiple TurtleBot3 Burger agents to navigate dynamic environments while avoiding static obstacles and each other.

## 🚀 Key Features

* **Anytime RRT* Global Planner**: Provides a rapid initial solution and asymptotically approaches the optimal path as the robot moves.
* **ORCA Coordination**: Implements velocity-space reciprocal collision avoidance, ensuring smooth multi-agent interactions without oscillations.
* **Real-time Lidar Integration**: Uses **LDS-01** sensor data to build a dynamic obstacle map in a 2D Euclidean space.
* **Hierarchical Control**: Combines high-level path planning with a robust **PID Heading Controller** for differential-drive kinematics.
* **Hardware Agnostic**: Tested on both **Pioneer 3-AT** and **TurtleBot3 Burger** platforms.

## 🛠 Tech Stack

* **Simulator**: Webots R2025a
* **Libraries**: `NumPy` (Vector Math), `SciPy` (KDTree for $O(\log n)$ neighbor search).
* **Algorithms**: RRT*, ORCA, PID Control.

## 📁 Project Structure

```text
├── controller/
│   └── planner_anytime.py   # Main multi-robot controller logic
├── protos/
│   └── TurtleBot3Burger.proto # Robot hardware definitions
├── worlds/
│   └── factory_floor.wbt    # Simulation environment
└── README.md



⚙️ Installation & Setup
Clone the Repository:

Bash
git clone [https://github.com/prathamsalvi03/Planning-obstacle-avoidance-multi-robot.git](https://github.com/prathamsalvi03/Planning-obstacle-avoidance-multi-robot.git)
cd Planning-obstacle-avoidance-multi-robot
Install Dependencies:

Bash
pip install numpy scipy
Simulation Configuration:

Open Webots and load the factory world.

Ensure each robot node has supervisor TRUE.

Set the controller field to <extern>.

🏃 Running the Simulation
Press Play in Webots.

Launch the controller for each robot in separate terminals:

Bash
# Terminal 1: Robot 1
WEBOTS_CONTROLLER_URL=ipc://1234/Robot_1 python3 planner_anytime.py

# Terminal 2: Robot 2
WEBOTS_CONTROLLER_URL=ipc://1234/Robot_2 python3 planner_anytime.py
