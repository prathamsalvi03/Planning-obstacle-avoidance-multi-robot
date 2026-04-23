# import sys
# import os

# # Ensure the Webots controller library is in the path
# # Replace '/usr/local/webots' with your actual WEBOTS_HOME if different
# os.environ['WEBOTS_HOME'] = '/usr/local/webots'
# sys.path.append(os.path.join(os.environ['WEBOTS_HOME'], 'lib/controller/python'))

# from controller import Robot, Supervisor
# import numpy as np

# class RRTPlannerController(Supervisor):
#     def __init__(self):
#         super().__init__()
#         self.time_step = int(self.getBasicTimeStep())
        
#         # Initialize Sensors (GPS for RRT* start state)
#         self.gps = self.getDevice('gps')
#         self.gps.enable(self.time_step)
        
#         # Initialize Motors
#         self.left_motor = self.getDevice('left wheel motor')
#         self.right_motor = self.getDevice('right wheel motor')
#         self.left_motor.setPosition(float('inf'))
#         self.right_motor.setPosition(float('inf'))
#         self.left_motor.setVelocity(0.0)
#         self.right_motor.setVelocity(0.0)

#     def run(self):
#         while self.step(self.time_step) != -1:
#             # 1. Get current position for RRT* re-rooting
#             pos = self.gps.getValues()
            
#             # 2. Execute Anytime RRT* logic here (from Karaman & Frazzoli)
#             # 3. Compute velocities based on the path
            
#             self.left_motor.setVelocity(1.0)
#             self.right_motor.setVelocity(1.0)

# if __name__ == "__main__":
#     controller = RRTPlannerController()
#     controller.run()



import os
import sys

sys.path.append(os.path.join(os.environ['WEBOTS_HOME'], 'lib/controller/python'))
from controller import Robot, Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

print(f"Connected to: {robot.getName()}")

# Main control loop
while robot.step(timestep) != -1:
    # Your planning / control logic goes here
    pass