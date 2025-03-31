# Create IK solver instance
from examples.iksolver import URIKSolver
import numpy as np
from math import pi

# source for DH parameters:
#make sure to use UR3 and not UR3e
#there are slight differences.
#
# "https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/"
a = [0, -0.24365, -0.21325, 0, 0, 0]
d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]
alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
weights = [6, 5, 4, 3, 2, 1]

ik_solver = URIKSolver(a=a, d=d, alpha=alpha, w=weights)

#First run: "[ 0.5672413  -1.79902643 -2.00784733  2.10207116  0.93183217  1.6228427 ]"
#[ 0.5672413  -1.79902643 -2.00784733  2.10207116  0.93183217  1.6228427 ]
# our wanted position x, y, z, roll, pitch, yaw
position = np.array([0.3, 0.0, 0.3, np.pi/4, 0, np.pi/4])

#orientation = np.array([np.pi/4, 0, np.pi/4])

#initial position, joint 1, 2, 3, 4, 5, 6
initialPosition = np.array([0, 0, 0, 0, 0, 0])

# Solve IK
jointAngles = ik_solver.solve(position, initialPosition)
print("Solution joint angles:", jointAngles)

for i in jointAngles:
    print(i*360/(2*pi))



# UR3 DH parameters (modify as needed)
a = [0, -0.24365, -0.21325, 0, 0, 0]
d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]
alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
weights = np.array([6, 5, 4, 3, 2, 1])  # Prioritize base joints

ik_solver = URIKSolver(a, d, alpha, weights)
posEul = [0.2, 0.1, 0.3, 0, -np.pi, 0]
initialPos = np.zeros(6)

solution_radians = ik_solver.solve(posEul, initialPos)
print("Joint angles (radians):", solution_radians)

for i in solution_radians:
    print(i*360/(2*pi))