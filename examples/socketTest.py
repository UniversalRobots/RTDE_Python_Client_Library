import socket
import time
#robot_ip = "10.0.12.245"
#port = 30004  # Replace with the port you want to test
from examples.iksolver import URIKSolver
import numpy as np
from math import pi
import sys
#import socket

#DH parameters
a = [0, -0.24365, -0.21325, 0, 0, 0]
d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]
alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
weights = [6, 5, 4, 3, 2, 1]


HOST = "10.0.12.245"  # Robot IP
PORT = 30002
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


ikSolver = URIKSolver(a=a, d=d, alpha=alpha, w=weights)


# our wanted position x, y, z, roll, pitch, yaw
position = np.array([0.048, 0.506, -0.255, 2.71, 1.35, 0])

#initial position, joint 1, 2, 3, 4, 5, 6
#WE HAVE TO FILL IN THE RIGHT INITIAL POSITIN TO GET THE RIGHT SOLUTION
initialPositionDegrees = [98.08, -149.00, -42.53, -83.97, 88.02, 0]
#initialPosition = np.array([0, 0, 0, 0, 0, 0])
wantedPositionDegrees = [32.66, -91.08, 28.65, -28.65, 57.30, 0]
wantedposXYZ = [-0.067, -0.2278, 0.40, 0.264, 0.530, -0.986]
#Some for-loop or recursive calling here???:
print(f" wanted pos xyz joint angles : {ikSolver.solve(wantedposXYZ, [x*pi/180 for x in wantedPositionDegrees])}")
print(f" wanted pos xyz position to degrees : {(ikSolver.solve(wantedposXYZ, [x*pi/180 for x in wantedPositionDegrees]))*180/pi}")
print(f"equal to {[x*pi/180 for x in wantedPositionDegrees]}")

wantedPos3 = ikSolver.solve([0.173, 0.30, 0.20564, pi, 0, 0], [0.57, -1.0, 0.5, -0.5, 1.0, 0.0])
setpointsQueue = [ #angles from joint 1 to the last
[1.5, -1.57, 0.0, -1.57, 1.57, 0.0],
[-1.5, -1.57, 0.0, -1.57, 1.57, 0.0]
    #[0.5, -1.0, 0.5, -1.5, 1.0, 0], #this worked veery well.
    #[0.57, -1.0, 0.5, -0.5, 1.0, 0.0],
    #[x*pi/180 for x in wantedPositionDegrees], #pos x, y, z inputs and then anges in radians
#(ikSolver.solve(wantedposXYZ, [x*pi/180 for x in wantedPositionDegrees])),
#kSolver.solve(posEul2, wantedPos1)
    #can add many more.
]
print(f"the 3rd command: {ikSolver.solve([0.0, 0.3, 0.3, 0, -pi/2, 0], [0.57, -1.0, 0.5, -0.5, 1.0, 0.0])}")

try:
    print(f"Connecting to robot at {HOST}:{PORT}...")
    s.connect((HOST, PORT))

    print("Connection established!")

    for i in range(len(setpointsQueue)):
        command = f"movej({[round(p, 3) for p in setpointsQueue[i]]}, a=1.4, v=1.0)" + "\n"
        s.send(command.encode('utf8'))
        print(f"nr {i+1} executed")
        time.sleep(5)

    print("All setpoints sent successfully!")

    s.close()

except socket.error as msg:
    print(f"Could not connect to the robot at {HOST}:{PORT}. Error: {msg}")
    sys.exit(1)

except Exception as e:
    # Handle any other unexpected exceptions
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)  # Terminate the program if an unknown error occurs

finally:
    # Ensure the socket is closed properly even if an error occurs
    if s:
        s.close()
        print("Socket closed.")
#https://www.zacobria.com/universal-robots-knowledge-base-tech-support-forum-hints-tips-cb2-cb3/index.php/ur-script-send-commands-from-host-pc-to-robot-via-socket-connection/
#https://forum.universal-robots.com/t/send-ur-script-commands-via-sockets-in-python/22751