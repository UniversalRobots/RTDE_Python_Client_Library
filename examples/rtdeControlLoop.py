import time
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import rtde_io



#how to download??
#https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html
# Connect to the robot using RTDE (port 30004)
robot_ip = "10.0.12.245"
rtde_c = RTDEControlInterface(robot_ip)
rtde_r = RTDEReceiveInterface(robot_ip)

try:
    # Set payload (same as urx)
    rtde_c.setPayload(0.5)

    # First movement (slower)
    target_pose_1 = [0.2, 0.35, 0.3, 0, 3.14, 0]  # Pose in meters and radians
    rtde_c.moveL(target_pose_1, speed=0.2, acceleration=0.2)  # Linear move
    print("Executed move 1")

    time.sleep(0.1)  # Pause between moves

    # Second movement
    target_pose_2 = [0.05, 0.35, 0.3, 0, 3.14, 0]
    rtde_c.moveL(target_pose_2, speed=0.5, acceleration=0.5)
    print("Executed move 2")

finally:
    # Stop RTDE communication properly
    rtde_c.stopScript()
