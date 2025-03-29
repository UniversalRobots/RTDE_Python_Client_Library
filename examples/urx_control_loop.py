import time

import urx
import urx.robot
import urx.urrobot
import numpy as np

import time
import urx
import numpy as np

robot = urx.Robot("10.0.12.245")
try:
    # Initialize connection
    time.sleep(1)  # Wait for robot
    #robot.set_tcp((0, 0, 0, 0, 0, 0))  # Default TCP
    robot.set_payload(0.5)  # Default payload

    # First movement (slower)
    target_pose = [0.2, 0.35, 0.3, 0, 3.14, 0]  # Use list instead of np.array
    robot.movel(target_pose, acc=0.2, vel=0.20)  # Reduced speed/acceleration
    print("Executed move 1")

    time.sleep(0.1)  # Pause between moves

    # Second movement
    target_pose = [0.05, 0.35, 0.3, 0, 3.14, 0]
    robot.movel(target_pose, acc=0.5, vel=0.05)
    print("Executed move 2")

finally:
    robot.close()

#pip uninstall urx
#pip install urx==0.8.2

#had to do modifications to the library:
#"https://github.com/SintefManufacturing/python-urx/issues/117"
#"https://github.com/SintefManufacturing/python-urx/issues/123"