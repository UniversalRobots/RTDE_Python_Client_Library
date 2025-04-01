#!/usr/bin/env python
# Copyright (c) 2016-2022, Universal Robots A/S,
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Universal Robots A/S nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL UNIVERSAL ROBOTS A/S BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
#from ur_ikfast import ur_kinematics
sys.path.append("..")
import logging
import urx
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
from math import pi
from examples.iksolver import URIKSolver
import numpy as np

# inverse kinematics library :
# https://github.com/cambel/ur_ikfast
# logging.basicConfig(level=logging.INFO)
# Robot IP


#https://www.universal-robots.com/articles/ur/application-installation/dealing-with-getinverse-unable-to-find-a-solution/

ROBOT_HOST = "10.0.12.245"
#usually standard for RTDE and UR robots
ROBOT_PORT = 30004
config_filename = "../xmlDataReader/control_loop_configuration.xml"

keep_running = True

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe("state")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

# get controller version
con.get_controller_version()

# setup recipes
payload = {
    "mass": 0.5,  # Mass of the payload
    "center_of_gravity": [0.0, 0.0, 0.1],  # Center of gravity coordinates
}


#con.send_payload(payload["mass"], payload["center_of_gravity"])
#con.set_payload(0.5)

con.send_output_setup(state_names, state_types)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)


setp.input_double_register_0 = pi/2
setp.input_double_register_1 = -pi/2
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0

# The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
watchdog.input_int_register_0 = 0


def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        sp_list.append(sp.__dict__["input_double_register_%i" % i])
    return sp_list

def list_to_setp(sp, list):
    for i in range(0, 6):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp


a = [0, -0.24365, -0.21325, 0, 0, 0]
d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]
alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
weights = [6, 5, 4, 3, 2, 1]

ikSolver = URIKSolver(a=a, d=d, alpha=alpha, w=weights)


# our wanted position x, y, z, roll, pitch, yaw
position = np.array([0.048, 0.506, -0.255, 2.71, 1.35, 0])

#initial position, joint 1, 2, 3, 4, 5, 6
#WE HAVE TO FILL IN THE RIGHT INITIAL POSITIN TO GET THE RIGHT SOLUTION
initialPositionDegrees = [98.08, -149.00, -42.53, -83.97, 88.02, 0]
#initialPosition = np.array([0, 0, 0, 0, 0, 0])
initialPosition = np.array([x*pi/180 for x in initialPositionDegrees])
posEul = [0, 0.3, 0.4, 0, 0, 0]
posEul2 = [0, -0.3, 0.3, 0, -np.pi, 0]
#Some for-loop or recursive calling here???:


wantedPos1 = ikSolver.solve(posEul, initialPosition)
wantedPos1[5] = wantedPos1[5] + pi


setpoints_queue = [ #angles from joint 1 to the last
    [1.5, -1.57, 0.0, -1.57, 1.57, 0.0]
    #[1.57, -1.0, 0.5, -0.5, 1.0, 0.0],
#kSolver.solve(posEul2, wantedPos1)
    #can add many more.
]
current_setpoint_index = 0

# start data synchronization
if not con.send_start():
    sys.exit()

# control loop
move_completed = True
while keep_running:
    # receive the current state
    state = con.receive()

    if state is None:
        break

    if move_completed and state.output_int_register_0 == 1:
        if current_setpoint_index < len(setpoints_queue):
            move_completed = False
            new_setp = setpoints_queue[current_setpoint_index]
            current_setpoint_index += 1
            list_to_setp(setp, new_setp)
            print(f"Moving to setpoint {current_setpoint_index}: {[x*180/pi for x in new_setp]}")
            print(f"Moving to setpoint {current_setpoint_index}: {new_setp}")
            print(f"")
            # send new setpoint
            con.send(setp)
            watchdog.input_int_register_0 = 1
        else:
            print("All setpoints completed!")
            keep_running = False  # or set to True if you want to loop forever
    elif not move_completed and state.output_int_register_0 == 0:
        print("Move to confirmed pose = " + str(state.target_q))
        move_completed = True
        watchdog.input_int_register_0 = 0

    # kick watchdog
    con.send(watchdog)

con.send_pause()