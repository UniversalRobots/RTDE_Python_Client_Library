#!/usr/bin/env python
# Read and decode RTDE controller properties (software version, control box, tool flange).
# Supported from Software version 5.26.0 / 10.15.0
# Usage Example: python example_read_properties.py --host 192.168.1.100
#
# Copyright (c) 2026, Universal Robots A/S,
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

import argparse
import sys

sys.path.append("..")
import rtde.rtde as rtde
import rtde.rtde_decode as rtde_decode

PROPERTY_NAMES = [
    "v1.software.version",
    "v1.control_box.type",
    "v1.robot_arm.tool_flange.type",
]

parser = argparse.ArgumentParser(description="Read RTDE properties from a controller")
parser.add_argument(
    "--host", default="localhost", help="name of host to connect to (localhost)"
)
parser.add_argument("--port", type=int, default=30004, help="port number (30004)")
args = parser.parse_args()

con = rtde.RTDE(args.host, args.port)
con.connect()

try:
    # Unpacked wire values (types + packed payload decoded in rtde.py)
    props = con.read_properties(PROPERTY_NAMES)
except rtde.RTDEPropertyReadError as e:
    print("RTDE property read failed: %s" % e)
    sys.exit(1)
except rtde.RTDEException as e:
    print("RTDE properties not supported: %s" % e)
    sys.exit(1)

# Semantic decode of known properties
sv = rtde_decode.decode_property(
    "v1.software.version", props["v1.software.version"]
)
cb = rtde_decode.decode_property("v1.control_box.type", props["v1.control_box.type"])
tf = rtde_decode.decode_property(
    "v1.robot_arm.tool_flange.type", props["v1.robot_arm.tool_flange.type"]
)
print(
    "software_version=%d.%d.%d.%d"
    % (sv["major"], sv["minor"], sv["patch"], sv["build"])
)
print("control_box type=%d subtype=%d" % (cb["type"], cb["subtype"]))
print("tool_flange type=%d subtype=%d" % (tf["type"], tf["subtype"]))

con.disconnect()
