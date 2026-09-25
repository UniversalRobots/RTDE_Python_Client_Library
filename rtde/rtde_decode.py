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

from .rtde import RTDEException


def _decode_software_version(wire_value):
    return {
        "major": (wire_value >> 48) & 0xFFFF,
        "minor": (wire_value >> 32) & 0xFFFF,
        "patch": (wire_value >> 16) & 0xFFFF,
        "build": wire_value & 0xFFFF,
    }


def _decode_hardware_type(wire_value):
    return {
        "type": (wire_value >> 24) & 0xFF,
        "subtype": (wire_value >> 16) & 0xFF,
    }


_PROPERTY_DECODERS = {
    "v1.software.version": _decode_software_version,
    "v1.control_box.type": _decode_hardware_type,
    "v1.robot_arm.tool_flange.type": _decode_hardware_type,
}


def decode_property(name, wire_value):
    """Map an unpacked RTDE property value to semantic fields.

    :meth:`RTDE.read_properties` returns unpacked wire values (e.g. UINT64).
    This function interprets known properties such as software version and
    hardware type.

    Args:
        name: property name string (e.g. ``"v1.software.version"``).
        wire_value: unpacked integer from :meth:`RTDE.read_properties`.

    Returns:
        dict with property-specific semantic keys.

    Raises:
        RTDEException: if ``name`` has no registered decoder.
    """
    decoder = _PROPERTY_DECODERS.get(name)
    if decoder is None:
        raise RTDEException("unknown RTDE property: {}".format(name))
    return decoder(wire_value)
