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

import struct
import sys


class ControlHeader(object):
    __slots__ = [
        "command",
        "size",
    ]

    @staticmethod
    def unpack(buf):
        rmd = ControlHeader()
        (rmd.size, rmd.command) = struct.unpack_from(">HB", buf)
        return rmd


class ControlVersion(object):
    __slots__ = ["major", "minor", "bugfix", "build"]

    @staticmethod
    def unpack(buf):
        rmd = ControlVersion()
        (rmd.major, rmd.minor, rmd.bugfix, rmd.build) = struct.unpack_from(">IIII", buf)
        return rmd


class ReturnValue(object):
    __slots__ = ["success"]

    @staticmethod
    def unpack(buf):
        rmd = ReturnValue()
        rmd.success = bool(struct.unpack_from(">B", buf)[0])
        return rmd


class MessageV1(object):
    @staticmethod
    def unpack(buf):
        rmd = Message()  # use V2 message object
        offset = 0
        rmd.level = struct.unpack_from(">B", buf, offset)[0]
        offset = offset + 1
        rmd.message = str(buf[offset:])
        rmd.source = ""

        return rmd


class Message(object):
    __slots__ = ["level", "message", "source"]
    EXCEPTION_MESSAGE = 0
    ERROR_MESSAGE = 1
    WARNING_MESSAGE = 2
    INFO_MESSAGE = 3

    @staticmethod
    def unpack(buf):
        rmd = Message()
        offset = 0
        msg_length = struct.unpack_from(">B", buf, offset)[0]
        offset = offset + 1
        rmd.message = str(buf[offset : offset + msg_length])
        offset = offset + msg_length

        src_length = struct.unpack_from(">B", buf, offset)[0]
        offset = offset + 1
        rmd.source = str(buf[offset : offset + src_length])
        offset = offset + src_length
        rmd.level = struct.unpack_from(">B", buf, offset)[0]

        return rmd


def get_item_size(data_type):
    if data_type.startswith("VECTOR6"):
        return 6
    elif data_type.startswith("VECTOR3"):
        return 3
    return 1


def unpack_field(data, offset, data_type):
    size = get_item_size(data_type)
    if data_type == "VECTOR6D" or data_type == "VECTOR3D":
        return [float(data[offset + i]) for i in range(size)]
    elif data_type == "VECTOR6UINT32":
        return [int(data[offset + i]) for i in range(size)]
    elif data_type == "DOUBLE":
        return float(data[offset])
    elif data_type == "UINT32" or data_type == "UINT64":
        return int(data[offset])
    elif data_type == "VECTOR6INT32":
        return [int(data[offset + i]) for i in range(size)]
    elif data_type == "INT32" or data_type == "UINT8":
        return int(data[offset])
    elif data_type == "BOOL":
        return bool(data[offset])
    raise ValueError("unpack_field: unknown data type: " + data_type)


class DataObject(object):
    recipe_id = None

    def pack(self, names, types):
        if len(names) != len(types):
            raise ValueError("List sizes are not identical.")
        l = []
        if self.recipe_id is not None:
            l.append(self.recipe_id)
        for i in range(len(names)):
            if self.__dict__[names[i]] is None:
                raise ValueError("Uninitialized parameter: " + names[i])
            if types[i].startswith("VECTOR"):
                l.extend(self.__dict__[names[i]])
            else:
                l.append(self.__dict__[names[i]])
        return l

    @staticmethod
    def unpack(data, names, types):
        if len(names) != len(types):
            raise ValueError("List sizes are not identical.")
        obj = DataObject()
        offset = 0
        obj.recipe_id = data[0]
        for i in range(len(names)):
            obj.__dict__[names[i]] = unpack_field(data[1:], offset, types[i])
            offset += get_item_size(types[i])
        return obj

    @staticmethod
    def create_empty(names, recipe_id):
        obj = DataObject()
        for i in range(len(names)):
            obj.__dict__[names[i]] = None
        obj.recipe_id = recipe_id
        return obj


class DataConfig(object):
    __slots__ = ["id", "names", "types", "fmt"]

    @staticmethod
    def unpack_recipe(buf):
        rmd = DataConfig()
        python_version = sys.version_info
        if python_version.major == 2:
            rmd.id = struct.unpack_from(">B", buf)[0]
            rmd.types = buf.decode("utf-8")[1:].split(",")
        else:
            rmd.id = buf[0]
            rmd.types = buf[1:].decode("utf-8").split(",")
        rmd.fmt = ">B"
        for i in rmd.types:
            if i == "INT32":
                rmd.fmt += "i"
            elif i == "UINT32":
                rmd.fmt += "I"
            elif i == "VECTOR6D":
                rmd.fmt += "d" * 6
            elif i == "VECTOR3D":
                rmd.fmt += "d" * 3
            elif i == "VECTOR6INT32":
                rmd.fmt += "i" * 6
            elif i == "VECTOR6UINT32":
                rmd.fmt += "I" * 6
            elif i == "DOUBLE":
                rmd.fmt += "d"
            elif i == "UINT64":
                rmd.fmt += "Q"
            elif i == "UINT8":
                rmd.fmt += "B"
            elif i == "BOOL":
                rmd.fmt += "?"
            elif i == "IN_USE":
                raise ValueError("An input parameter is already in use.")
            else:
                raise ValueError("Unknown data type: " + i)
        return rmd

    def pack(self, state):
        l = state.pack(self.names, self.types)
        return struct.pack(self.fmt, *l)

    def unpack(self, data):
        li = struct.unpack_from(self.fmt, data)
        return DataObject.unpack(li, self.names, self.types)
