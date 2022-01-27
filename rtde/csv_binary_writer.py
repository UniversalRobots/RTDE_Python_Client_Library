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

sys.path.append("..")

import struct
from rtde import serialize


class CSVBinaryWriter(object):
    def __init__(self, file, names, types, delimiter=" "):
        if len(names) != len(types):
            raise ValueError("List sizes are not identical.")
        self.__file = file
        self.__names = names
        self.__types = types
        self.__delimiter = delimiter
        self.__header_names = []
        self.__columns = 0
        for i in range(len(self.__names)):
            size = serialize.get_item_size(self.__types[i])
            self.__columns += size
            if size > 1:
                for j in range(size):
                    name = self.__names[i] + "_" + str(j)
                    self.__header_names.append(name)
            else:
                name = self.__names[i]
                self.__header_names.append(name)

    def getType(self, vtype):
        if vtype == "VECTOR3D":
            return "DOUBLE" + self.__delimiter + "DOUBLE" + self.__delimiter + "DOUBLE"
        elif vtype == "VECTOR6D":
            return (
                "DOUBLE"
                + self.__delimiter
                + "DOUBLE"
                + self.__delimiter
                + "DOUBLE"
                + self.__delimiter
                + "DOUBLE"
                + self.__delimiter
                + "DOUBLE"
                + self.__delimiter
                + "DOUBLE"
            )
        elif vtype == "VECTOR6INT32":
            return (
                "INT32"
                + self.__delimiter
                + "INT32"
                + self.__delimiter
                + "INT32"
                + self.__delimiter
                + "INT32"
                + self.__delimiter
                + "INT32"
                + self.__delimiter
                + "INT32"
            )
        elif vtype == "VECTOR6UINT32":
            return (
                "UINT32"
                + self.__delimiter
                + "UINT32"
                + self.__delimiter
                + "UINT32"
                + self.__delimiter
                + "UINT32"
                + self.__delimiter
                + "UINT32"
                + self.__delimiter
                + "UINT32"
            )
        else:
            return str(vtype)

    def writeheader(self):
        # Header names
        headerStr = str("")
        for i in range(len(self.__header_names)):
            if i != 0:
                headerStr += self.__delimiter

            headerStr += self.__header_names[i]

        headerStr += "\n"
        self.__file.write(struct.pack(str(len(headerStr)) + "s", headerStr))

        # Header types
        typeStr = str("")
        for i in range(len(self.__names)):
            if i != 0:
                typeStr += self.__delimiter

            typeStr += self.getType(self.__types[i])

        typeStr += "\n"
        self.__file.write(struct.pack(str(len(typeStr)) + "s", typeStr))

    def packToBinary(self, vtype, value):
        print(vtype)
        if vtype == "BOOL":
            print("isBOOL" + str(value))
        if vtype == "UINT8":
            print("isUINT8" + str(value))
        elif vtype == "INT32":
            print("isINT32" + str(value))
        elif vtype == "INT64":
            print("isINT64" + str(value))
        elif vtype == "UINT32":
            print("isUINT32" + str(value))
        elif vtype == "UINT64":
            print("isUINT64" + str(value))
        elif vtype == "DOUBLE":
            print(
                "isDOUBLE" + str(value) + str(type(value)) + str(sys.getsizeof(value))
            )
        elif vtype == "VECTOR3D":
            print(
                "isVECTOR3D" + str(value[0]) + "," + str(value[1]) + "," + str(value[2])
            )
        elif vtype == "VECTOR6D":
            print(
                "isVECTOR6D"
                + str(value[0])
                + ","
                + str(value[1])
                + ","
                + str(value[2])
                + ","
                + str(value[3])
                + ","
                + str(value[4])
                + ","
                + str(value[5])
            )
        elif vtype == "VECTOR6INT32":
            print(
                "isVECTOR6INT32"
                + str(value[0])
                + ","
                + str(value[1])
                + ","
                + str(value[2])
                + ","
                + str(value[3])
                + ","
                + str(value[4])
                + ","
                + str(value[5])
            )
        elif vtype == "VECTOR6UINT32":
            print(
                "isVECTOR6UINT32"
                + str(value[0])
                + ","
                + str(value[1])
                + ","
                + str(value[2])
                + ","
                + str(value[3])
                + ","
                + str(value[4])
                + ","
                + str(value[5])
            )

    def writerow(self, data_object):
        self.__file.write(data_object)
