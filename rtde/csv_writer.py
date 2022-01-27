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

import csv

import sys

sys.path.append("..")

from rtde import serialize


class CSVWriter(object):
    def __init__(self, csvfile, names, types, delimiter=" "):
        if len(names) != len(types):
            raise ValueError("List sizes are not identical.")
        self.__names = names
        self.__types = types
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
        self.__writer = csv.writer(csvfile, delimiter=delimiter)

    def writeheader(self):
        self.__writer.writerow(self.__header_names)

    def writerow(self, data_object):
        data = []
        for i in range(len(self.__names)):
            size = serialize.get_item_size(self.__types[i])
            value = data_object.__dict__[self.__names[i]]
            if size > 1:
                data.extend(value)
            else:
                data.append(value)
        self.__writer.writerow(data)
