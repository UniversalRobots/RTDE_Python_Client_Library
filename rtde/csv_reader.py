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
import numpy as np
import logging

from .rtde import LOGNAME

_log = logging.getLogger(LOGNAME)


runtime_state = "runtime_state"
runtime_state_running = "2"


class CSVReader(object):
    __samples = None
    __filename = None

    def get_header_data(self, __reader):
        header = next(__reader)
        return header

    def __init__(self, csvfile, delimiter=" ", filter_running_program=False):
        self.__filename = csvfile.name

        csvfile = [
            csvfile for csvfile in csvfile.readlines() if csvfile.strip()
        ]  # remove any empty lines

        reader = csv.reader(csvfile, delimiter=delimiter)
        header = self.get_header_data(reader)

        # read csv file
        data = [row for row in reader]

        if len(data) == 0:
            _log.warn("No data read from file: " + self.__filename)

        # filter data
        if filter_running_program:
            if runtime_state not in header:
                _log.warn(
                    "Unable to filter data since runtime_state field is missing in data set"
                )
            else:
                idx = header.index(runtime_state)
                data = [row for row in data if row[idx] == runtime_state_running]

        self.__samples = len(data)

        if self.__samples == 0:
            _log.warn("No data left from file: " + self.__filename + " after filtering")

        # transpose data
        data = list(zip(*data))

        # create dictionary from  header elements (keys) to float arrays
        self.__dict__.update(
            {
                header[i]: np.array(list(map(float, data[:][i])))
                for i in range(len(header))
            }
        )

    def get_samples(self):
        return self.__samples

    def get_name(self):
        return self.__filename
