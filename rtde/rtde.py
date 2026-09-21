# Copyright (c) 2020-2026, Universal Robots A/S,
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
import socket
import select
import sys
import logging

if sys.version_info[0] < 3:
    import serialize
else:
    from rtde import serialize

DEFAULT_TIMEOUT = 1.0

LOGNAME = "rtde"
_log = logging.getLogger(LOGNAME)


class Command:
    RTDE_REQUEST_PROTOCOL_VERSION = ord("V")  # V=86
    RTDE_GET_URCONTROL_VERSION = ord("v")  # v=118
    RTDE_TEXT_MESSAGE = ord("M")  # M=77
    RTDE_DATA_PACKAGE = ord("U")  # U=85
    RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS = ord("O")  # O=79
    RTDE_CONTROL_PACKAGE_SETUP_INPUTS = ord("I")  # I=73
    RTDE_CONTROL_PACKAGE_START = ord("S")  # S=83
    RTDE_CONTROL_PACKAGE_PAUSE = ord("P")  # P=80
    RTDE_READ_PROPERTIES = ord("R")  # R=82


class Protocol:
    VERSION_1 = 1
    VERSION_2 = 2
    VERSION_3 = 3


RTDE_PROTOCOL_VERSION_1 = Protocol.VERSION_1
RTDE_PROTOCOL_VERSION_2 = Protocol.VERSION_2
RTDE_PROTOCOL_VERSION_3 = Protocol.VERSION_3


class ConnectionState:
    DISCONNECTED = 0
    CONNECTED = 1
    STARTED = 2
    PAUSED = 3


class RTDEException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class RTDETimeoutException(RTDEException):
    def __init__(self, msg):
        super(RTDETimeoutException, self).__init__(msg)


class RTDEPropertyReadError(RTDEException):
    """Raised when one or more requested RTDE properties failed to read.

    Attributes:
        tokens: dict mapping each failed property name to its failure
                token (``"NOT_FOUND"`` or ``"NOT_SET"``).
    """

    def __init__(self, tokens):
        self.tokens = tokens
        msg = "; ".join(
            "{}: {}".format(name, token) for name, token in tokens.items()
        )
        super(RTDEPropertyReadError, self).__init__(msg)


class RTDE(object):
    def __init__(self, hostname, port=30004):
        self.hostname = hostname
        self.port = port
        self.__conn_state = ConnectionState.DISCONNECTED
        self.__sock = None
        self.__output_config = None
        self.__input_config = {}
        self.__skipped_package_count = 0
        self.__protocolVersion = Protocol.VERSION_1
        self.__warning_counter = {}

    def connect(self):
        if self.__sock:
            return

        self.__buf = b""  # buffer data in binary format
        try:
            self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.__sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.__sock.settimeout(DEFAULT_TIMEOUT)
            self.__skipped_package_count = 0
            self.__sock.connect((self.hostname, self.port))
            self.__conn_state = ConnectionState.CONNECTED
        except (socket.timeout, socket.error):
            self.__sock = None
            raise
        if not self.negotiate_protocol_version():
            raise RTDEException("Unable to negotiate protocol version")

    def disconnect(self):
        if self.__sock:
            self.__sock.close()
            self.__sock = None
        self.__conn_state = ConnectionState.DISCONNECTED

    def is_connected(self):
        return self.__conn_state is not ConnectionState.DISCONNECTED

    def get_controller_version(self):
        cmd = Command.RTDE_GET_URCONTROL_VERSION
        version = self.__sendAndReceive(cmd)
        if version:
            _log.info(
                "Controller version: "
                + str(version.major)
                + "."
                + str(version.minor)
                + "."
                + str(version.bugfix)
                + "."
                + str(version.build)
            )
            if version.major == 3 and version.minor <= 2 and version.bugfix < 19171:
                _log.error(
                    "Please upgrade your controller to minimally version 3.2.19171"
                )
                sys.exit()
            return version.major, version.minor, version.bugfix, version.build
        return None, None, None, None

    def read_properties(self, names):
        """Read named properties from the controller.

        Unpacks the RTDE_READ_PROPERTIES reply into wire values. The reply
        carries a comma-separated type list (or ``NOT_FOUND`` / ``NOT_SET``
        tokens) followed by packed values. Semantic decoding of known
        properties is done separately with :func:`rtde.rtde_decode.decode_property`.

        Args:
            names: list of property name strings
                   (e.g. ``["v1.software.version", "v1.control_box.type"]``).

        Returns:
            dict mapping each property name to its unpacked wire value
            (scalar for single-item types, list for VECTOR* types).

        Raises:
            RTDEPropertyReadError: if any property returned ``NOT_FOUND``
                or ``NOT_SET`` (no values are parsed in that case).
            RTDEException: on protocol or connection errors.
        """
        cmd = Command.RTDE_READ_PROPERTIES
        payload = bytearray(",".join(names), "utf-8")
        result = self.__sendAndReceive(cmd, payload)
        if result is None:
            raise RTDEException(
                "RTDE_READ_PROPERTIES: no response or invalid property package"
            )
        types, values_data = result

        if len(types) != len(names):
            raise RTDEException(
                "RTDE_READ_PROPERTIES: expected {} types, got {}".format(
                    len(names), len(types)
                )
            )

        error_tokens = {}
        fmt = ">"
        for i, token in enumerate(types):
            wt = serialize.get_wire_type_format(token)
            if wt is None:
                error_tokens[names[i]] = token
            else:
                fmt_char, count = wt
                fmt += fmt_char * count
        if error_tokens:
            raise RTDEPropertyReadError(error_tokens)

        expected = struct.calcsize(fmt)
        if len(values_data) != expected:
            raise RTDEException(
                "RTDE_READ_PROPERTIES: values payload size mismatch "
                "(expected {} bytes, got {})".format(expected, len(values_data))
            )

        values = struct.unpack_from(fmt, values_data)
        props = {}
        vi = 0
        for i, wire_type in enumerate(types):
            size = serialize.get_item_size(wire_type)
            if size == 1:
                props[names[i]] = values[vi]
            else:
                props[names[i]] = list(values[vi : vi + size])
            vi += size
        return props

    def negotiate_protocol_version(self, version=Protocol.VERSION_2):
        cmd = Command.RTDE_REQUEST_PROTOCOL_VERSION
        payload = struct.pack(">H", version)
        success = self.__sendAndReceive(cmd, payload)
        if success:
            self.__protocolVersion = version
        return success

    def send_input_setup(self, variables, types=[]):
        cmd = Command.RTDE_CONTROL_PACKAGE_SETUP_INPUTS
        payload = bytearray(",".join(variables), "utf-8")
        result = self.__sendAndReceive(cmd, payload)
        if len(types) != 0 and not self.__list_equals(result.types, types):
            _log.error(
                "Data type inconsistency for input setup: "
                + str(types)
                + " - "
                + str(result.types)
            )
            return None
        result.names = variables
        self.__input_config[result.id] = result
        return serialize.DataObject.create_empty(variables, result.id)

    def send_output_setup(self, variables, types=[], frequency=125):
        cmd = Command.RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS
        payload = struct.pack(">d", frequency)
        payload = payload + (",".join(variables).encode("utf-8"))
        result = self.__sendAndReceive(cmd, payload)
        if len(types) != 0 and not self.__list_equals(result.types, types):
            _log.error(
                "Data type inconsistency for output setup: "
                + str(types)
                + " - "
                + str(result.types)
            )
            return False
        result.names = variables
        self.__output_config = result
        return True

    def send_start(self):
        cmd = Command.RTDE_CONTROL_PACKAGE_START
        success = self.__sendAndReceive(cmd)
        if success:
            _log.info("RTDE synchronization started")
            self.__conn_state = ConnectionState.STARTED
        else:
            _log.error("RTDE synchronization failed to start")
        return success

    def send_pause(self):
        cmd = Command.RTDE_CONTROL_PACKAGE_PAUSE
        success = self.__sendAndReceive(cmd)
        if success:
            _log.info("RTDE synchronization paused")
            self.__conn_state = ConnectionState.PAUSED
        else:
            _log.error("RTDE synchronization failed to pause")
        return success

    def send(self, input_data):
        if self.__conn_state != ConnectionState.STARTED:
            _log.error("Cannot send when RTDE synchronization is inactive")
            return
        if not input_data.recipe_id in self.__input_config:
            _log.error("Input configuration id not found: " + str(input_data.recipe_id))
            return
        config = self.__input_config[input_data.recipe_id]
        return self.__sendall(Command.RTDE_DATA_PACKAGE, config.pack(input_data))

    def receive(self, binary=False):
        """Receive the latest data package.
        If multiple packages has been received, older ones are discarded
        and only the newest one will be returned. Will block until a package
        is received or the connection is lost
        """
        if self.__output_config is None:
            raise RTDEException("Output configuration not initialized")
        if self.__conn_state != ConnectionState.STARTED:
            raise RTDEException("Cannot receive when RTDE synchronization is inactive")
        return self.__recv(Command.RTDE_DATA_PACKAGE, binary)

    def receive_buffered(self, binary=False, buffer_limit=None):
        """Receive the next data package.
        If multiple packages has been received they are buffered and will
        be returned on subsequent calls to this function.
        Returns None if no data is available.
        """

        if self._RTDE__output_config is None:
            logging.error("Output configuration not initialized")
            return None

        try:
            while (
                self.is_connected()
                and (buffer_limit == None or len(self.__buf) < buffer_limit)
                and self.__recv_to_buffer(0)
            ):
                pass
        except RTDEException as e:
            data = self.__recv_from_buffer(Command.RTDE_DATA_PACKAGE, binary)
            if data == None:
                raise e
        else:
            data = self.__recv_from_buffer(Command.RTDE_DATA_PACKAGE, binary)

        return data

    def send_message(
        self, message, source="Python Client", type=serialize.Message.INFO_MESSAGE
    ):
        cmd = Command.RTDE_TEXT_MESSAGE
        fmt = ">B%dsB%dsB" % (len(message), len(source))
        payload = struct.pack(fmt, len(message), message, len(source), source, type)
        return self.__sendall(cmd, payload)

    def __on_packet(self, cmd, payload):
        if cmd == Command.RTDE_REQUEST_PROTOCOL_VERSION:
            return self.__unpack_protocol_version_package(payload)
        elif cmd == Command.RTDE_GET_URCONTROL_VERSION:
            return self.__unpack_urcontrol_version_package(payload)
        elif cmd == Command.RTDE_TEXT_MESSAGE:
            return self.__unpack_text_message(payload)
        elif cmd == Command.RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS:
            return self.__unpack_setup_outputs_package(payload)
        elif cmd == Command.RTDE_CONTROL_PACKAGE_SETUP_INPUTS:
            return self.__unpack_setup_inputs_package(payload)
        elif cmd == Command.RTDE_CONTROL_PACKAGE_START:
            return self.__unpack_start_package(payload)
        elif cmd == Command.RTDE_CONTROL_PACKAGE_PAUSE:
            return self.__unpack_pause_package(payload)
        elif cmd == Command.RTDE_DATA_PACKAGE:
            return self.__unpack_data_package(payload, self.__output_config)
        elif cmd == Command.RTDE_READ_PROPERTIES:
            return self.__unpack_read_properties_package(payload)
        else:
            _log.error("Unknown package command: " + str(cmd))

    def __sendAndReceive(self, cmd, payload=b""):
        if self.__sendall(cmd, payload):
            return self.__recv(cmd)
        else:
            return None

    def __sendall(self, command, payload=b""):
        fmt = ">HB"
        size = struct.calcsize(fmt) + len(payload)
        buf = struct.pack(fmt, size, command) + payload

        if self.__sock is None:
            _log.error("Unable to send: not connected to Robot")
            return False

        _, writable, _ = select.select([], [self.__sock], [], DEFAULT_TIMEOUT)
        if len(writable):
            self.__sock.sendall(buf)
            return True
        else:
            self.__trigger_disconnected()
            return False

    def has_data(self):
        timeout = 0
        readable, _, _ = select.select([self.__sock], [], [], timeout)
        return len(readable) != 0

    def __recv(self, command, binary=False):

        previous_skipped_package_count = self.__skipped_package_count

        while self.is_connected():
            try:
                self.__recv_to_buffer(DEFAULT_TIMEOUT)
            except RTDETimeoutException:
                return None

            # unpack_from requires a buffer of at least 3 bytes
            while len(self.__buf) >= 3:
                # Attempts to extract a packet
                packet_header = serialize.ControlHeader.unpack(self.__buf)

                if len(self.__buf) >= packet_header.size:
                    packet, self.__buf = (
                        self.__buf[3 : packet_header.size],
                        self.__buf[packet_header.size :],
                    )
                    data = self.__on_packet(packet_header.command, packet)
                    if len(self.__buf) >= 3 and command == Command.RTDE_DATA_PACKAGE:
                        next_packet_header = serialize.ControlHeader.unpack(self.__buf)
                        if next_packet_header.command == command:
                            self.__skipped_package_count += 1
                            continue
                    if packet_header.command == command:
                        if (
                            self.__skipped_package_count
                            > previous_skipped_package_count
                        ):
                            _log.debug(
                                "Total number of skipped packages increased to {}".format(
                                    self.__skipped_package_count
                                )
                            )

                        if self.__warning_counter:
                            for warn in self.__warning_counter:
                                _log.debug(
                                    "A total of {} packets with command {} received, before expected command.".format(
                                        self.__warning_counter[warn], warn
                                    )
                                )
                            self.__warning_counter.clear()

                        if binary:
                            return packet[1:]

                        return data
                    else:
                        if not packet_header.command in self.__warning_counter:
                            _log.debug(
                                "Packet with command {} doesn't match the expected command {}. It will be skipped.".format(
                                    packet_header.command, command
                                )
                            )
                            self.__warning_counter[packet_header.command] = 1
                        else:
                            self.__warning_counter[packet_header.command] = (
                                self.__warning_counter[packet_header.command] + 1
                            )
                else:
                    break
        raise RTDEException(" _recv() Connection lost ")

    def __recv_to_buffer(self, timeout):
        readable, _, xlist = select.select([self.__sock], [], [self.__sock], timeout)
        if len(readable):
            more = self.__sock.recv(4096)
            # When the controller stops while the script is running
            if len(more) == 0:
                _log.error(
                    "received 0 bytes from Controller, probable cause: Controller has stopped"
                )
                self.__trigger_disconnected()
                raise RTDEException("received 0 bytes from Controller")

            self.__buf = self.__buf + more
            return True

        if (
            len(xlist) or len(readable) == 0
        ) and timeout != 0:  # Effectively a timeout of timeout seconds
            _log.warning("no data received in last %d seconds ", timeout)
            raise RTDETimeoutException("no data received within timeout")

        return False

    def __recv_from_buffer(self, command, binary=False):
        # unpack_from requires a buffer of at least 3 bytes
        while len(self.__buf) >= 3:
            # Attempts to extract a packet
            packet_header = serialize.ControlHeader.unpack(self.__buf)

            if len(self.__buf) >= packet_header.size:
                packet, self.__buf = (
                    self.__buf[3 : packet_header.size],
                    self.__buf[packet_header.size :],
                )
                data = self.__on_packet(packet_header.command, packet)
                if packet_header.command == command:
                    if binary:
                        return packet[1:]

                    return data
                else:
                    _log.debug("skipping package(2)")
            else:
                return None

    def __trigger_disconnected(self):
        _log.info("RTDE disconnected")
        self.disconnect()  # clean-up

    def __unpack_protocol_version_package(self, payload):
        if len(payload) != 1:
            _log.error("RTDE_REQUEST_PROTOCOL_VERSION: Wrong payload size")
            return None
        result = serialize.ReturnValue.unpack(payload)
        return result.success

    def __unpack_urcontrol_version_package(self, payload):
        if len(payload) != 16:
            _log.error("RTDE_GET_URCONTROL_VERSION: Wrong payload size")
            return None
        version = serialize.ControlVersion.unpack(payload)
        return version

    def __unpack_text_message(self, payload):
        if len(payload) < 1:
            _log.error("RTDE_TEXT_MESSAGE: No payload")
            return None
        if self.__protocolVersion == Protocol.VERSION_1:
            msg = serialize.MessageV1.unpack(payload)
        else:
            msg = serialize.Message.unpack(payload)

        if (
            msg.level == serialize.Message.EXCEPTION_MESSAGE
            or msg.level == serialize.Message.ERROR_MESSAGE
        ):
            _log.error(msg.source + ": " + msg.message)
        elif msg.level == serialize.Message.WARNING_MESSAGE:
            _log.warning(msg.source + ": " + msg.message)
        elif msg.level == serialize.Message.INFO_MESSAGE:
            _log.info(msg.source + ": " + msg.message)

    def __unpack_setup_outputs_package(self, payload):
        if len(payload) < 1:
            _log.error("RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS: No payload")
            return None
        output_config = serialize.DataConfig.unpack_recipe(payload)
        return output_config

    def __unpack_setup_inputs_package(self, payload):
        if len(payload) < 1:
            _log.error("RTDE_CONTROL_PACKAGE_SETUP_INPUTS: No payload")
            return None
        input_config = serialize.DataConfig.unpack_recipe(payload)
        return input_config

    def __unpack_start_package(self, payload):
        if len(payload) != 1:
            _log.error("RTDE_CONTROL_PACKAGE_START: Wrong payload size")
            return None
        result = serialize.ReturnValue.unpack(payload)
        return result.success

    def __unpack_pause_package(self, payload):
        if len(payload) != 1:
            _log.error("RTDE_CONTROL_PACKAGE_PAUSE: Wrong payload size")
            return None
        result = serialize.ReturnValue.unpack(payload)
        return result.success

    def __unpack_data_package(self, payload, output_config):
        if output_config is None:
            _log.error("RTDE_DATA_PACKAGE: Missing output configuration")
            return None
        output = output_config.unpack(payload)
        return output

    def __unpack_read_properties_package(self, payload):
        """Split an RTDE_READ_PROPERTIES payload into type tokens and value bytes.

        Layout: uint16 types_length, UTF-8 types string of that length
        (comma-separated wire types or error tokens), then packed values.
        """
        if len(payload) < 2:
            _log.error("RTDE_READ_PROPERTIES: Payload too short")
            return None
        (types_length,) = struct.unpack_from(">H", payload)
        types_end = 2 + types_length
        if len(payload) < types_end:
            _log.error(
                "RTDE_READ_PROPERTIES: Payload shorter than declared types length "
                "(expected at least {} bytes, got {} bytes)".format(
                    types_end, len(payload)
                )
            )
            return None
        types_str = payload[2:types_end].decode("utf-8")
        types = types_str.split(",")
        values_data = payload[types_end:]
        return types, values_data

    def __list_equals(self, l1, l2):
        if len(l1) != len(l2):
            return False
        for i in range(len((l1))):
            if l1[i] != l2[i]:
                return False
        return True

    @property
    def skipped_package_count(self):
        """The skipped package count, resets on connect"""
        return self.__skipped_package_count
