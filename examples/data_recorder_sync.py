import asyncio
import argparse
import logging
import sys
import time
import csv
import serial_asyncio  # Async serial communication
sys.path.append("..")
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config


# ------------------------------
# ARGUMENT PARSER (Robot + CSV)
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="192.168.80.102", help="Robot IP")
parser.add_argument("--port", type=int, default=30004, help="Port number")
parser.add_argument("--frequency", type=int, default=125, help="Sampling frequency")
parser.add_argument("--config", default="record_configuration.xml", help="Config file")
parser.add_argument("--output_rtde", default="../csv_data/jointData.csv", help="Robot Output CSV")
parser.add_argument("--output_arduino", default="../csv_data/arduinoData.csv", help="Arduino Output CSV")
parser.add_argument("--serial_port", default="COM3", help="Arduino Serial Port")
parser.add_argument("--baud_rate", type=int, default=9600, help="Arduino Baud Rate")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)


# ------------------------------
# Communication Setup
# ------------------------------
conf = rtde_config.ConfigFile(args.config)
outputNames, outputTypes = conf.get_recipe("out")

con = rtde.RTDE(args.host, args.port)
con.connect()
con.get_controller_version()

if not con.send_output_setup(outputNames, outputTypes, frequency=args.frequency):
    logging.error("Unable to configure output")
    sys.exit()

if not con.send_start():
    logging.error("Unable to start synchronization")
    sys.exit()


# ------------------------------
# ASYNC RTDE DATA COLLECTION TASK
# ------------------------------
async def collect_rtde_data():
    with open(args.output_rtde, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["timestamp"] + outputNames
        writer.writerow(header)

        i = 1
        while True:
            try:
                state = con.receive()
                if state is not None:
                    timestamp = time.time()
                    row = [timestamp] + [getattr(state, name) for name in outputNames]
                    writer.writerow(row)
                    i += 1
                    await asyncio.sleep(1 / args.frequency)
            except rtde.RTDEException:
                con.disconnect()
                sys.exit()


# ------------------------------
# ASYNC ARDUINO SERIAL READING TASK
# ------------------------------
class SerialReaderProtocol(asyncio.Protocol):
    def __init__(self, csv_writer):
        self.csv_writer = csv_writer
        self.buffer = ""

    def data_received(self, data):
        self.buffer += data.decode()  # Decode incoming data

        if "\n" in self.buffer:  # If a complete line is received
            lines = self.buffer.split("\n")
            for line in lines[:-1]:  # Process full lines
                self.process_line(line.strip())
            self.buffer = lines[-1]  # Keep remaining incomplete data

    def process_line(self, line):
        parts = line.split(",")  # Will receive three sensor values from the arduino
        if len(parts) == 3:
            timestamp = time.time()
            row = [timestamp] + parts
            self.csv_writer.writerow(row)
            print(f"Arduino Data: {row}")


async def read_arduino():
    with open(args.output_arduino, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["timestamp", "sensor1", "sensor2", "sensor3"])

        loop = asyncio.get_running_loop()
        transport, protocol = await serial_asyncio.create_serial_connection(
            loop, lambda: SerialReaderProtocol(writer), args.serial_port, baudrate=args.baud_rate
        )

        try:
            await asyncio.Future()  # Keep running
        finally:
            transport.close()


# ------------------------------
# MAIN ASYNC FUNCTION
# ------------------------------
async def main():
    task1 = asyncio.create_task(collect_rtde_data())
    task2 = asyncio.create_task(read_arduino())       # Arduino Serial Reading
    await asyncio.gather(task1, task2)               # Run both concurrently


# ------------------------------
# RUN THE PROGRAM
# ------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        con.send_pause()
        con.disconnect()
