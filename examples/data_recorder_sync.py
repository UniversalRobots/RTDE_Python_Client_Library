import asyncio
import argparse
import logging
import sys
import time
import csv
import serial_asyncio  # Async serial communication
from serial_asyncio import create_serial_connection  # Explicit import
from arduinoRecorder_standalone.arduinoRecorder import SerialReaderProtocol




sys.path.append("..")
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config



parser = argparse.ArgumentParser()
parser.add_argument("--host", default="10.0.12.245", help="Robot IP")
parser.add_argument("--port", type=int, default=30004, help="Port number")
parser.add_argument("--frequency", type=int, default=125, help="Sampling frequency")
parser.add_argument("--config", default="../xmlDataReader/record_configuration.xml", help="Config file")





#   Output files
parser.add_argument("--output_rtde", default="../csv_data/jointDatWOuP1404dataset3.csv", help="Robot Output CSV")
parser.add_argument("--output_arduino", default="../csv_data/arduinoDatWOuP1404dataset3.csv", help="Arduino Output CSV")

#------------------------------------------
# Fill inn arduino com serial port under here
#------------------------------------------
parser.add_argument("--serial_port", default="COM12", help="Arduino Serial Port")


parser.add_argument("--baud_rate", type=int, default=115200, help="Arduino Baud Rate")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)


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


async def collectRTDEData():
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




async def readArduino():
    with open(args.output_arduino, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "sensor1_x", "sensor1_y", "sensor1_z", "sensor2_x", "sensor2_y", "sensor2_z", "sensor3_x", "sensor3_y", "sensor3_z", "ux", "uy"])

        transport, protocol = await serial_asyncio.create_serial_connection(
            asyncio.get_event_loop(),
            lambda: SerialReaderProtocol(writer),
            args.serial_port,
            baudrate=args.baud_rate
        )

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            transport.close()


async def main():
    task1 = asyncio.create_task(collectRTDEData())
    task2 = asyncio.create_task(readArduino())
    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        con.send_pause()
        con.disconnect()
