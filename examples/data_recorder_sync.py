import argparse
import threading
import asyncio
import sys
import logging
import time
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import rtde.csv_writer as csv_writer
from examples.serial_asyncio import start_serial


# Your existing RTDE setup
parser = argparse.ArgumentParser()
#have to check this.
parser.add_argument("--host", default="192.168.80.102", help="Robot IP")
parser.add_argument("--port", type=int, default=30004, help="Port")
parser.add_argument("--frequency", type=int, default=125, help="Sampling rate")
parser.add_argument("--config", default="record_configuration.xml", help="Config file")
parser.add_argument("--output", default="../csv_data/jointData.csv", help="RTDE output CSV")
args = parser.parse_args()

conf = rtde_config.ConfigFile(args.config)
output_names, output_types = conf.get_recipe("out")

con = rtde.RTDE(args.host, args.port)
con.connect()
con.get_controller_version()

if not con.send_output_setup(output_names, output_types, frequency=args.frequency):
    logging.error("Unable to configure output")
    sys.exit()

if not con.send_start():
    logging.error("Unable to start synchronization")
    sys.exit()

writeModes = "w"
with open(args.output, writeModes) as csvfile:
    writer = csv_writer.CSVWriter(csvfile, ["timestamp"] + output_names, output_types)
    writer.writeheader()

    i = 1
    keep_running = True

    def teensy_thread():
        """Thread function for running asyncio event loop"""
        asyncio.run(start_serial("../csv_data/teensyData.csv", port="/dev/ttyACM0"))

    # Start Teensy reader in a separate thread
    teensy_thread = threading.Thread(target=teensy_thread, daemon=True)
    teensy_thread.start()

    while keep_running:
        if i % args.frequency == 0:
            sys.stdout.write(f"\r{i} samples recorded.")
            sys.stdout.flush()

        if args.samples > 0 and i >= args.samples:
            keep_running = False

        try:
            state = con.receive()
            if state is not None:
                timestamp = time.time()
                state_dict = vars(state)
                state_dict["timestamp"] = timestamp
                writer.writerow(state_dict)
                i += 1

        except rtde.RTDEException:
            con.disconnect()
            sys.exit()

    sys.stdout.write("\rComplete! \n")
    con.send_pause()
    con.disconnect()