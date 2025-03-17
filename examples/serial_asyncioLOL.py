import asyncio
import serial
import csv
import time

class SerialReader(asyncio.Protocol):
    def __init__(self, csv_writer):
        self.csv_writer = csv_writer
        self.transport = None

    def connection_made(self, transport):
        """Callback when connection to Teensy is established"""
        self.transport = transport
        print("Connected to Teensy!")

    def data_received(self, data):
        """Callback when new data is received from Teensy"""
        line = data.decode().strip()  # Decode and clean the line
        print(f"Teensy Data: {line}")  # Debugging output

        # Expecting "value1,value2,value3" from Teensy
        values = line.split(",")
        if len(values) == 3:
            try:
                # Convert values to floats
                v1, v2, v3 = map(float, values)
                timestamp = time.time()

                # Write to CSV
                self.csv_writer.writerow([timestamp, v1, v2, v3])
            except ValueError:
                print(f"Invalid data format: {line}")

    def connection_lost(self, exc):
        print("Teensy connection lost!")

async def start_serial(csv_file_path, port="/dev/ttyACM0", baudrate=115200, serial_asyncio=None):
    """Starts the async serial reader"""
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["timestamp", "value1", "value2", "value3"])  # Header

        loop = asyncio.get_running_loop()
        transport, protocol = await serial_asyncio.create_serial_connection(
            loop, lambda: SerialReader(csv_writer), port, baudrate
        )
        await asyncio.sleep(3600)  # Keep running for an hour (adjust as needed)