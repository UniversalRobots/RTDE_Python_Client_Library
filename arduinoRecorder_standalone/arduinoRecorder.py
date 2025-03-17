import asyncio
import argparse
import logging
import sys
import time
import csv
import serial

from serial_asyncio import create_serial_connection  # Explicit import
sys.path.append("..")
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

parser = argparse.ArgumentParser()

# ------------------------------
# Fill in CSV data file locations here
# ------------------------------
parser.add_argument("--output_arduino", default="../csv_data/standalone_arduino_data/arduinoTraining.csv", help="Arduino Output CSV")

#------------------------------------------
# Fill in arduino com serial port under here
#------------------------------------------
parser.add_argument("--serial_port", default="COM12", help="Arduino Serial Port")

parser.add_argument("--baud_rate", type=int, default=9600, help="Arduino Baud Rate")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

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
            self.buffer = lines[-1]

    def process_line(self, line):
        parts = line.split(",")
        if len(parts) == 11:
            timestamp = time.time()
            row = [timestamp] + parts
            self.csv_writer.writerow(row)
            print(f"Arduino Data: {row}")

def write_headers():
    with open(args.output_arduino, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "sensor1_x", "sensor1_y", "sensor1_z", "sensor2_x", "sensor2_y", "sensor2_z", "sensor3_x", "sensor3_y", "sensor3_z","ux", "uy"])

def read_arduino(x):
    data = str(x.readline().decode('utf-8')).rstrip()
    if data != '':
        with open(args.output_arduino, 'a', newline='') as csvfile:
            csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow([time.time(), data])

if __name__ == "__main__":
    com = "COM12"
    baud = 115200
    ardComm = serial.Serial(com, baud, timeout=0.1)
    write_headers()
    while ardComm.isOpen() == True:
        read_arduino(ardComm)

"""
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
            self.buffer = lines[-1]

    def process_line(self, line):
        parts = line.split(",")
        if len(parts) == 11:
            timestamp = time.time()
            row = [timestamp] + parts
            self.csv_writer.writerow(row)
            print(f"Arduino Data: {row}")


def write_headers():
    with open(args.output_arduino, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "sensor1_x", "sensor1_y"])
def read_arduino(x):
    data = str(x.readline().decode('utf-8')).rstrip()
    if data is not '':
        with open(args.output_arduino, 'w', newline='') as csvfile:
            csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow([int(data), str(time.asctime())])

# link: https://www.phippselectronics.com/storing-arduino-sensor-data-in-csv-format-using-python/?srsltid=AfmBOorLG0YMsk7oNXE3KX5G9UoT82TnlKzuZKL8k1aIpKA9yTStyuiG


"""
