import csv
import time
import socket
import pandas as pd

def sendRobotJointData(host, port, csvFilePath, velocity, acceleration, delay):
    #list of 6 spots
    angleRow = [0] * 6
    print(f"This is the list of angles: {angleRow}")
    # Read the CSV file
    with open(csvFilePath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                print(f"Connected to {host}:{port}")

                for row in reader:
                    #source: "https://stackoverflow.com/questions/74641071/how-do-i-extract-specific-rows-from-a-csv-file"
                    if len(row) == 6:
                        print(row)
                        print(row[1])
                        #joint_data = ",".join(row) + "\n"
                        for i in range(len(row)):
                            print(f"iteration {i + 1}")
                            angleRow[i] = float(row[i])


                        command = f"movej({angleRow}, a={acceleration}, v={velocity})\n"
                        print("Sending command: ", command)
                        print("Now sending command: ", command)
                        s.send(command.encode('utf-8'))

                        #time.sleep(2)
                        time.sleep(delay)

            except socket.error as e:
                print(f"Socket error: {e}")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    robotIp = "10.0.12.245"
    port = 30003
    CSVFILE = "../csv_data/jointPaths/jointPathMagLev28.csv"

    sendRobotJointData(robotIp, port, CSVFILE, velocity=1.0, acceleration=1.4, delay=2)

    sendRobotJointData(robotIp, port, CSVFILE, velocity=0.3, acceleration=0.7, delay=4)

    sendRobotJointData(robotIp, port, CSVFILE, velocity=0.3, acceleration=0.7, delay=6)