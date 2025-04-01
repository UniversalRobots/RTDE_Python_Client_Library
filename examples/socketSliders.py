import sys
import socket
import time
import numpy as np
import csv
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QPushButton
from PyQt6.QtCore import Qt

from examples.iksolver import URIKSolver

# DH-parametre
a = [0, -0.24365, -0.21325, 0, 0, 0]
d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]
alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
weights = [6, 5, 4, 3, 2, 1]

# Initialiser IK-solver
ikSolver = URIKSolver(a=a, d=d, alpha=alpha, w=weights)

HOST = "10.0.12.245"
PORT = 30002


class UR3Controller(QWidget):
    def __init__(self):
        super().__init__()
        # 114, -98, -105, -75, 90, 0
        self.joint_angles = [114, -98, -105, -75, 90, 0]

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.sliders = []
        self.labels = []

        for i in range(6):
            label = QLabel(f"Joint {i + 1}: {self.joint_angles[i]}°")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-180)
            slider.setMaximum(180)
            slider.setValue(self.joint_angles[i])
            slider.setTickInterval(1)
            slider.valueChanged.connect(self.updateValue)

            self.labels.append(label)
            self.sliders.append(slider)

            layout.addWidget(label)
            layout.addWidget(slider)

        self.save_button = QPushButton("Save setpoint to CSV")
        self.save_button.clicked.connect(self.saveToCsv)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.setWindowTitle("UR3 Joint Control")
        self.setGeometry(200, 200, 400, 300)

    def updateValue(self):
        self.joint_angles = [slider.value() for slider in self.sliders]
        for i, label in enumerate(self.labels):
            label.setText(f"Joint {i + 1}: {self.joint_angles[i]}°")

        self.sendRobotCommand()

    def sendRobotCommand(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT))
            print("Connected to robot!")
            #converting from numpy float to normal float
            JointRadians = [float(round(np.radians(angle), 3)) for angle in self.joint_angles]
            print(f"JointRadians = {JointRadians}")
            command = f"movej({JointRadians}, a=1.4, v=1.0)\n"
            s.send(command.encode('utf-8'))
            #print(f"Sent command: {command}")

            s.close()
        except socket.error as e:
            print(f"Connection error: {e}")

    def saveToCsv(self):
        filename = "../csv_data/jointPaths/jointPath1.csv"
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([float(round(np.radians(x), 3)) for x in self.joint_angles])
        print(f"Saved current joint angles to {filename}: {[float(round(np.radians(x), 3)) for x in self.joint_angles]}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = UR3Controller()
    controller.show()
    sys.exit(app.exec())
