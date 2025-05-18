# RTDE Client Library for the MagLev Project

This repository is a customized fork of the [Universal Robots RTDE Python Client Library](https://github.com/UniversalRobots/RTDE_Python_Client_Library), adapted for synchronized recording of UR robot and Teensy 4.1 microcontroller data on the MagLev 2.6 platform.

The datasets in then used for fining the relationship between hall effect data and robot position.

Developed as part of a bachelor project by Hans Kristian Urdahl and Simon Johannes Jensen.

---

## Table of Contents

- [Hardware Reference](#hardware-reference)
- [Features](#features)

- [Further Work](#further-work)
- [Acknowledgments](#acknowledgments)

---

## Hardware Reference

- **MagLev 2.6 PCB hardware files:**  
  [Hansolini/Take-home-Maglev-lab](https://github.com/Hansolini/Take-home-Maglev-lab/tree/main/physical_system/hardware/blueprints/maggyV2.6)  
  (Repository by NTNU employee Hans Alvar Engmark)

- **UR3 robot is used for data collection** 

---

## Features

### Main Functionality

- **Synchronized Data Collection** (`examples/data_recorder_sync.py`)
  - Simultaneously records hall effect sensor data (via serial communication) from the Teensy 4.1 and joint data from the UR robot.
  - The robot returns joint position data as specified in `xmlDataReader/record_configuration.xml` by the 'actual_q' setting, which provides a list of six joint positions.
  - Stores hall effect data and UR robot joint angle data in separate CSV files.
  - Denavit-Hartenberg matrix conversion to Cartesian coordinates (x, y, z, roll, pitch, yaw) is performed in a separate CMake repository.

- **UR Robot Control and Logging** (`examples/socketSliders.py`)
  - Provides interactive sliders for controlling each UR robot joint.
  - Saves joint positions and trajectories to CSV files for later playback using `examples/sendCSVData.py`.

- **Joint Command Execution / Playback from CSV File** (`examples/sendCSVData.py`)
  - Sends joint move commands to the UR robot via TCP sockets using the `movej` command.
  - Allows configurable speed and acceleration.
  - Can run concurrently with data collection (`examples/data_recorder_sync.py`), enabling collection of hall effect data while controlling the magnet over the sensors with the UR robot.
  - The collected data can later be used in `filePrepping/Neural.py` to determine the relationship between hall effect data and position data.

- **RTDE Library Enhancement** (`rtde/csv_writer.py`)
  - Extends the `writerow` function to support both list-based and attribute-based data objects.
  - All other library functions remain unchanged from the original fork.

### Analysis and Utilities

- **Data Analysis** (`csvAnalysis/`)
  - Scripts for normalizing and analyzing position, orientation, and hall effect sensor data.

- **Neural Network Construction** (`filePrepping/Neural.py`)
  - PyTorch-based neural network training and evaluation.
  - Supports exporting trained models to ONNX format for C++ and CUDA inference.
  - Includes visualization tools for evaluation, such as mean squared error plots and heatmaps for assessing neural network performance based on magnet position.

- **Data Interpolation** (`filePrepping/interpolation2.py`, `filePrepping/interpolation1.py`)
  - Aligns datasets with differing sample rates using nearest-neighbor and timestamp matching.

- **Angle Normalization** (`filePrepping/dataCalibration.py`)
  - Handles angle wrapping and normalization for UR robot end effector data.

---


---

## Further Work

- **Inverse Kinematics:**  
  - Development of inverse kinematics solutions (`examples/iksolver.py`, `examples/urx_control_loop.py`).
  - Based on [this MATLAB repository](https://github.com/JensOHI/IK_Solver_UR5) and adapted for Python.
  - Future improvements in this area will enable more structured robot movements and higher-quality datasets for neural network training.


- **Upgrade Robot:**
  - Upgrading from the UR3 to the e-Series UR3e can more than double the robot's data sampling rate.
    - The UR3 can deliver data at rates between 1 and 125 Hz, while e-Series UR robots can reach up to 500 Hz.
      - [Source: Universal Robots RTDE Guide](https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/)
    - Higher data rates can potentially improve neural network performance.

- **Record Data at the Highest Possible Time Resolution:**
  - Test using a time function that returns timestamps in microseconds or nanoseconds.
  

---

## Acknowledgments

- Based on the [Universal Robots RTDE Python Client Library](https://github.com/UniversalRobots/RTDE_Python_Client_Library).
- Neural network methodology inspired by [this Medium article](https://medium.com/@gaurangmehra/master-non-linear-modeling-neural-networks-with-pytorch-dc1490d427be).
- Additional sources of development and inspiration are referenced within the code files.
---




---
## GitHub project link (if public and available):
https://github.com/hanurd25/RTDEPythonClientLibraryMagLev
