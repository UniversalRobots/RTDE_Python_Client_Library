

#Linear interpolation of datasets

#interpolation:
# interpolation does not require the data to have a linear relationship — but the method of interpolation you choose matters a lot.

#dont have the joint extension in the current dataset.


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

robot_df = pd.read_csv("robot_data.csv")
robot_time = robot_df["timestamp"].values
robot_X = robot_df["X"].values
robot_Y = robot_df["Y"].values
robot_Z = robot_df["Z"].values
robot_Roll = robot_df["Roll"].values
robot_Pitch = robot_df["Pitch"].values
robot_Yaw = robot_df["Yaw"].values


#headers:
#sensor1_x,sensor1_y,sensor1_z,sensor2_x,sensor2_y,sensor2_z,sensor3_x,sensor3_y,sensor3_z,ux,uy
arduino_df = pd.read_csv("arduino_data.csv")
arduino_time = arduino_df["timestamp"].values
arduino_sensor1_x = arduino_df["sensor1_x"].values
arduino_sensor1_y = arduino_df["sensor1_y"].values
arduino_sensor1_z = arduino_df["sensor1_z"].values
arduino_sensor2_x = arduino_df["sensor2_x"].values
arduino_sensor2_y = arduino_df["sensor2_y"].values
arduino_sensor2_z = arduino_df["sensor2_z"].values
arduino_sensor3_x = arduino_df["sensor3_x"].values
arduino_sensor3_y = arduino_df["sensor3_y"].values
arduino_sensor3_z = arduino_df["sensor3_z"].values

arduino_sensor3_y = arduino_df["ux"].values
arduino_sensor3_z = arduino_df["sensor3_z"].values
# ... (other sensor columns)


interp_X = interp1d(robot_time, robot_X, kind='linear', fill_value='extrapolate')
interp_Y = interp1d(robot_time, robot_Y, kind='linear', fill_value='extrapolate')
interp_Z = interp1d(robot_time, robot_Z, kind='linear', fill_value='extrapolate')
interp_Roll = interp1d(robot_time, robot_Roll, kind='linear', fill_value='extrapolate')
interp_Pitch = interp1d(robot_time, robot_Pitch, kind='linear', fill_value='extrapolate')
interp_Yaw = interp1d(robot_time, robot_Yaw, kind='linear', fill_value='extrapolate')

X_interp = interp_X(arduino_time)
Y_interp = interp_Y(arduino_time)
Z_interp = interp_Z(arduino_time)
Roll_interp = interp_Roll(arduino_time)
Pitch_interp = interp_Pitch(arduino_time)
Yaw_interp = interp_Yaw(arduino_time)


aligned_data = pd.DataFrame({
    "timestamp": arduino_time,
    "sensor1_x": arduino_sensor1_x,
    "sensor1_y": arduino_sensor1_y,
    # ... (other sensor columns)
    "X": X_interp,
    "Y": Y_interp,
    "Z": Z_interp,
    "Roll": Roll_interp,
    "Pitch": Pitch_interp,
    "Yaw": Yaw_interp
})

aligned_data.to_csv("aligned_dataset.csv", index=False)