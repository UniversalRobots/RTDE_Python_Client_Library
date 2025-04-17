

#Linear interpolation of datasets

#interpolation:
# interpolation does not require the data to have a linear relationship — but the method of interpolation you choose matters a lot.

#dont have the joint extension in the current dataset.

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Load and sort data
robot_df = pd.read_csv("../csv_data/xyzEulerWOuP1404dataset3.csv").sort_values("timestamp")
arduino_df = pd.read_csv("../csv_data/arduinoDatWOuP1404dataset3.csv").sort_values("timestamp")

robot_time = robot_df["timestamp"].values
arduino_time = arduino_df["timestamp"].values


print(f"robot data points  {len(robot_time)}")
print(f"arduino data points  {len(arduino_time)}")

time_offset = arduino_time[0] - robot_time[0]
arduino_time_aligned = arduino_time - time_offset

# Verify alignment range
print(f"Robot time range: {robot_time[0]:.3f} to {robot_time[-1]:.3f}")
print(f"Arduino time range: {arduino_time_aligned[0]:.3f} to {arduino_time_aligned[-1]:.3f}")

# Create interpolation functions for each robot dimension
interpolators = {
    col: interp1d(robot_time, robot_df[col].values,
                 kind='nearest',
                 bounds_error=False,
                 fill_value="extrapolate")  # Only if small mismatches exist
    for col in ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
}

# Interpolate all robot data at Arduino timestamps
interpolated_robot = {
    f"robot_{col}": interpolators[col](arduino_time_aligned)
    for col in interpolators
}

aligned_data = pd.DataFrame({
    "timestamp": arduino_time_aligned,
    **{f"arduino_{col}": arduino_df[col].values for col in arduino_df.columns if col != "timestamp"},
    **interpolated_robot
})

aligned_data.to_csv("alignedDatasets/alignedWithOP1404Dataset3.csv", index=False)
print(f"Success! Aligned {len(aligned_data)} samples")
print(f"Arduino/robot ratio: {len(aligned_data)/len(robot_df):.1f}x")


resultFile = pd.read_csv("alignedDatasets/alignedWithOP1404Dataset3.csv").sort_values("timestamp")


resultFileTimes = robot_df["timestamp"].values
print(f"resultFileTimes: {len(resultFileTimes)}")