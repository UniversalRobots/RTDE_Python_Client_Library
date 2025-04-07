

#Linear interpolation of datasets

#interpolation:
# interpolation does not require the data to have a linear relationship — but the method of interpolation you choose matters a lot.

#dont have the joint extension in the current dataset.


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Load data
robot_df = pd.read_csv("../csv_data/XYZEuler0404Data.csv").sort_values("timestamp")
arduino_df = pd.read_csv("../csv_data/arduinoDataInSync0404.csv").sort_values("timestamp")

robot_time = robot_df["timestamp"].values
arduino_time = arduino_df["timestamp"].values


time_offset = arduino_time[0] - robot_time[0]
arduino_time_aligned = arduino_time - time_offset

print(f"arduino time is {arduino_time[-1]-arduino_time[0]}")
print(f"robot time is {robot_time[-1]-robot_time[0]}")

"""
/*
# Verify alignment
print(f"Robot timestamps: {robot_time.min()} to {robot_time.max()}")
print(f"Aligned Arduino timestamps: {arduino_time_aligned.min()} to {arduino_time_aligned.max()}")

overlap = (arduino_time_aligned >= robot_time.min()) & (arduino_time_aligned <= robot_time.max())
if not np.any(overlap):
    print("ERROR: No overlap after alignment!")
    exit()

arduino_time_aligned = arduino_time_aligned[overlap]
arduino_df = arduino_df.iloc[overlap]

interpolated = {}
for col in ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]:
    interp_func = interp1d(robot_time, robot_df[col].values, kind='linear', fill_value="extrapolate")
    interpolated[col] = interp_func(arduino_time_aligned)

# Combine DataFrames
aligned_data = pd.DataFrame({
    "timestamp": arduino_time_aligned,
    **{col: arduino_df[col].values for col in arduino_df.columns if col != "timestamp"},
    **interpolated
})

aligned_data.to_csv("alignedDatasets/alingedData0404.csv", index=False)
print("Success! Output shape:", aligned_data.shape)
*/"""