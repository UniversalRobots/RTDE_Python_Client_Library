
import numpy as np
import pandas as pd

# Load and sort data
robotDf = pd.read_csv("../csv_data/xyzEulDS25_04_2025.csv").sort_values("timestamp")
arduinoDf = pd.read_csv("../csv_data/arduinoDatWOuP1404dataset3.csv").sort_values("timestamp")

robotTime = robotDf["timestamp"].values
arduinoTime = arduinoDf["timestamp"].values

#aligning timestaps, but should already be very close.
timeOffset = arduinoTime[0] - robotTime[0]
arduinoTimeAligned = arduinoTime - timeOffset

# Find, for each robot timestamp, the index of the nearest Arduino timestamp
idx = np.abs(arduinoTimeAligned[:, None] - robotTime).argmin(axis=0)

# Get unique indices to avoid duplicate Arduino samples
uniqueIdx = np.unique(idx)

# Keep only Arduino samples that are matched
matchedTeensy = arduinoDf.iloc[uniqueIdx].reset_index(drop=True)
matched_robot = robotDf.reset_index(drop=True)

# Optionally, you can also store the robot sample each Arduino sample is matched to
matched_robot = robotDf.iloc[np.arange(len(robotTime))].reset_index(drop=True)

# Save the aligned data
alignedData = pd.concat([
    matchedTeensy.reset_index(drop=True),
    matched_robot.reset_index(drop=True).add_prefix("robot_")
], axis=1)

alignedData.to_csv("alignedDatasets/aligned_unique_05_05_2025_Dataset3.csv", index=False)
print(f"Aligned {len(alignedData)} samples")


