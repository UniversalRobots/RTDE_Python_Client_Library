
import numpy as np
import pandas as pd

robotDf = pd.read_csv("../csv_data/xyzEulDS25_04_2025.csv").sort_values("timestamp")
arduinoDf = pd.read_csv("alignedDatasets/arduino_matched_to_robot.csv").sort_values("timestamp")

robotTime = robotDf["timestamp"].values
arduinoTime = arduinoDf["timestamp"].values

# Align Arduino time if needed (optional)
timeOffset = arduinoTime[0] - robotTime[0]
arduinoTimeAligned = arduinoTime - timeOffset

# For each robot timestamp, find the index of the closest Arduino timestamp
idx = np.searchsorted(arduinoTimeAligned, robotTime, side='left')
idx = np.clip(idx, 1, len(arduinoTimeAligned) - 1)

left = arduinoTimeAligned[idx - 1]
right = arduinoTimeAligned[idx]
idx -= robotTime - left < right - robotTime

# Keep only unique Arduino indices (so each Arduino row is only kept if it is the closest to at least one robot timestamp)
uniqueIdx = np.unique(idx)
matchedTeensy = arduinoDf.iloc[uniqueIdx].reset_index(drop=True)
matched_robot = robotDf.reset_index(drop=True)


matchedTeensy.to_csv("calibratedAlignedDatasets/calibrated06__05__2025.csv", index=False)
print(f"Kept {len(matchedTeensy)} Arduino samples (matched to robot timestamps)")


