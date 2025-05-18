
import numpy as np
import pandas as pd

#Nearest neighbout interpolation:
# https://www.geeksforgeeks.org/nearest-neighbor-interpolation-algorithm-in-matlab/


robotDf = pd.read_csv("../csv_data/xyzEulDS25_04_2025.csv").sort_values("timestamp")
arduinoDf = pd.read_csv("alignedDatasets/arduino_matched_to_robot.csv").sort_values("timestamp")

robotTime = robotDf["timestamp"].values
arduinoTime = arduinoDf["timestamp"].values

timeOffset = arduinoTime[0] - robotTime[0]
arduinoTimeAligned = arduinoTime - timeOffset

idx = np.searchsorted(arduinoTimeAligned, robotTime, side='left')
idx = np.clip(idx, 1, len(arduinoTimeAligned) - 1)

left = arduinoTimeAligned[idx - 1]
right = arduinoTimeAligned[idx]
idx -= robotTime - left < right - robotTime

uniqueIdx = np.unique(idx)
matchedTeensy = arduinoDf.iloc[uniqueIdx].reset_index(drop=True)
matchedRobot = robotDf.reset_index(drop=True)


matchedTeensy.to_csv("calibratedAlignedDatasets/calibrated06__05__2025.csv", index=False)
print(f"Kept {len(matchedTeensy)} Arduino samples (matched to robot timestamps)")


