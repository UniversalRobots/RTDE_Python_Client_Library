

#Linear interpolation of datasets

#interpolation:
# interpolation does not require the data to have a linear relationship — but the method of interpolation you choose matters a lot.

#dont have the joint extension in the current dataset.

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Load and sort data
robotDf = pd.read_csv("../csv_data/xyzEulDS25_04_2025.csv").sort_values("timestamp")
arduinoDf = pd.read_csv("../filePrepping/alignedDatasets/arduino_matched_to_robot.csv").sort_values("timestamp")
#csv_data/arduinoDatWOuP1404dataset2.csv
#robotDf = pd.read_csv("../csv_data/jointDatWOuP1404dataset2.csv").sort_values("timestamp")
#arduinoDf = pd.read_csv("../csv_data/arduinoDatWOuP1404dataset2.csv").sort_values("timestamp")

robotTime = robotDf["timestamp"].values
arduinoTime = arduinoDf["timestamp"].values


print(f"robot data points  {len(robotTime)}")
print(f"arduino data points  {len(arduinoTime)}")

timeOffset = arduinoTime[0] - robotTime[0]
arduinoTimeAligned = arduinoTime - timeOffset

print(f"Robot time range: {robotTime[0]:.3f} to {robotTime[-1]:.3f}")
print(f"Arduino time range: {arduinoTimeAligned[0]:.3f} to {arduinoTimeAligned[-1]:.3f}")

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
interpolators = {
    col: interp1d(robotTime, robotDf[col].values,
                 kind='nearest',
                 bounds_error=False,
                 fill_value="extrapolate")
    for col in ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
}

interpolatedRobotData = {
    f"robot_{col}": interpolators[col](arduinoTimeAligned)
    for col in interpolators
}

alignedData = pd.DataFrame({
    "timestamp": arduinoTimeAligned,
    **{f"arduino_{col}": arduinoDf[col].values for col in arduinoDf.columns if col != "timestamp"},
    **interpolatedRobotData
})

alignedData.to_csv("calibratedAlignedDatasets/calibratedData06052025.csv", index=False)
print(f"Aligned {len(alignedData)} samples")
print(f"Arduino/robot ratio: {len(alignedData)/len(robotDf):.1f}x")


#Output:
#robot data points  33066
#arduino data points  112003
#Robot time range: 1744669586.316 to 1744670103.181
#Arduino time range: 1744669586.316 to 1744670103.174
#Aligned 112003 samples
#Arduino/robot ratio: 3.4x""


#resultFile = pd.read_csv("alignedDatasets/alignedWithOP1404Dataset3.csv").sort_values("timestamp")


#resultFileTimes = robotDf["timestamp"].values
#print(f"resultFileTimes: {len(resultFileTimes)}")