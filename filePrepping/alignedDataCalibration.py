
import pandas as pd



#sources:
#https://www.geeksforgeeks.org/use-pandas-to-calculate-stats-from-an-imported-csv-file/

import pandas as pd

df = pd.read_csv('calibratedAlignedDatasets/calibratedData06052025.csv')

variables = ['robot_X', 'robot_Y', 'robot_Z', 'robot_Roll', 'robot_Pitch', 'robot_Yaw']

dfCalibrated = df.copy()
for variable in variables:
    dfCalibrated[variable] = df[variable] - df[variable].mean()

dfCalibrated.to_csv('calibratedAlignedDatasets/calibrDataCentered06052025.csv', index=False)
print(dfCalibrated)





