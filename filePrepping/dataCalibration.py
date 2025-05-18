
import pandas as pd
import numpy as np

#sources:
#https://www.geeksforgeeks.org/use-pandas-to-calculate-stats-from-an-imported-csv-file/

import pandas as pd

df = pd.read_csv('calibratedAlignedDatasets/calibratedData06052025.csv')

variables = ['robot_X', 'robot_Y', 'robot_Z', 'robot_Pitch', 'robot_Yaw']

dfCalibrated = df.copy()
for variable in variables:
    dfCalibrated[variable] = df[variable] - df[variable].mean()

dfCalibrated['robot_Roll'] = (dfCalibrated['robot_Roll'] + np.pi) % (2 * np.pi) - np.pi

dfCalibrated.to_csv('calibratedAlignedDatasets/calibrDataCentered06052025.csv', index=False)
print(dfCalibrated)





