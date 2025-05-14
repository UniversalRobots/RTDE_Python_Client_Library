

import numpy as np
import pandas as pd
#https://softwareengineering.stackexchange.com/questions/268128/how-do-deal-with-angle-wraparounds-when-comparing-them
df = pd.read_csv("../filePrepping/calibratedAlignedDatasets/calibrDataCentered06052025.csv")

#Normalization of roll from -pi to pi
df['robot_Roll'] = (df['robot_Roll'] + np.pi) % (2 * np.pi) - np.pi

df.to_csv("../filePrepping/calibratedAlignedDatasets/normalized14052025.csv", index=False)