
import pandas as pd



#sources:
#https://www.geeksforgeeks.org/use-pandas-to-calculate-stats-from-an-imported-csv-file/

#Remembter
df = pd.read_csv('../csv_data/standalone_arduino_data/xyzEuler/xyzEulerNeuralWUxUy25__04__2025.csv')
#C:/Users/hanur/Documents/UNIVERSITY/pythonProjects/RTDEPythonClientLibraryMagLev/csv_data/standalone_arduino_data/xyzEuler/xyzEulerNeuralWUxUy25__04__2025.csv
addDerivatives = True
itsMictoSecUnit = True
#csv file columns: Timestamp,X,Y,Z,Roll,Pitch,ux,uy

#meanX =df['X'].mean()
#print('Mean Value of X: '+str(meanX))
def findMean(file, variableStr):
    mean = file[variableStr].mean()
    print(f'Mean Value of {variableStr} is: {mean}')
    return mean


def findStd(file, variableStr):
    std = file[variableStr].std()
    print(f'Standard Deviation of {variableStr} is: {std}')
    return std


def findVariance(file, variableStr):
    variance = file[variableStr].var()
    print(f'Variance of {variableStr} is: {variance}')
    return variance

def findDelta(df, variable, deltaVar, timestamp):
    df[deltaVar] = df[variable].diff() / (df[timestamp] - df[timestamp][0]).diff().mean()
    print(f'{deltaVar}:')
    print(df[deltaVar].head())
    return df

variables = ['X', 'Y', 'Z', 'Roll', 'Pitch']
deltaVariablesNames = [('X', 'dx_dt'), ('Y', 'dy_dt'), ('Z', 'dz_dt'), ('Roll', 'dRoll_dt'), ('Pitch', 'dPitch_dt')]


cols = ['Timestamp', 'X', 'Y', 'Z', 'ux', 'uy']
df = df[(df[cols] != df[cols].shift()).any(axis=1)]



for var in variables:
    findMean(file=df,variableStr=var)
    findStd(file=df,variableStr=var)
    findVariance(file=df,variableStr=var)


dfCalibrated = df.copy()
dfCalibrated[variables] = df[variables] - df[variables].mean()

if (itsMictoSecUnit):
    dfCalibrated['Timestamp'] = df['Timestamp']/(10**6)
    for i in range(0, len(dfCalibrated['Timestamp'])):
        dfCalibrated['Timestamp'] = dfCalibrated['Timestamp'] - dfCalibrated.loc[0,'Timestamp']

dfCalibrated['dt'] = dfCalibrated['Timestamp'].diff()

print(dfCalibrated['dt'])

#fine tuning:
dfCalibrated['dt'].values[0] = 0



#Using the mean dt
meanDt = (dfCalibrated['Timestamp'] - dfCalibrated['Timestamp'][0]).diff().mean()
dfCalibrated['dt'] = meanDt

print('mean_dt', meanDt)
#print(mean_dt)
#print(dfCalibrated['dt'])


#0.0365 m says maggy book
#0.03
equlibrium = 0.0365
dfCalibrated['Z'] = dfCalibrated['Z'] + equlibrium
if(addDerivatives):
    #Does not work because of to small d_times
    for var, deltaVar in deltaVariablesNames:
        dfCalibrated = findDelta(dfCalibrated, var, deltaVar, 'Timestamp')


# List of columns to check
deltaCols = ['dx_dt', 'dy_dt', 'dz_dt', 'dRoll_dt', 'dPitch_dt']

# Remove rows where all these columns are zero (or NaN)
mask = ~(dfCalibrated[deltaCols].fillna(0) == 0).all(axis=1)
dfCalibrated = dfCalibrated[mask].reset_index(drop=True)


meanDt = (dfCalibrated['Timestamp'] - dfCalibrated['Timestamp'][0]).diff().mean()
dfCalibrated['dt'] = meanDt




if(addDerivatives):
    #Does not work because of to small d_times
    for var, deltaVar in deltaVariablesNames:
        dfCalibrated = findDelta(dfCalibrated, var, deltaVar, 'Timestamp')


for var in deltaCols:
    dfCalibrated.loc[0, var] = 0


print(dfCalibrated)

dfCalibrated.to_csv('../csv_data/standalone_arduino_data/xyzEuler/calibrated/calibrated25__04__2025.csv', index=False)








"""
Variance of Pitch is: 0.00011112055511005849
0       NaN
1       0.0
2       0.0
3       0.0
4       0.0
       ... 
4903    0.0
4904    0.0
4905    0.0
4906    0.0
4907    0.0
Name: dt, Length: 4908, dtype: float64
mean_dt 0.015691868758915834
dx_dt:
0         NaN
1   -0.045348
2    0.000000
3    0.387411
4   -1.149283
Name: dx_dt, dtype: float64
dy_dt:
0         NaN
1    0.688000
2    0.000000
3    0.032373
4   -0.761860
Name: dy_dt, dtype: float64
dz_dt:
0         NaN
1   -0.871598
2    0.000000
3   -0.129175
4    0.874402
Name: dz_dt, dtype: float64
dRoll_dt:
0           NaN
1   -371.381515
2      0.000000
3      0.035687
4     -0.525750
Name: dRoll_dt, dtype: float64
dPitch_dt:
0         NaN
1   -5.578431
2    0.000000
3   -0.224900
4    0.438584
Name: dPitch_dt, dtype: float64



"""
