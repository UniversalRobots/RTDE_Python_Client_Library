import pandas as pd

#sources:
#https://www.geeksforgeeks.org/use-pandas-to-calculate-stats-from-an-imported-csv-file/


df = pd.read_csv('../csv_data/standalone_arduino_data/hallEffect/SSdataWithAllSensors15042025.csv')


#csv file columns: timestamp,sensor1_x,sensor1_y,sensor1_z,sensor2_x,sensor2_y,sensor2_z,sensor3_x,sensor3_y,sensor3_z,ux,uy

#meanX =df['X'].mean()
#print('Mean Value of X: '+str(meanX))
def findMean(file,variableStr):
    mean = file[variableStr].mean()
    print(f'Mean Value of {variableStr} is: {mean}')
    return mean

def findStd(file,variableStr):
    std = file[variableStr].std()
    print(f'Mean Value of {variableStr} is: {std}')
    return std


def findVariance(file,variableStr):
    variance = file[variableStr].var()
    print(f'Mean Value of {variableStr} is: {variance}')
    return variance

sensorData = ['sensor1_x','sensor1_y', 'sensor1_z', 'sensor2_x', 'sensor2_y', 'sensor2_z', 'sensor3_x', 'sensor3_y', 'sensor3_z']

for sd in sensorData:
    findMean(file=df,variableStr=sd)
    df['sensor1_x'].variance()




