import pandas as pd

#sources:
#https://www.geeksforgeeks.org/use-pandas-to-calculate-stats-from-an-imported-csv-file/


df = pd.read_csv('../csv_data/standalone_arduino_data/xyzEuler/xyzEulerNeuralWithUxUy.csv')


#csv file columns: Timestamp,X,Y,Z,Roll,Pitch,ux,uy

#meanX =df['X'].mean()
#print('Mean Value of X: '+str(meanX))
def findMean(file,variableStr):
    mean = df[variableStr].mean()
    print(f'Mean Value of {variableStr} is: {mean}')
    return mean

variables = ['X', 'Y', 'Roll', 'Pitch']

for var in variables:
    findMean(file=df,variableStr=var)


dfCalibrated = df.copy()
dfCalibrated[variables] = df[variables] - df[variables].mean()

dfCalibrated.to_csv('meanCalibrated_file.csv', index=False)