from torch import nn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torchview import draw_graph
from torchviz import make_dot
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') #Tkinter don't work
import matplotlib.pyplot as plt
#https://medium.com/@benjybo7/unlocking-success-the-5-essential-metrics-you-must-track-in-neural-network-training-52dcb8874ff0

createOnnxFile = False
plotComparison = True
plotError = True
findIntgratedError = True
filePath= "../filePrepping/calibratedAlignedDatasets/calibrDataCentered06052025.csv"

instancesForIntError = 200


print("Filepath")
print(filePath)


def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, 1:12]
    Y = data[:, 12:17]
    return X, Y


def predict(model, input_data):

    model.eval()

    if not isinstance(input_data, torch.Tensor):
        input_data = torch.from_numpy(input_data).float()

    input_data = input_data.to(device)

    with torch.no_grad():
        predictions = model(input_data)

    return predictions.cpu().numpy()


def loopPlot(yPredicted,yReal, stateLabels, instances=30):  # The plotting: "https://stackoverflow.com/questions/35829961/using-matplotlib-with-tkinter-tkagg"
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for i in range(5):
        axes[i].plot(yReal[:, i], 'bo--', label=f'Real {stateLabels[i]}')
        axes[i].plot(yPredicted[:, i], 'r--', label=f'Predicted {stateLabels[i]}')
        axes[i].set_xlim(yReal.shape[0]-instances, yReal.shape[0])
        axes[i].set_xlabel('Iteration')
        if i < 3:
            axes[i].set_ylabel('Meters')
        else:
            axes[i].set_ylabel('Radians')
        axes[i].set_title(stateLabels[i])
        axes[i].legend()
    return fig, axes


def findIntegratedError(yPredicted,yReal, stateLabels, range1=200):  # The plotting: "https://stackoverflow.com/questions/35829961/using-matplotlib-with-tkinter-tkagg"
    for i in range(5):
        listOfErrors=[]
        for p in range(len(yPredicted)):
            listOfErrors.append((yReal[p, i]-yPredicted[p, i])**2)#, 'bo--', label=f'Error of {stateLabels[i]}')
        print(f'MeanSquarederror for test dataset of {len(yPredicted)} instances of {stateLabels[i]} is {sum(listOfErrors)/len(listOfErrors)}')
        listOfErrors.clear()




def loopPlotError(yPredicted,yReal, stateLabels, instances=30):  # The plotting: "https://stackoverflow.com/questions/35829961/using-matplotlib-with-tkinter-tkagg"
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for i in range(5):
        #Problably most appropiate to make two graphs but let's see
        #plot(x, y, color='green', marker='o', linestyle='dashed',
        #linewidth=2, markersize=12)
        #axes[i].scatter(yReal[:, i], yPredicted[:, i], alpha=0.5)
        axes[i].plot(yReal[:, i]-yPredicted[:, i], 'bo--', label=f'Error of {stateLabels[i]}')
        #axes[i].plot(, 'r--', label=f'Predicted {stateLabels[i]}')
        axes[i].set_xlim(yReal.shape[0]-instances, yReal.shape[0])
        axes[i].set_xlabel('Iteration')
        if i < 3:
            axes[i].set_ylabel('Meters')
        else:
            axes[i].set_ylabel('Radians')
        axes[i].set_title(f"NN prediction error of {stateLabels[i]}")
        axes[i].legend()
    return fig, axes

#sources
# https://medium.com/@piyushkashyap045/building-a-simple-neural-network-with-pytorch-42337d90a065
# https://medium.com/@gaurangmehra/master-non-linear-modeling-neural-networks-with-pytorch-dc1490d427be

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X, Y = load_data(filePath)


act = nn.ReLU()
print("X:")
print(X)
print("Y:")
print(Y)



#Definition of the network structure
class Model_comp(nn.Module):
    def __init__(self, additionalfactor=1): #, hiddenSize =
        super().__init__()
        self.addiFactor = additionalfactor

        #optional activation functions
        #could be benificial to use many different to capture the complexity of the system.
        # nn.Sigmoid()
        # nn.ReLU()
        # nn.SiLU()
        # nn.Tanh() # positivity: zero centered
         # torch.nn.Softmax(dim=-1)

        self.ll1 = nn.Linear(in_features=11, out_features=int(8 * additionalfactor))
        self.bn1 = nn.BatchNorm1d(int(8 * additionalfactor))  # BatchNorm added
        self.ac1 = nn.ReLU()

        # LSTM layer for recurrence
        #LSTM can capture long term dependencies
        #https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/
        #self.lstm = nn.LSTM(
        #    input_size=int(8 * additionalfactor),
        #    hidden_size=hiddenSize,
        #    batch_first=True
        #)

        self.ll2 = nn.Linear(in_features=int(8 * additionalfactor), out_features=int(32 * additionalfactor))
        self.bn2 = nn.BatchNorm1d(int(32 * additionalfactor))  # BatchNorm added
        #self.ac2 = nn.ReLU()
        # self.ac3 = nn.ReLU()
        self.ac2 = nn.Tanh()

        self.ll3 = nn.Linear(in_features=int(32 * additionalfactor), out_features=(16 * additionalfactor))
        self.bn3 = nn.BatchNorm1d(int(16 * additionalfactor))  # BatchNorm added
        #self.ac3 = nn.ReLU()
        self.ac3 = nn.Tanh()

        self.ll4 = nn.Linear(in_features=(16 * additionalfactor), out_features=(4 * additionalfactor))
        self.bn4 = nn.BatchNorm1d(int(4 * additionalfactor))  # BatchNorm added
        #self.ac4 = nn.ReLU()
        self.ac4 = nn.ReLU()

        # Output
        self.output = nn.Linear(in_features=(4 * additionalfactor), out_features=5)

    def forward(self, X):
        X = self.ll1(X)
        X = self.bn1(X)  # BatchNorm before activation
        X = self.ac1(X)

        X = self.ll2(X)
        X = self.bn2(X)  # BatchNorm before activation
        X = self.ac2(X)

        X = self.ll3(X)
        X = self.bn3(X)  # BatchNorm before activation
        X = self.ac3(X)

        X = self.ll4(X)
        X = self.bn4(X)  # BatchNorm before activation
        X = self.ac4(X)

        X = self.output(X)
        return X



XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True)


XTrain = torch.from_numpy(XTrain).float().to(device)
XTest = torch.from_numpy(XTest).float().to(device)
yTrain = torch.from_numpy(yTrain).float().to(device)
yTest = torch.from_numpy(yTest).float().to(device)

model = Model_comp(additionalfactor=16).to(device)
loss_fn = nn.L1Loss()

#ln is learning rate and is a very important perameter
optimizer = torch.optim.Adam(params = model.parameters(), lr =0.001)

epochs = 5000
for epoch in range(epochs):

    model.train()

    yPredicted = model(XTrain)

    lossTrain = loss_fn(yPredicted, yTrain)

    optimizer.zero_grad()


    lossTrain.backward()

    optimizer.step()

    model.eval()
    with torch.inference_mode():
        yPredTest = model(XTest)

        lossTest = loss_fn(yPredTest, yTest)

    if epoch % 50 == 0:
        print(f"epoch:{epoch} | lossTrain: {lossTrain} | lossTest: {lossTest}")





if (createOnnxFile):
    dummyInput = torch.randn(1, 11, dtype=torch.float32).to(device)
    torch.onnx.export(
        model,
        dummyInput,
        "mappedModels/neural25__04__2025set3.onnx",
        opset_version=20,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        }
    )



#How to vizualise: https://stackoverflow.com/questions/52468956/how-do-i-visualize-a-net-in-pytorch

#dummy_input = torch.randn(1, 11).to(device)
#output = model(dummy_input)
#dot = make_dot(output, params=dict(model.named_parameters()))
#dot.format = 'png'
#dot.render('model_architecture')

#modelArchitecture = draw_graph(model, input_size=(1, 11), expand_nested=True, device=device)
#modelArchitecture.visual_graph
#modelArchitecture.visual_graph.render('resnet18_architecture', format='png')
#modelArchitecture.visual_graph.render('resnet18SVGarchitecture', format='svg')

yReal = yTest.cpu().numpy()
#yPredicted = yPredTest.cpu().numpy()

yPredicted = predict(model, XTest.cpu().numpy())


print("MAE:", lossTest.item())
print("MSE:", mean_squared_error(yReal, yPredicted))
print("R2 Score:", r2_score(yReal, yPredicted))


print('yPredicted')
print(yPredicted)

print('yReal')
print(yReal)

print('yPredicted.shape()')
print(yPredicted.shape)

print('yReal')
print(yReal.shape)


#import matplotlib.pyplot as plt
#import numpy as np



#fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))
#Plots in for loops
#plots = zip(x, y)

newXList = np.zeros((5, 11201))
newYList = np.zeros((5, 11201))



#plots = zip(yPredicted, yReal)
stateLables = ["X", "Y", "Z", "Roll", "Pitch"]


if plotComparison:
    figs, axs = loopPlot(yPredicted, yReal, stateLables)
    plt.tight_layout()
    plt.show()

if findIntgratedError:
    findIntegratedError(yPredicted, yReal, stateLables, range1=instancesForIntError)

if plotError:
    figs, axs = loopPlotError(yPredicted, yReal, stateLables)
    plt.tight_layout()
    plt.show()






"""
For ReLU and TanH() in the middle
epoch:4800 | lossTrain: 0.010161289945244789 | lossTest: 0.009733160026371479
epoch:4850 | lossTrain: 0.01170132216066122 | lossTest: 0.01345023699104786
epoch:4900 | lossTrain: 0.009981871582567692 | lossTest: 0.011563187465071678
epoch:4950 | lossTrain: 0.00842907465994358 | lossTest: 0.009272655472159386
"""

"""
For only ReLU()
epoch:4800 | lossTrain: 0.007721226662397385 | lossTest: 0.009417756460607052
epoch:4850 | lossTrain: 0.009070700965821743 | lossTest: 0.01015558186918497
epoch:4900 | lossTrain: 0.007974457927048206 | lossTest: 0.00914800725877285
epoch:4950 | lossTrain: 0.008149172179400921 | lossTest: 0.009256409481167793

"""


"""
test_predictions = predict(model, X)
print("test_predictions:")
print(test_predictions)

"""
test_predictions2 = predict(model, torch.from_numpy(np.array([[-1.19, -0.22, -14.42, 5.2, -5.51, -6.34, 3.37, 150.73, -11.19, 0, 0],
                                    [-1.16, -0.24, -14.45, 5.19, -5.45, -6.34, 3.36, 150.79, -11.2, 0, 0],
                                    [-1.19, -0.19, -14.42, 5.21, -5.49, -6.28, 3.38, 150.76, -11.13, 0, 0],
                                    [-1.16, -0.23, -14.44, 5.19, -5.47, -6.31, 3.37, 150.78, -11.16, 0, 0],
                                    [-1.17, -0.27, -14.38, 5.2, -5.46, -6.31, 3.38, 150.79, -11.16, 0, 0],
                                    [-1.18, -0.19, -14.47, 5.21, -5.47, -6.29, 3.38, 150.78, -11.15, 0, 0]
                                    ])).float().to(device)
                            )

"""""
print("test_predictions2:")
print(test_predictions2)


new_data = np.random.rand(5, 11)
new_predictions = predict(model, new_data)
print("new_predictions:")
print(new_predictions)
"""