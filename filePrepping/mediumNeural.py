from torch import nn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torchview import draw_graph
from torchviz import make_dot


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


#sources
# https://medium.com/@piyushkashyap045/building-a-simple-neural-network-with-pytorch-42337d90a065
# https://medium.com/@gaurangmehra/master-non-linear-modeling-neural-networks-with-pytorch-dc1490d427be

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X, Y = load_data("alignedDatasets/aligned__25_04_2025_Dataset3.csv")


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

#https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
#train_size = int(0.8 * len(full_dataset))
#test_size = len(full_dataset) - train_size
#train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True)

#shuffle
#bool, default=True
#Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

#stratify
#array-like, default=None
#If not None, data is split in a stratified fashion, using this as the class labels. Read more in the User Guide.

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

    # Put the model in training mode at the beginning of an epoch
    model.train()


    yPredicted = model(XTrain)

    # 2. Calculate loss
    lossTrain = loss_fn(yPredicted, yTrain)

    # 3. Optimizer zero grad
    optimizer.zero_grad()


    lossTrain.backward()

    optimizer.step()

    model.eval()
    with torch.inference_mode():
        yPredTest = model(XTest)

        lossTest = loss_fn(yPredTest, yTest)

    if epoch % 50 == 0:
        print(f"epoch:{epoch} | lossTrain: {lossTrain} | lossTest: {lossTest}")

"""
dummy_input = torch.randn(1, 11, dtype=torch.float32).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "mappedModels/neural25__04__2025set3.onnx",
    opset_version=20,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)
"""


#How to vizualise: https://stackoverflow.com/questions/52468956/how-do-i-visualize-a-net-in-pytorch

#dummy_input = torch.randn(1, 11).to(device)
#output = model(dummy_input)
#dot = make_dot(output, params=dict(model.named_parameters()))
#dot.format = 'png'
#dot.render('model_architecture')

modelArchitecture = draw_graph(model, input_size=(1, 11), expand_nested=True, device=device)
modelArchitecture.visual_graph
#modelArchitecture.visual_graph.render('resnet18_architecture', format='png')
modelArchitecture.visual_graph.render('resnet18SVGarchitecture', format='svg')

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


test_predictions2 = predict(model, torch.from_numpy(np.array([[-1.19, -0.22, -14.42, 5.2, -5.51, -6.34, 3.37, 150.73, -11.19, 0, 0],
                                    [-1.16, -0.24, -14.45, 5.19, -5.45, -6.34, 3.36, 150.79, -11.2, 0, 0],
                                    [-1.19, -0.19, -14.42, 5.21, -5.49, -6.28, 3.38, 150.76, -11.13, 0, 0],
                                    [-1.16, -0.23, -14.44, 5.19, -5.47, -6.31, 3.37, 150.78, -11.16, 0, 0],
                                    [-1.17, -0.27, -14.38, 5.2, -5.46, -6.31, 3.38, 150.79, -11.16, 0, 0],
                                    [-1.18, -0.19, -14.47, 5.21, -5.47, -6.29, 3.38, 150.78, -11.15, 0, 0]
                                    ])).float().to(device)
                            )
print("test_predictions2:")
print(test_predictions2)


new_data = np.random.rand(5, 11)
new_predictions = predict(model, new_data)
print("new_predictions:")
print(new_predictions)
"""