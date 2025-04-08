from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import size
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


#lowest yet: "1.2157888412475586"


#https://medium.com/@gaurangmehra/master-non-linear-modeling-neural-networks-with-pytorch-dc1490d427be

# Device setup (simplified)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, 1:12]  # Sensor inputs (columns 1-11)
    Y = data[:, 12:17]  # Coordinates (X,Y,Z,Roll,Pitch)
    return X, Y


X, Y = load_data("alignedDatasets/alignedInterpoaNoOutPut0704.csv")

for i in range(len(X)):
    X[i][-1] = 0.0
    X[i][-2] = 0.0


print("X:")
print(size(X))
print(f"with length: {len(X[0])}")

print("Y")
print(size(Y))
print(f"with length: {len(Y[0])}")


#X = (X - X.mean(axis=0)) / X.std(axis=0)
#Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)



additionalfactor = 1

# Model Definition (simplified with Sequential)
class MappingNN(nn.Module):
    def __init__(self, additionalfactor=1.0):
        super(MappingNN, self).__init__()
        self.additionalfactor = additionalfactor

        # Main pathway
        self.fc1 = nn.Linear(11, int(512 * additionalfactor))
        self.fc2 = nn.Linear(int(512 * additionalfactor), int(256 * additionalfactor))
        self.fc3 = nn.Linear(int(256 * additionalfactor), int(128 * additionalfactor))
        self.fc4 = nn.Linear(int(128 * additionalfactor), int(64 * additionalfactor))
        self.fc5 = nn.Linear(int(64 * additionalfactor), 5)

        # Skip connections (modified)
        self.skip1 = nn.Linear(11, int(64 * additionalfactor))  # From input
        self.skip2 = nn.Linear(int(128 * additionalfactor), int(64 * additionalfactor))  # Changed to match fc3 output

        # Normalization and regularization
        self.bn1 = nn.BatchNorm1d(int(512 * additionalfactor))
        self.bn2 = nn.BatchNorm1d(int(256 * additionalfactor))
        self.dropout = nn.Dropout(0.3)
        self.attention = nn.Linear(int(64 * additionalfactor), 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x_in = x.clone()

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.mish(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = self.fc3(x)
        x = nn.functional.selu(x)

        skip1 = self.skip1(x_in)
        skip2 = self.skip2(x)

        fused = skip1 + skip2
        attention_weights = torch.sigmoid(self.attention(fused))
        x = self.fc4(x) * attention_weights + fused

        return nn.functional.gelu(self.fc5(x))


""""# Initialize and Train
model = MappingNN().to(device)
X_train = torch.FloatTensor(X_train).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(5000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")


"""

# 1. Normalize your data (strongly recommended)
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
X = (X - X_mean) / (X_std + 1e-8)
Y = (Y - Y_mean) / (Y_std + 1e-8)

# 2. Verify shapes before training
#print("Input shape:", X_train.shape)
#print("Output shape:", Y_train.shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = MappingNN(additionalfactor=0.5).to(device)

criterion = nn.MSELoss()


optimizer = optim.Adam(
    model.parameters(),  # Pass model's learnable parameters
    lr=0.001,
    weight_decay=1e-4
)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=64)



for epoch in range(5000):
    model.train()
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()



dummy_input = torch.randn(1, 11, dtype=torch.float32).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "mappedModels/complexDataset08042025.onnx",
    opset_version=20,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)