from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import size
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Device setup (simplified)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, 1:12]  # Sensor inputs (columns 1-11)
    Y = data[:, 12:17]  # Coordinates (X,Y,Z,Roll,Pitch)
    return X, Y


X, Y = load_data("alignedDatasets/alignedData_interpolated0704.csv")

print("X:")
print(size(X))
print(f"with length: {len(X[0])}")

print("Y")
print(size(Y))
print(f"with length: {len(Y[0])}")


X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Model Definition (simplified with Sequential)
class MappingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, x):
        return self.net(x)


model = MappingNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop (fixed)
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            val_loss += criterion(outputs, batch_y).item()

    if epoch % 50 == 0:
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# Save model
torch.save(model.state_dict(), "mappedModels/mapped_model_sensorA0_and_A1.pth")

# ONNX Export (improved)
dummy_input = torch.randn(1, 11, dtype=torch.float32).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "mappedModels/inputOutput07042025.onnx",
    opset_version=20,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)