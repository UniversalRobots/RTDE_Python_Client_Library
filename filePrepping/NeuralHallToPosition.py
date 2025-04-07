from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.onnx
import onnx
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


print("ONNX version:", onnx.__version__)  # Should output "1.15.0"
print("Protobuf version:", onnx.__version__)  # Should match 3.20.x



#importand note: "Latest PyTorch requires Python 3.9 or later."
print(torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU
else:
    device = torch.device('cpu')   # Use CPU



#https://www.youtube.com/watch?v=Xp0LtPBcos0&t=1s
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
#hiddel layers:
# "https://www.linkedin.com/pulse/choosing-number-hidden-layers-neurons-neural-networks-sachdev/"

data = np.loadtxt("alignedDatasets/alingedNewData0704.csv", delimiter=",", skiprows=1)  # Load CSV data
#The two last are coil output
nestInline = 3

#setup: "timestamp,sensor1_x,sensor1_y,sensor1_z,sensor2_x,sensor2_y,sensor2_z,sensor3_x,sensor3_y,sensor3_z,ux,uy,X,Y,Z,Roll,Pitch,Yaw"
X = data[:, [1,2,3,4,5,6,7,8,9,10,11]]
#number 1 is the clock
# X, Y, Z, Roll and Pitch
Y = data[:, [12,13,14,15,16]]
print(f"here is Input/Hall effect data: {X}")
print(f"here is Y/Real coordinates: {Y}")
print(Y)

print(X)
print("This was the first rows of data data")
print(Y)
print("This was the first rows of data data")

# Convert to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
Y_train = torch.tensor(Y, dtype=torch.float32)

# Define the Neural Network
class MappingNN(nn.Module):
    def __init__(self):
        super(MappingNN, self).__init__()
        self.fc1 = nn.Linear(11, 64)  # Input: 11 features
        self.bn1 = nn.BatchNorm1d(64)  # Batch norm
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 5)   # Output: 5 (X,Y,Z,Roll,Pitch)
        self.dropout = nn.Dropout(0.2)  # Optional: Regularization

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Optional
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
# Initialize and Train
model = MappingNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
#can define ouself how many epochs we want right??
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

best_loss = float('inf')
patience = 20
no_improve = 0

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_test)
        val_loss = criterion(val_output, Y_test)

    # Early stopping check
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item()}, Val Loss = {val_loss.item()}")

# Save the model
#torch.save(model.state_dict(), "mapped_models/mapped_model_sensorA0_and_A1.pth")

#use double instead??
#11 for 11 inputs
dummy_input = torch.randn(1, 11, dtype=torch.float32)



print("Layer fc1 Weights:\n", model.fc1.weight.data)
print("Layer fc1 Bias:\n", model.fc1.bias.data)

print("Layer fc2 Weights:\n", model.fc2.weight.data)
print("Layer fc2 Bias:\n", model.fc2.bias.data)

print("Layer fc3 Weights:\n", model.fc3.weight.data)
print("Layer fc3 Bias:\n", model.fc3.bias.data)

for name, param in model.named_parameters():
    print(f"{name}: {param.data}")


# Generate computation graph visualization
#sample_input = torch.randn(1, 3)  # Match input dimensions
#output = model(sample_input)
#make_dot(output, params=dict(model.named_parameters())).render("nn_architecture", format="png")


torch.onnx.export(
    model,                       # Trained model
    dummy_input,                 # Example input
    "mappedModels/HEToCoordinates.onnx",
    opset_version=20,
    #verbose=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},  # Support variable batch size
)