from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.onnx
import onnx
import numpy as np
import matplotlib.pyplot as plt

#importand note: "Latest PyTorch requires Python 3.9 or later."
print(torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU
else:
    device = torch.device('cpu')   # Use CPU

print(f"ONNX version: {onnx.__version__}")


#https://www.youtube.com/watch?v=Xp0LtPBcos0&t=1s
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
# Load your recorded data
data = np.loadtxt("../csv_data/standalone_arduino_data/arduinoTrainingCalm25032025.csv", delimiter=",", skiprows=1)  # Load CSV data
#The two last are coil output
nestInline = 3
X = data[:, [4+0, 5+0, 6+0, 9, 10]]
#number 1 is the clock
Y = data[:, [1, 2, 3]]
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
        self.fc1 = nn.Linear(5, 32)  # Input 5 -> Hidden 32
        self.fc2 = nn.Linear(32, 16) # Hidden 32 -> Hidden 16
        self.fc3 = nn.Linear(16, 3)  # Hidden 16 -> Output 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Initialize and Train
model = MappingNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
#can define ouself how many epochs we want right??
for epoch in range(500):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Save the model
#torch.save(model.state_dict(), "mapped_models/mapped_model_sensorA0_and_A1.pth")

#use double instead??
dummy_input = torch.randn(1, 5, dtype=torch.float32)



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
    "mapped_models_with_coils/CalmsSensrCoils_A1_and_A0_25032025.onnx",  # Output ONNX file name
    opset_version=20,
    #verbose=False,
    input_names=["input"],       # Input node name
    output_names=["output"],     # Output node name
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},  # Support variable batch size
)