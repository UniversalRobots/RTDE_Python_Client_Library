from torch import nn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, 1:12]
    Y = data[:, 12:17]
    return X, Y


def predict(model, input_data):
    """
    Make predictions using the trained PyTorch model.

    Args:
        model: Trained PyTorch model
        input_data: Input data (numpy array or torch tensor)

    Returns:
        Predictions as numpy array
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Convert input to tensor if needed
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.from_numpy(input_data).float()

    # Move to same device as model
    input_data = input_data.to(device)

    # Make prediction with gradient tracking disabled
    with torch.no_grad():
        predictions = model(input_data)

    return predictions.cpu().numpy()


#sources
# https://medium.com/@piyushkashyap045/building-a-simple-neural-network-with-pytorch-42337d90a065
# https://medium.com/@gaurangmehra/master-non-linear-modeling-neural-networks-with-pytorch-dc1490d427be

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X, Y = load_data("alignedDatasets/alignedWithOP1404Dataset3.csv")


act = nn.ReLU()
print("X:")
print(X)
print("Y:")
print(Y)



#Definition of the network structure
class Model_comp(nn.Module):
    def __init__(self, additionalfactor=1):
        super().__init__()
        self.addiFactor = additionalfactor

        # Layer 1
        self.ll1 = nn.Linear(in_features=11, out_features=int(8 * additionalfactor))
        self.bn1 = nn.BatchNorm1d(int(8 * additionalfactor))  # BatchNorm added
        self.ac1 = nn.ReLU()

        # Layer 2
        self.ll2 = nn.Linear(in_features=int(8 * additionalfactor), out_features=int(32 * additionalfactor))
        self.bn2 = nn.BatchNorm1d(int(32 * additionalfactor))  # BatchNorm added
        self.ac2 = nn.ReLU()

        # Layer 3
        self.ll3 = nn.Linear(in_features=int(32 * additionalfactor), out_features=(16 * additionalfactor))
        self.bn3 = nn.BatchNorm1d(int(16 * additionalfactor))  # BatchNorm added
        self.ac3 = nn.ReLU()

        # Layer 4
        self.ll4 = nn.Linear(in_features=(16 * additionalfactor), out_features=(4 * additionalfactor))
        self.bn4 = nn.BatchNorm1d(int(4 * additionalfactor))  # BatchNorm added
        self.ac4 = nn.ReLU()

        # Output Layer (no BatchNorm here)
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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True)

#shuffle
#bool, default=True
#Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

#stratify
#array-like, default=None
#If not None, data is split in a stratified fashion, using this as the class labels. Read more in the User Guide.

X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

model = Model_comp(additionalfactor=16).to(device)
loss_fn = nn.L1Loss()

#ln is learning rate and is a very important perameter
optimizer = torch.optim.Adam(params = model.parameters(), lr =0.001)

epochs = 5000
for epoch in range(epochs):

    # Put the model in training mode at the beginning of an epoch
    model.train()


    y_pred = model(X_train)

    # 2. Calculate loss
    loss_train = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Backward pass
    loss_train.backward()

    # 5 Optimizer step
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_pred_test = model(X_test)

        loss_test = loss_fn(y_pred_test, y_test)

    if epoch % 50 == 0:
        print(f"epoch:{epoch} | loss_train: {loss_train} | loss_test: {loss_test}")


dummy_input = torch.randn(1, 11, dtype=torch.float32).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "mappedModels/neuralWithAllOutput1404set3.onnx",
    opset_version=20,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)
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