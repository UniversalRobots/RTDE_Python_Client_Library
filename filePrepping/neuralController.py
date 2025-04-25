

#ANFIS
#Neural fuzzy controller:
# https://github.com/gregorLen/S-ANFIS-PyTorch
import torch.nn as nn

class NNController(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 64)  # 11 input features
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)   # 2 outputs: ux, uy

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        return self.fc3(x)
