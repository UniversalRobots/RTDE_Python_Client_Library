#https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network

from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.onnx
# Generate computation graph visualization
sample_input = torch.randn(1, 3)  # Match input dimensions
output = model(sample_input)
make_dot(output, params=dict(model.named_parameters())).render("nn_architecture", format="png")