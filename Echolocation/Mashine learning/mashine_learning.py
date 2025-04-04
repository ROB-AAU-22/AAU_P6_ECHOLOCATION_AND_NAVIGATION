import torch
from torch import nn

class Echo_model(nn.Module):
    def __init__(self):
        super().__init__()
        #2. Create 2 nn.linear layers capable of handling the shape of our data
        self.layer1=nn.Linear(in_features=2, out_features=5) # takes in 2 features and outputs 5 features
        self.layer2=nn.Linear(in_features=5, out_features=1) # takes in 5 features and outputs 1 features

      # 3. Define a forward() method that outlines the forward pass
    def forward(self,x):
      return self.layer2(self.layer1(x)) # x --> layer_1 --> layer_2 --> output