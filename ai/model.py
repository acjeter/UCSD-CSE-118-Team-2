import torch.nn as nn


# Define the model (MLP)
class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(66, 256),     
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 26)
        )
    
    def forward(self, x):
        return self.net(x)

model = ASLModel()

# MANUAL MODEL DEFINITION REFERENCE
# class my_nn(nn.Module):
#     def __init__(self):
#         super(my_nn, self).__init__()
#         self.layer1 = nn.Linear(784, 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = self.layer1(x)
#         x = torch.relu(x)
#         x = self.layer2(x)
#         x = torch.relu(x)
#         x = self.layer3(x)
#         return x