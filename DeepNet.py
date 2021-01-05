import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, output=4):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output)
        )

    def forward(self, x):
        x = self.features(x.reshape([-1, 4, 84, 84]))
        x = self.classifier(x.view(x.size(0), -1))
        return x

    def forward(self, x):
        x = self.features(x.reshape([-1, 4, 84, 84]))
        x = x.view(x.size(0), -1)

        v = self.value(x)
        adv = self.advantage(x)

        return v + (adv - adv.mean())




    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals
