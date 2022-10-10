# model.py
#     model for go
# by: Noah Syrkis

# imports
import torch
from torch import nn


# model class for playing go
class Model(nn.Module):
    def __init__(self, input_space, output_space):
        super(Model, self).__init__()
        print("Input Space:", input_space['observation'].shape)
        print("Output Space:", output_space.n)
        exit()

        self.input_space = input_space.shape
        self.output_space = output_space.n

        self.cnn1 = nn.Conv2d(self.input_space[2], 32, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cnn3 = nn.Conv2d(64, 128, kernel_size=5, stride=3)
        self.cnn4 = nn.Conv2d(128, 256, kernel_size=6, stride=4)

        self.fc1 = nn.Linear(256 * 7 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.output_space)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x / 255
        x = x.reshape(self.input_space[2], self.input_space[0], self.input_space[1])
        x = nn.functional.relu(self.cnn1(x))
        x = nn.functional.relu(self.cnn2(x))
        x = nn.functional.relu(self.cnn3(x))
        x = nn.functional.relu(self.cnn4(x))
        x = x.view(-1, 256 * 7 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def fit(self, x, y):
        loss = nn.functional.mse_loss(self.forward(x), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
