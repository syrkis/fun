# main.py
#     mnist model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
from torchvision.datasets import MNIST


# main
def main():
    # load data
    train_data = MNIST(root='data', train=True, download=True)
    test_data = MNIST(root='data', train=False, download=True)

    # create model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1)
    )

    # train model
    for epoch in range(10):
        for i, (x, y) in enumerate(train_data):
            break
        break


# run main
if __name__ == '__main__':
    main()

#%%

#%%
