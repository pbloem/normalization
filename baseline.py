import torch
from torch.autograd import Variable

import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import torchsample as ts
from torchsample.modules import ModuleTrainer

from torchsample.metrics import *

from tensorboardX import SummaryWriter

from argparse import ArgumentParser

import time, tqdm, util

def load_model(name):

    activation = None

    if name == 'plain':
        activation = nn.ReLU()
    if name == 'plain':
        activation = nn.Sigmoid()
    elif name == 'bn-relu':
        activation = nn.Sequential(nn.BatchNorm2d(), nn.ReLU())
    elif name == 'relu-bn':
        activation = nn.Sequential(nn.ReLU(), nn.BatchNorm2d())
    elif name == 'bn-sigmoid-bn':
        activation = nn.Sequential(nn.BatchNorm2d(), nn.Sigmoid())
    elif name == 'sigmoid-bn':
        activation = nn.Sequential(nn.Sigmoid(), nn.BatchNorm2d())
    else:
        raise Exception('Model "{}" not recognized.'.format(name))


    model = Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
        activation,
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
        activation,
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
        activation,
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        activation,
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        activation,
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        activation,
        nn.MaxPool2d(stride=2, kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
        activation,
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        activation,
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        activation,
        nn.MaxPool2d(stride=2, kernel_size=2),
        util.Flatten(),
        nn.Linear(4 * 4 * 64, 10),
        nn.Softmax()
    )

    return model

def go(options):
    EPOCHS = options.epochs
    BATCH_SIZE = options.batch_size
    CUDA = options.cuda

    w = SummaryWriter()

    # Set up the dataset

    normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=normalize)

    trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=normalize)

    testloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = load_model('plain')

    if CUDA:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for e in range(EPOCHS):
        for i, data in tqdm.tqdm(enumerate(trainloader, 0)):

            # get the inputs
            inputs, labels = data

            if CUDA:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            w.add_scalar('normalization/loss', loss.data[0], i * BATCH_SIZE + e)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="The number of epochs.",
                        default=75, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-m", "-- odel",
                        dest="model",
                        help="Which model to use.",
                        default='plain', type=str)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
