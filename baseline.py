import torch
from torch.autograd import Variable

import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from argparse import ArgumentParser

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import time, tqdm, util, pickle

from util import Det, DetCuda, LogDetDiag
import itertools

class Relu(nn.Module):

    def __init__(self, num_features):
        super(Relu, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

class Sigmoid(nn.Module):

    def __init__(self, num_features):
        super(Sigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)

class BNRelu(nn.Module):

    def __init__(self, num_features):
        super(BNRelu, self).__init__()
        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(x))


class ReluBN(nn.Sequential):

    def __init__(self, num_features):
        super(ReluBN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.norm(self.relu(x))

class BNSigmoid(nn.Sequential):

    def __init__(self, num_features):
        super(BNSigmoid, self).__init__()
        self.norm = nn.BatchNorm2d(num_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.norm(x))

class SigmoidBN(nn.Sequential):

    def __init__(self, num_features):
        super(SigmoidBN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.norm(self.sigmoid(x))

def load_model(name, big=True):

    activation = None

    if name == 'relu':
        activation = Relu
    elif name == 'sigmoid':
        activation = Sigmoid
    elif name == 'relu-lambda':
        activation = Relu
    elif name == 'sigmoid-lambda':
        activation = Sigmoid
    elif name == 'relu-sigloss':
        activation = Relu
    elif name == 'sigmoid-sigloss':
        activation = Sigmoid
    elif name == 'bn-relu':
        activation = BNRelu
    elif name == 'relu-bn':
        activation = ReluBN
    elif name == 'bn-sigmoid':
        activation = BNSigmoid
    elif name == 'sigmoid-bn':
        activation = SigmoidBN
    else:
        raise Exception('Model "{}" not recognized.'.format(name))

    if big:
        model = Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),  # 0
            activation(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2), # 2
            activation(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2), # 4
            activation(16),                                                                   # 5
            nn.MaxPool2d(stride=2, kernel_size=2),                                          # 6
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2), # 7
            activation(32),                                                                   # 8
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2), # 9
            activation(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2), # 11
            activation(32),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2), # 14
            activation(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2), # 16
            activation(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2), # 18
            activation(64),
            nn.MaxPool2d(stride=2, kernel_size=2),
            util.Flatten(),
            nn.Linear(4 * 4 * 64, 10), # 22
            nn.Softmax()
        )
    else:
        model = Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),  # 0
            activation(8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),  # 2
            activation(8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),  # 2
            activation(8),
            nn.MaxPool2d(stride=2, kernel_size=2),  # 6
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),  # 7
            activation(16),  # 8
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),  # 9
            activation(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),  # 9
            activation(16),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 14
            activation(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # 16
            activation(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # 16
            activation(32),
            nn.MaxPool2d(stride=2, kernel_size=2),
            # util.Debug(lambda x: print(x.size())),
            util.Flatten(),
            nn.Linear(4 * 4 * 32, 10),  # 22
            nn.Softmax()
        )

    return model

def loss_terms(model, input):
    losses = []

    hidden = input
    for i, module in enumerate(list(model.modules())[1:]):
        hidden = module(hidden)

        # if isinstance(module, nn.ReLU) or isinstance(module, nn.Sigmoid):
        if isinstance(module, nn.Conv2d):
            ll = layer_loss(hidden)
            losses.append(ll)

    return losses


def layer_loss(hidden):
    b, _ = hidden.size()
    hidden = hidden.view(b, -1)
    b, d = hidden.size()

    mean = hidden.mean(dim=0, keepdim=True)

    hidden = hidden - mean

    diacov = torch.bmm(hidden.view(d, 1, b), hidden.view(d, b, 1)).squeeze() / (b - 1)

    logvar = torch.log(diacov)

    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return kl

def go(options):

    marker = itertools.cycle((',', '+', '.', 'o', '*'))

    EPOCHS = options.epochs
    BATCH_SIZE = options.batch_size
    CUDA = options.cuda

    # Set up the dataset

    normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train = torchvision.datasets.CIFAR10(root=options.data, train=True, download=True, transform=normalize)

    trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test = torchvision.datasets.CIFAR10(root=options.data, train=False, download=True, transform=normalize)

    testloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for modelname in ['relu', 'sigmoid', 'relu-lambda', 'sigmoid-lambda', 'relu-sigloss', 'sigmoid-sigloss', 'bn-relu', 'relu-bn', 'sigmoid-bn']:

        print('testing model ', modelname)
        model = load_model(modelname, False)
        print(util.count_params(model), ' parameters')

        if CUDA:
            model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)

        accuracies = []
        results = {}

        for e in tqdm.trange(EPOCHS):
            for i, data in enumerate(trainloader, 0):

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

                if 'lambda' in modelname:
                    lloss = sum(loss_terms(model, inputs))
                    loss = loss + options.lambd * lloss

                if 'sigloss' in modelname:
                    lterms = loss_terms(model, inputs)
                    lloss = (1.0/len(lterms)) * sum([nn.functional.sigmoid(l) for l in lterms])
                    loss = loss + options.lambd * lloss

                loss.backward()
                optimizer.step()

                # w.add_scalar('normalization/mloss', mloss.data[0], i * BATCH_SIZE + e)
                # w.add_scalar('normalization/lloss', lloss.data[0], i * BATCH_SIZE + e)
                #
                # if i > 2:
                #     break

            correct = 0
            total = 0
            for i, data in enumerate(testloader):

                inputs, labels = data

                if CUDA:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.data).sum()

                # if i > 2:
                #     break

            accuracies.append(correct / total)

        # accuracies = np.asarray(accuracies)
        plt.plot(accuracies, label=modelname, marker = next(marker))
        results[modelname] = list(accuracies)

    plt.title('lambda ' + str(options.lambd))
    plt.legend()
    plt.savefig('loss-curves.pdf')

    pickle.dump(results, open('results.pkl', 'wb'))

def go_learnrate(options):

    marker = itertools.cycle((',', '+', '.', 'o', '*'))

    EPOCHS = options.epochs
    BATCH_SIZE = options.batch_size
    CUDA = options.cuda

    w = SummaryWriter()

    # Set up the dataset

    normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train = torchvision.datasets.CIFAR10(root=options.data, train=True, download=True, transform=normalize)
    trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test = torchvision.datasets.CIFAR10(root=options.data, train=False, download=True, transform=normalize)
    testloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for learnrate in [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:

        print('testing learning rate ', learnrate)
        model = load_model('relu', False)
        print(util.count_params(model), ' parameters')

        if CUDA:
            model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learnrate)

        accuracies = []
        results = {}

        for e in tqdm.trange(EPOCHS):
            for i, data in enumerate(trainloader, 0):

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

                # w.add_scalar('normalization/mloss', mloss.data[0], i * BATCH_SIZE + e)
                # w.add_scalar('normalization/lloss', lloss.data[0], i * BATCH_SIZE + e)
                #
                # if i > 2:
                #     break

            correct = 0
            total = 0
            for i, data in enumerate(testloader):

                inputs, labels = data

                if CUDA:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.data).sum()

                # if i > 2:
                #     break

            accuracies.append(correct / total)

        # accuracies = np.asarray(accuracies)
        plt.plot(accuracies, label=str(learnrate), marker = next(marker))
        results[str(learnrate)] = list(accuracies)

    plt.title('learning rates ')
    plt.legend()
    plt.savefig('loss-curves-lr.pdf')

    pickle.dump(results, open('results-lr.pkl', 'wb'))

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=128, type=int)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="The number of epochs.",
                        default=75, type=int)

    parser.add_argument("-d", "--data",
                        dest="data",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-L", "--test-learn-rate", dest="learn_rate",
                        help="Run a learning rate experiment.",
                        action="store_true")

    parser.add_argument("-l", "--lambd",
                        dest="lambd",
                        help="The weight of the loss terms",
                        default=0.0001, type=float)

    parser.add_argument("-r", "--learning-rate",
                        dest="learning_rate",
                        help="The learning rate",
                        default=0.0001, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)
    if(options.learn_rate):
        go_learnrate(options)
    else:
        go(options)
