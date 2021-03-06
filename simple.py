import torch
from torch.autograd import Variable

import torch.nn as nn
from torch.nn import Sequential, Linear
import torch.nn.functional as F

import torch.optim as optim

from argparse import ArgumentParser

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import time, tqdm, util, pickle, math

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

class Nne(nn.Module):

    def __init__(self, num_features):
        super(Nne, self).__init__()

    def forward(self, x):
        return x

class BN(nn.Module):

    def __init__(self, num_features):
        super(BN, self).__init__()
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.norm(x)

class BNRelu(nn.Module):

    def __init__(self, num_features):
        super(BNRelu, self).__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(x))


class ReluBN(nn.Sequential):

    def __init__(self, num_features):
        super(ReluBN, self).__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.norm(self.relu(x))

class BNSigmoid(nn.Sequential):

    def __init__(self, num_features):
        super(BNSigmoid, self).__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.norm(x))

class SigmoidBN(nn.Sequential):

    def __init__(self, num_features):
        super(SigmoidBN, self).__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.norm(self.sigmoid(x))

def load_model(name, size=4, hidden=32, mult=0.0001):

    activation = None

    if name == 'relu':
        activation = Relu
    elif name == 'linear':
        activation = Nne
    elif name == 'linear-lambda':
        activation = Nne
    elif name == 'linear-bn':
        activation = BN
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

    model = Sequential(
        Linear(size, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, hidden),
        activation(hidden),
        Linear(hidden, size),
    )

    # shrink the weights to induc e vanishing gradient
    for layer in list(model.modules())[:10]:
        if isinstance(layer, Linear):
            size = layer.weight.size()
            torch.rand(size, out=layer.weight.data)
            layer.weight.data *= - mult

    return model

def loss_terms(model, input):
    losses = []

    hidden = input
    # for i, module in enumerate(list(model.modules())[1:]):
    #     hidden = module(hidden)
    #
    #     if isinstance(module, nn.ReLU) or isinstance(module, nn.Sigmoid):
    #         ll = layer_loss(hidden)
    #         losses.append(ll)

    for i, module in enumerate(list(model.modules())[1:]):
        hidden = module(hidden)

        if i in [1, 3, 5, 7, 9, 11]:
            ll = layer_loss(hidden)
            losses.append(ll)

    return losses

def layer_loss(hidden):
    b, d = hidden.size()

    mean = hidden.mean(dim=0, keepdim=True)

    hidden = hidden - mean

    diacov = torch.bmm(hidden.view(d, 1, b), hidden.view(d, b, 1)).squeeze() / (b - 1)

    logvar = torch.log(diacov)

    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return kl

def full_loss(hidden):

    b, d = hidden.size()

    mean = hidden.mean(dim=0, keepdim=True)

    diffs = hidden - mean

    cov = torch.mm(diffs.transpose(0, 1), diffs)
    cov = cov / (b - 1)

    t1 = torch.trace(cov)
    t2 = torch.dot(mean.squeeze(), mean.squeeze())


    if True:
        t3 = - torch.log( torch.potrf(cov).diag().prod()**2 )
    else:
        diacov = torch.bmm(diffs.view(d, 1, b), diffs.view(d, b, 1)).squeeze() / (b - 1)
        t3 = torch.log(diacov).sum()

    return 0.5 * (t1 + t2 + t3 + math.e - d)

def go(options):

    marker = itertools.cycle((',', '+', '.', 'o', '*'))

    SIZE = options.size
    EPOCHS = options.epochs
    BATCH_SIZE = options.batch_size
    TRAIN_SIZE = 60000 // BATCH_SIZE
    TEST_SIZE = 10000 // BATCH_SIZE

    CUDA = options.cuda

    # for modelname in ['relu', 'sigmoid', 'relu-lambda', 'sigmoid-lambda', 'relu-sigloss', 'sigmoid-sigloss', 'bn-relu', 'relu-bn', 'sigmoid-bn']:
    for modelname in ['relu-lambda', 'relu', 'relu-bn']: #, 'linear-bn']:

        print('testing model ', modelname)
        model = load_model(modelname, size=SIZE, mult=options.mult)
        print(util.count_params(model), ' parameters')

        if CUDA:
            model.cuda()

        criterion = nn.MSELoss(size_average=False)
        optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)

        accuracies = []
        results = {}

        for e in tqdm.trange(EPOCHS):
            for i in range(TRAIN_SIZE):

                x = torch.rand(BATCH_SIZE, SIZE)
                if CUDA:
                    x = x.cuda()
                x = Variable(x)
                x.requires_grad = False

                optimizer.zero_grad()

                y = model(x)

                loss = criterion(y, x)

                if 'lambda' in modelname:
                    lloss = sum(loss_terms(model, x))

                    # print(loss.data[0], lloss.data[0])
                    loss = loss + options.lambd * lloss

                if 'sigloss' in modelname:
                    lterms = loss_terms(model, x)
                    lloss = (1.0/len(lterms)) * sum([nn.functional.sigmoid(l) for l in lterms])
                    loss = loss + options.lambd * lloss

                loss.backward()
                optimizer.step()

                # w.add_scalar('normalization/mloss', mloss.data[0], i * BATCH_SIZE + e)
                # w.add_scalar('normalization/lloss', lloss.data[0], i * BATCH_SIZE + e)
                #
                # if i > 2:
                #     break

            sm = 0
            for i in range(TEST_SIZE):

                x = torch.rand(BATCH_SIZE, + SIZE)
                if CUDA:
                    x = x.cuda()
                x = Variable(x)
                x.requires_grad = False

                y = model(x)
                loss = criterion(y, x)
                # print(loss.data[0] / BATCH_SIZE )

                sm += float(loss.data[0])

                # if i > 2:
                #     break

            accuracies.append(sm / (TEST_SIZE * BATCH_SIZE))

        # accuracies = np.asarray(accuracies)
        plt.plot(accuracies, label=modelname, marker = next(marker))
        results[modelname] = list(accuracies)

    plt.title('lambda ' + str(options.lambd))
    plt.legend()
    plt.savefig('loss-curves.pdf')

    pickle.dump(results, open('results.pkl', 'wb'))

def go_learnrate(options):

    marker = itertools.cycle((',', '+', '.', 'o', '*'))

    SHAPE = options.size
    EPOCHS = options.epochs
    BATCH_SIZE = options.batch_size
    CUDA = options.cuda

    w = SummaryWriter()

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

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size of the input.",
                        default=128, type=int)

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

    parser.add_argument("-m", "--mult",
                        dest="mult",
                        help="Weight multiplier",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)
    if(options.learn_rate):
        go_learnrate(options)
    else:
        go(options)
