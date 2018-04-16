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


def loss_function(out):
    b, d = out.size()

    mean = out.mean(dim=0, keepdim=True)

    out = out - mean

    diacov = torch.bmm(out.view(d, 1, b), out.view(d, b, 1)).squeeze() / (b - 1)

    logvar = torch.log(diacov)

    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return kl

def full_loss(out):

    b, d = out.size()

    mean = out.mean(dim=0, keepdim=True)

    diffs = out - mean

    cov = torch.mm(diffs.transpose(0, 1), diffs)
    cov = cov / (b - 1)

    t1 = torch.trace(cov)
    t2 = torch.dot(mean.squeeze(), mean.squeeze())

    t3 = - torch.log( torch.potrf(cov).diag().prod()**2 )

    return 0.5 * (t1 + t2 + t3 + math.e - d)

def go(options):

    CUDA = options.cuda
    SIZE = options.size
    HIDDEN = options.hidden
    PLOTN = options.plotn
    ITS = options.iterations
    BATCH = options.batch_size
    LR = options.learning_rate

    model = nn.Sequential(
        nn.Linear(SIZE, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 2),
    )

    if(CUDA):
        model.cuda()

    ## Train

    optimizer = optim.Adam(model.parameters(), lr=LR)

    for i in tqdm.trange(ITS):
        optimizer.zero_grad()
        x = torch.randn(BATCH, SIZE)

        if CUDA:
            x = x.cuda()

        x = Variable(x)

        y = model(x)

        loss = full_loss(y)

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(loss.data[0])

    ## Plot

    x = torch.randn(PLOTN, SIZE)

    if CUDA:
        x = x.cuda()

    x = Variable(x)

    # y = x.data.numpy()
    y = model(x).data.cpu().numpy()
    plt.figure(figsize=(10, 10))

    plt.scatter(y[:, 0], y[:, 1], alpha=0.01, linewidth=0)

    plt.savefig('scatter.png')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size of the input.",
                        default=2, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=256, type=int)

    parser.add_argument("-H", "--hidden",
                        dest="hidden",
                        help="The number of hidden units.",
                        default=6, type=int)

    parser.add_argument("-p", "--plot-n",
                        dest="plotn",
                        help="Number of points in the scatterplot",
                        default=100000, type=int)

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="The number of batches.",
                        default=100000, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-l", "--learning-rate",
                        dest="learning_rate",
                        help="The learning rate",
                        default=0.0001, type=float)


    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
