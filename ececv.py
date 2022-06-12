#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Enbo Zhou"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class AWGN(object):
    def __init__(self, sigma, mu = 0, vmin = 0, vmax = 1):
        self.mu = mu
        self.sigma = sigma
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, X):
        #gaussian noise
        noise = torch.from_numpy(np.random.normal(self.mu, self.sigma, X.shape))
        #synthesized image
        output = X + noise
        #clip values
        output[output < 0] = 0.0
        output[output > 1] = 1.0
        return output.float()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    conf_mat = np.zeros((10,10),int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            conf_mat += confusion_matrix(target, torch.reshape(pred, (-1,)))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # suppress: suppress scientific notation
    with np.printoptions(precision=3, suppress=True):
        print(np.array(conf_mat))


if __name__ == "__main__":
    # load data
    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_test1 = datasets.MNIST('../data', train=False, transform=transform1)

    # get the first image and display it
    img = dataset_test1.__getitem__(0)[0].numpy()
    img = np.transpose(img, [1, 2, 0])
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('first_image.png')
    plt.show()

    # test different gaussian noises
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(AWGN(0).__call__(torch.from_numpy(img)), cmap='gray')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_title('sigma=0')
    axs[0, 1].imshow(AWGN(0.3).__call__(torch.from_numpy(img)), cmap='gray')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_title('sigma=0.3')
    axs[1, 0].imshow(AWGN(0.6).__call__(torch.from_numpy(img)), cmap='gray')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_title('sigma=0.6')
    axs[1, 1].imshow(AWGN(1.0).__call__(torch.from_numpy(img)), cmap='gray')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_title('sigma=1')
    plt.savefig('gaussian_noises.png')
    plt.show()

    # set up some parameters
    use_cuda = False
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    test_kwargs = {'batch_size': 1000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        test_kwargs.update(cuda_kwargs)

    # get different test datasets
    transform_noise1 = transforms.Compose([
        transforms.ToTensor(),
        AWGN(sigma=0),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_noise2 = transforms.Compose([
        transforms.ToTensor(),
        AWGN(sigma=0.3),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_noise3 = transforms.Compose([
        transforms.ToTensor(),
        AWGN(sigma=0.6),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_noise4 = transforms.Compose([
        transforms.ToTensor(),
        AWGN(sigma=1.0),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset_test_noise1 = datasets.MNIST('../data', train=False, transform=transform_noise1)
    dataset_test_noise2 = datasets.MNIST('../data', train=False, transform=transform_noise2)
    dataset_test_noise3 = datasets.MNIST('../data', train=False, transform=transform_noise3)
    dataset_test_noise4 = datasets.MNIST('../data', train=False, transform=transform_noise4)

    test_loader_noise1 = torch.utils.data.DataLoader(dataset_test_noise1, **test_kwargs)
    test_loader_noise2 = torch.utils.data.DataLoader(dataset_test_noise2, **test_kwargs)
    test_loader_noise3 = torch.utils.data.DataLoader(dataset_test_noise3, **test_kwargs)
    test_loader_noise4 = torch.utils.data.DataLoader(dataset_test_noise4, **test_kwargs)

    # restore the trained model
    model = Net()
    model.load_state_dict(torch.load("./mnist_cnn.pt"))
    model = model.to(device)

    # test using different datasets
    test(model, device, test_loader_noise1)
    test(model, device, test_loader_noise2)
    test(model, device, test_loader_noise3)
    test(model, device, test_loader_noise4)