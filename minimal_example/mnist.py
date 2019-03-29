import os

import torch
import torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pycandle.general.experiment import Experiment
from pycandle.torch.model_trainer import ModelTrainer
from pycandle.torch.callbacks import *


class Net(nn.Module):
    """ Model to be traind. """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(800, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_datasets(batch_size_train, batch_size_test):
    """ Downloads, transforms and wraps datasets in DataLoaders. """
    transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize( (0.1307,), (0.3081,))])
    mnist_dataset_train = torchvision.datasets.MNIST('./mnist_data', train=True, download=True, transform=transformer)
    train_loader = torch.utils.data.DataLoader(mnist_dataset_train, batch_size=batch_size_train, shuffle=True)
    mnist_dataset_test = torchvision.datasets.MNIST('./mnist_data', train=False, download=True, transform=transformer)
    val_loader = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=batch_size_train, shuffle=True)
    return train_loader, val_loader

def accuracy(y_pred, batch):
    """ Prediction accuracy metric. """
    y_true = batch[1]
    pred = y_pred.data.max(1, keepdim=True)[1]
    count_correct = pred.eq(y_true.data.view_as(pred)).sum().double()
    return count_correct / y_pred.size(0)

def my_nll_loss(batch, model):
    """ Example for a loss in case 'custom_model_eval' is activated. """
    batch_x, batch_y = batch
    output = model(batch_x)
    return F.nll_loss(output, batch_y), output

class RunConfig:
    """ Training parameters. """
    epochs = 5
    learning_rate = 0.01

# core training code
experiment = Experiment('test_mnist', exclude_dirs=['mnist_data'])
train_loader, val_loader = load_datasets(batch_size_train=64, batch_size_test=1000)
model = Net().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=RunConfig.learning_rate)
model_trainer = ModelTrainer(model=model, optimizer=optimizer, loss=F.nll_loss, epochs=RunConfig.epochs, train_data_loader=train_loader, val_data_loader=val_loader, gpu=0)
model_trainer.set_metrics([accuracy])
history_recorder = HistoryRecorder()
model_trainer.add_callback(history_recorder)
model_trainer.add_callback(ModelCheckpoint(experiment.path))
model_trainer.start_training()

# plotting
for key in history_recorder.history.keys():
    if 'val' in key:
        continue
    plt.figure()
    plt.plot(history_recorder.history[key])
    val_key = 'val_' + key
    if val_key in history_recorder.history:
        plt.plot(history_recorder.history['val_' + key])
    plt.title('model ' + key)
    plt.ylabel(key)
    plt.xlabel('Epoch')
    if val_key in history_recorder.history:
        plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(experiment.plots, '{}.png'.format(key)))
    plt.close()