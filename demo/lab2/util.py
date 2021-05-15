import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


def load_mnist(DATA_DIR):
    def dense_to_one_hot(y, class_count):
        return np.eye(class_count)[y]

    ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
    train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
    train_y = ds_train.targets.numpy()
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
    test_y = ds_test.targets.numpy()
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def load_cifar10(data_dir, val_split):
    train = CIFAR10(data_dir, train=True, transform=transforms.ToTensor(), download=True)

    lengths = [int(len(train) * val_split), len(train) - int(len(train) * val_split)]
    indices = torch.randperm(sum(lengths)).tolist()
    train_indices, valid_indices = indices[:lengths[0]], indices[lengths[0]:]

    _, (train, _) = next(enumerate(DataLoader(Subset(train, train_indices), batch_size=len(train_indices))))
    train_mean = train.mean((0, 2, 3))
    train_std = train.std((0, 2, 3))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(train_mean, train_std)])

    train = CIFAR10(data_dir, train=True, transform=transform, download=True)
    test = CIFAR10(data_dir, train=False, transform=transform)
    train, valid = Subset(train, train_indices), Subset(train, valid_indices)

    return train, test, valid
