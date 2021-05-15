import os
import time

import numpy as np

from demo.lab2.util import load_mnist
from model import layers
from model import nn
from utils.util import project_path, ensure_dirs

# weight_decay = 1e-3
# weight_decay = 1e-2
weight_decay = 1e-1

DATA_DIR = os.path.join(project_path, 'datasets/MNIST')
SAVE_DIR = os.path.join(project_path, f'imgs/lab2/l2reg_wd={weight_decay}')
ensure_dirs([DATA_DIR, SAVE_DIR])
print(f"DATA_DIR:{DATA_DIR}\nSAVE_DIR:{SAVE_DIR}\n")

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = weight_decay
config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}

# np.random.seed(100)
np.random.seed(int(time.time() * 1e6) % 2 ** 31)

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_mnist(DATA_DIR)

weight_decay = config['weight_decay']
net = []
regularizers = []
inputs = np.random.randn(config['batch_size'], 1, 28, 28)
net += [layers.Convolution(inputs, 16, 5, "conv1")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'conv1_l2reg')]
net += [layers.MaxPooling(net[-1], "pool1")]
net += [layers.ReLU(net[-1], "relu1")]
net += [layers.Convolution(net[-1], 32, 5, "conv2")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'conv2_l2reg')]
net += [layers.MaxPooling(net[-1], "pool2")]
net += [layers.ReLU(net[-1], "relu2")]
## 7x7
net += [layers.Flatten(net[-1], "flatten3")]
net += [layers.FC(net[-1], 512, "fc3")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'fc3_l2reg')]
net += [layers.ReLU(net[-1], "relu3")]
net += [layers.FC(net[-1], 10, "logits")]

data_loss = layers.SoftmaxCrossEntropyWithLogits()
loss = layers.RegularizedLoss(data_loss, regularizers)

print(f"config:{config}")
nn.train(train_x, train_y, valid_x, valid_y, net, loss, config)
nn.evaluate("Test", test_x, test_y, net, loss, config)
