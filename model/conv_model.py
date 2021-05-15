import math

import torch.nn as nn


class SimpleConvolutionalModel(nn.Module):
    def __init__(self, in_channels, image_height, image_width, conv1_channels, conv2_channels, fc1_width, class_count):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        fc3_in = conv2_channels * math.floor(math.floor(image_width / 2) / 2) * math.floor(
            math.floor(image_height / 2) / 2)  # padding and stride of conv layers are hardcoded!
        self.fc3 = nn.Linear(fc3_in, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = h.relu()

        h = self.conv2(h)
        h = self.pool2(h)
        h = h.relu()

        h = h.view(h.shape[0], -1)
        h = self.fc3(h)
        h = h.relu()

        logits = self.fc_logits(h)
        return logits


class CifarSimpleConvolutionalModel(nn.Module):
    def __init__(self, in_channels=3, image_height=32, image_width=32, conv1_channels=16, conv2_channels=32,
                 fc1_width=256, fc2_width=128, class_count=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        fc1_in = conv2_channels * math.floor((math.floor((image_width - 3 + 2) / 2) - 3 + 2) / 2) * math.floor(
            (math.floor((image_width - 3 + 2) / 2) - 3 + 2) / 2)  # padding and stride of conv layers are hardcoded!
        self.fc1 = nn.Linear(fc1_in, fc1_width, bias=True)
        self.fc2 = nn.Linear(fc1_width, fc2_width, bias=True)
        self.fc_logits = nn.Linear(fc2_width, class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = h.relu()

        h = self.conv2(h)
        h = self.pool2(h)
        h = h.relu()

        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = h.relu()

        h = self.fc2(h)
        h = h.relu()

        logits = self.fc_logits(h)
        return logits
