from torch import nn, relu
from torch.nn.functional import avg_pool2d


class Baseline(nn.Module):
    def __init__(self, in_dim=300, hid_dim_1=150, hid_dim_2=150):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim_1)
        self.fc2 = nn.Linear(hid_dim_1, hid_dim_2)
        self.fc3 = nn.Linear(hid_dim_2, 1)

    def forward(self, x, lengths=None):
        # it might make sense to exclude padding from average
        x = avg_pool2d(x, (x.shape[1], 1)).reshape((x.shape[0], x.shape[2]))

        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)
