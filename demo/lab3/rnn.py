from torch import nn, relu, vstack


class MaRNN(nn.Module):
    def __init__(self, rnn_class, in_dim=300, hid_dim_1=150, hid_dim_2=150, num_layers=2, dropout=0):
        super().__init__()
        self.rnn = rnn_class(input_size=in_dim, hidden_size=hid_dim_1, num_layers=num_layers, dropout=dropout)
        self.fc1 = nn.Linear(hid_dim_1, hid_dim_2)
        self.fc2 = nn.Linear(hid_dim_2, 1)

    def forward(self, x, lengths):
        x, _ = self.rnn(x)
        # Better off using torch.nn.utils.rnn.pack_padded_sequence instead of lengths and vstack
        x = vstack([x[lengths[i] - 1, i] for i in range(x.shape[1])])
        x = relu(self.fc1(x))
        return self.fc2(x)
