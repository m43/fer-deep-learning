import numpy as np
import torch
from torch.utils.data import DataLoader

dataset = [(torch.randn(4,4), torch.randint(5, size=())) for _ in range(25)]
dataset = [(x.numpy(), y.numpy()) for x,y in dataset]  # for demonstration purposes. the data is not copied
loader = DataLoader(dataset, batch_size=8, shuffle=False,
                    num_workers=0, collate_fn=None, drop_last=False)

for x,y in loader:
    print(x.shape, y.shape)

# def supervised_training_step(ctx, x, y):
#     ctx.model.train()
#     output = ctx.model(x)
#     loss = ctx.loss(output, y).mean()

#     ctx.optimizer.zero_grad()
#     loss.backward()
#     ctx.optimizer.step()

# from torch.optim import SGD
# optimizer = SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
