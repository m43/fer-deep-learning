import torch
import torch.optim as optim

# Hyperparameters
epochs = 500
eta = 0.1

## Definition of the computational graph
# data and parameters, init of parameters
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2, 3, 4, 3.5])
Y = torch.tensor([3, 5, 7, 9, 7.9])
N = X.shape[0]

# optimization procedure: gradient descent
optimizer = optim.SGD([a, b], lr=eta)

for i in range(epochs):
    # affine regression model
    Y_ = a * X + b
    diff = (Y - Y_)

    # square loss
    loss = torch.sum(diff ** 2) / N

    # compute the gradient
    loss.backward()
    a_grad_hand = torch.sum(-2 * (Y - Y_) * X) / N
    b_grad_hand = torch.sum(-2 * (Y - Y_)) / N

    # optimization step
    optimizer.step()

    if i <= 20 or i % 50 == 0:
        print(
            f"step: {i:06d}\n\tloss:{loss:e}\n\tY_:{Y_.detach().numpy()}\n\ta:{a.detach().numpy()}\n\tb:{b.detach().numpy()}")
        print(f"\tpytorch gradients:\n\t\ta:{a.grad}\n\t\tb:{b.grad}")
        print(f"\tgradients by ma hand:\n\t\ta:{a_grad_hand}\n\t\tb:{b_grad_hand}")

    # reset the gradients to zero
    optimizer.zero_grad()

print("Done")
