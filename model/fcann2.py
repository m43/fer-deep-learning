import numpy as np
import sys
from tqdm import tqdm

from utils.util import softmax, relu
from utils.data import class_to_onehot


def fcann2_forward(x, w1, b1, w2, b2):
    s1 = x@w1+b1  # NxH
    h1 = relu(s1)  # NxH
    s2 = h1@w2 + b2  # NxC
    p = softmax(s2)  # NxC
    return s1, h1, s2, p


def fcann2_loss(Y_onehot_, probs, w1, w2, lambda_):
    N = Y_onehot_.shape[0]
    loss = -np.sum(Y_onehot_ * np.log(probs + sys.float_info.min),
                   axis=(-1, -2)) / N  # -LL
    loss += lambda_ * (np.sum(w1 ** 2, axis=(0, 1)) +
                       np.sum(w2 ** 2, axis=(0, 1)))  # L2 regularization
    return loss


def fcann2_train(X, Y_, H, eta, epochs, lambda_, log_interval=10, callback_fn=None):
    N = X.shape[0]
    D = X.shape[1]
    C = Y_.max()+1

    w1 = np.random.randn(D, H)
    b1 = np.zeros(H)
    w2 = np.random.randn(H, C)
    b2 = np.zeros(C)
    Y_onehot_ = class_to_onehot(Y_)

    for e in tqdm(range(epochs)):
        if callback_fn is not None and e == 0:
            callback_fn(0, w1, b1, w2, b2)

        s1, h1, s2, probs = fcann2_forward(X, w1, b1, w2, b2)
        loss = fcann2_loss(Y_onehot_, probs, w1, w2, lambda_)

        Gs2 = probs - Y_onehot_  # NxC
        grad_w2 = (Gs2.T @ h1).T / N + 2 * lambda_ * w2
        grad_b2 = np.sum(Gs2, axis=0) / N
        Gh1 = Gs2 @ w2.T
        Gs1 = Gh1 * np.where(s1 > 0, 1, 0)
        grad_w1 = (Gs1.T @ X).T / N + 2 * lambda_ * w1
        grad_b1 = np.sum(Gs1, axis=0) / N

        if e <= 15 or e % log_interval == 0:
            print(f"iteration: {e:5d} loss: {loss.item():2.8f}")

        w1 -= eta * grad_w1
        b1 -= eta * grad_b1
        w2 -= eta * grad_w2
        b2 -= eta * grad_b2

        if e <= 15 or e % log_interval == 0:
            if callback_fn is not None:
                callback_fn(0, w1, b1, w2, b2)

    return w1, b1, w2, b2, loss
