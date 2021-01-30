import numpy as np
import sys
from tqdm import tqdm

from utils.util import softmax
from utils.data import class_to_onehot

def logreg_get_scores(X, w, b):
    scores = X @ w + b
    return scores


def logreg_get_probs(X, w, b, scores=None):
    if scores is None:
        scores = logreg_get_scores(X, w, b)
    probs = softmax(scores)
    return probs


def logreg_loss(Y_onehot_, probs, w, lambda_):
    N = Y_onehot_.shape[0]
    loss = -np.sum(Y_onehot_ * np.log(probs + sys.float_info.min), axis=(-1,-2)) / N
    loss += lambda_ * np.sum(w ** 2, axis=(0,1))  # L2 regularization
    return loss


def logreg_train(X, Y_, eta, epochs, lambda_, log_interval=10, log_grad_check=False, grad_check_epsilon=1e-5,
                    callback_fn=None):
    N = X.shape[0]
    D = X.shape[1]
    C = Y_.max()+1
    w = np.random.randn(D,C)
    b = np.zeros(C)
    Y_onehot_ = class_to_onehot(Y_)

    for e in tqdm(range(epochs)):
        if callback_fn is not None and e == 0:
            callback_fn(0, w, b)

        probs = logreg_get_probs(X, w, b)
        loss = logreg_loss(Y_onehot_, probs, w, lambda_)

        dL_dscores = probs - Y_onehot_

        grad_w = (dL_dscores.T @ X).T / N + 2 * lambda_ * w
        grad_b = np.sum(dL_dscores, axis=0) / N

        if e <= 15 or e % log_interval == 0:
            print(f"iteration: {e:5d} loss: {loss.item():2.8f}")

        w -= eta * grad_w
        b -= eta * grad_b

        if e <= 15 or e % log_interval == 0:
            if callback_fn is not None:
                callback_fn(e + 1, w, b)

    return w, b, loss
