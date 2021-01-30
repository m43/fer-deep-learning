import numpy as np
import sys
from tqdm import tqdm

from utils.util import sigmoid


def binlogreg_loss(Y_, probs, w, lambda_):
    N = Y_.shape[0]
    loss = -(Y_ @ np.log(probs + sys.float_info.min) + (1 - Y_) @ np.log(1 - probs + sys.float_info.min)) / N
    loss += lambda_ * np.sum(w ** 2, axis=0)  # L2 regularization
    return loss


def binlogreg_train(X, Y_, eta, epochs, lambda_, log_interval=10, log_grad_check=False, grad_check_epsilon=1e-5,
                    callback_fn=None):
    '''
    Argumenti
    X: podatci, np.array NxD
    Y_: indeksi razreda, np.array Nx1
    Povratne vrijednosti
    w, b: parametri logističke regresije
    loss: loss at end of train process
    '''

    N = X.shape[0]
    D = X.shape[1]
    w = np.random.randn(D)
    b = 0.

    for e in tqdm(range(epochs)):
        if callback_fn is not None and e == 0:
            callback_fn(0, w, b)
        scores = X @ w + b
        probs = sigmoid(scores)

        loss = binlogreg_loss(Y_, probs, w, lambda_)

        dL_dscores = probs - Y_

        grad_w = (dL_dscores.T @ X).T / N + 2 * lambda_ * w
        grad_b = np.sum(dL_dscores) / N

        if e <= 15 or e % log_interval == 0:
            print(f"iteration: {e:5d} loss: {loss.item():2.8f}")
            if log_grad_check:
                w_extended = w + np.eye(w.shape[0] + 1, w.shape[0]) * grad_check_epsilon
                w_extended = np.transpose(w_extended)
                b_extended = np.array(([b] * w.shape[0] + [b + grad_check_epsilon]))
                loss_plus = binlogreg_loss(Y_, sigmoid(X @ w_extended + b_extended), w_extended, lambda_)

                w_extended = w - np.eye(w.shape[0] + 1, w.shape[0]) * grad_check_epsilon
                w_extended = np.transpose(w_extended)
                b_extended = np.array(([b] * w.shape[0] + [b - grad_check_epsilon]))
                loss_minus = binlogreg_loss(Y_, sigmoid(X @ w_extended + b_extended), w_extended, lambda_)

                numerical_grad = (loss_plus - loss_minus) / (2 * grad_check_epsilon)
                grad = np.concatenate((grad_w, [grad_b]))

                # Andrew Ng, http://cs230.stanford.edu/files/C2M1_old.pdf
                numerator = np.linalg.norm(grad - numerical_grad)
                denominator = np.linalg.norm(grad) + np.linalg.norm(numerical_grad)
                grad_diff = numerator / denominator

                print(f"\tGradcheck grad difference (should be below 1e-6): {grad_diff}")

        w -= eta * grad_w
        b -= eta * grad_b

        if e <= 15 or e % log_interval == 0:
            if callback_fn is not None:
                callback_fn(e + 1, w, b)

    return w, b, loss


def binlogreg_classify(X, w, b):
    '''
    Argumenti
    X:
    podatci, np.array NxD
    w, b: parametri logističke regresije
    Povratne vrijednosti
    probs: vjerojatnosti razreda c1
    '''
    probs = sigmoid(X @ w + b)
    return probs
