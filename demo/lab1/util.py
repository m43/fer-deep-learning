import os
from functools import reduce

import numpy as np
import torch
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt

from utils.data import eval_perf_multi, graph_surface, graph_data, sample_gauss_2d, sample_gmm_2d
from utils.util import project_path


class PTUtil:
    @staticmethod
    def train(model, X, Yoh_, epochs, eta, weight_decay, callback_fn=None):
        """
        Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - epochs: number of training iterations
        - eta: learning rate
        - weight_decay: L2 regularization parameter
        - callback_fn: callback function called before training and after each iteration, f(i, model, X, Yoh_)
        """
        optimizer = optim.SGD(model.parameters(), lr=eta)

        if callback_fn is not None:
            callback_fn(0, model, X, Yoh_)

        for i in range(epochs):
            loss = model.get_loss(X, Yoh_, weight_decay)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i <= 20 or i % 100 == 0:
                print(f"step:{i:06d}   model_loss:{loss}")

            if callback_fn is not None:
                callback_fn(i + 1, model, X, Yoh_)

        return loss

    @staticmethod
    def eval(model, X):
        """
        Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD] TODO not numpy.
        Returns: predicted class probabilites [NxC], type: np.array
        """
        # X = torch.from_numpy(X)
        probs = model.forward(X)
        return probs.detach().cpu().numpy()

    @staticmethod
    def log_2d(X, Y_, Yoh_, params, pretty_printer, model, train_loss, special=[]):
        probs = PTUtil.eval(model, X.numpy())
        Y = np.argmax(probs, axis=-1)

        accuracy, pr, conf = eval_perf_multi(Y, Y_)
        results = {}
        results["train_loss"] = train_loss
        results["accuracy"] = accuracy
        results["(recall_i, precision_i)"] = pr
        results["confusion matrix"] = conf
        pretty_printer.pprint(results)

        # Log to text file
        with open(os.path.join(params["save_dir"], "results.txt"), "w") as f:
            f.write("params:\n")
            f.write(pretty_printer.pformat(params))
            f.write("\n")

            f.write("results:")
            f.write(pretty_printer.pformat(results))
            f.write("\n")

        # PLOT TRAIN 1
        X, Yoh_ = X.numpy(), Yoh_.numpy()
        plt.title("TRAIN " + params["run_name"])

        rect = (np.min(X, axis=0), np.max(X, axis=0))
        graph_surface(lambda x: np.argmax(PTUtil.eval(model, x), axis=-1), rect)
        graph_data(X, Y_, Y, special=special)

        plt.savefig(os.path.join(params["save_dir"], "plot_train_1.png"), dpi=200)
        if params["show_plots"]:
            plt.show()
        plt.close()

        # PLOT TRAIN 2
        plt.title("TRAIN " + params["run_name"])
        graph_surface(lambda x: PTUtil.eval(model, x).max(axis=-1), rect)
        graph_data(X, Y_, Y, special=special)
        plt.savefig(os.path.join(params["save_dir"], "plot_train_2.png"), dpi=200)
        if params["show_plots"]:
            plt.show()
        plt.close()

        return results

    @staticmethod
    def animation_callback():
        pass

    @staticmethod
    def count_params(model):
        print("#" * 60)
        print("#" * 60)
        print(f"Parameters of model {model}")
        print("#" * 60)
        total_n_of_parameters = 0
        for name, parameter in model.named_parameters():
            parameter_shape = list(parameter.shape)
            print(f"{str(parameter_shape):>20s} <-- {name}")
            total_n_of_parameters += reduce(lambda a, b: a * b, parameter_shape) if len(parameter_shape) else 0
        print(f"Total number of parameters: {total_n_of_parameters}")
        print("#" * 60)
        print("#" * 60)

    @staticmethod
    def load_dataset(id):
        if id == 1:
            return sample_gauss_2d(3, 100)
        elif id == 2:
            return sample_gmm_2d(4, 2, 40)
        elif id == 3:
            return sample_gmm_2d(6, 2, 10)
        else:
            raise RuntimeError(f"Dataset with id {id} is not supported")

    @staticmethod
    def get_loss_and_acc(model, weight_decay, x, y, yoh, prefix=""):
        loss = model.get_loss(x, yoh, weight_decay).item()

        probs = model.forward(x)
        y_pred = torch.argmax(probs, dim=-1)

        accuracy = ((y == y_pred).float().sum() / y_pred.shape[0]).item()

        results = {
            prefix + "loss": loss,
            prefix + "accuracy": accuracy,
        }

        return results

    @staticmethod
    def get_perf_multi(model, weight_decay, x, y, yoh, prefix="", for_each_class=True):
        loss = model.get_loss(x, yoh, weight_decay).item()

        probs = model.forward(x).clone().detach().cpu()
        y_pred = torch.argmax(probs, dim=-1)

        accuracy, pr, conf = eval_perf_multi(y_pred.numpy(), y.clone().detach().cpu().numpy())
        recall = sum([r for r, _ in pr]) / len(pr)
        precision = sum([p for _, p in pr]) / len(pr)
        f1 = 2 / ((1 / recall) + (1 / precision))
        results = {
            prefix + "loss": loss,
            prefix + "accuracy": accuracy,
            prefix + "recall": recall,
            prefix + "precision": precision,
            prefix + "f1": f1,
        }

        if for_each_class:
            results[prefix + "(recall_i, precision_i)"] = pr
            results[prefix + "confusion matrix"] = conf

        return results

    @staticmethod
    def standardize_inplace(x_train, x_rest=[]):
        train_mean = x_train.mean()
        train_std = x_train.std()

        x_train.sub_(train_mean).div_(train_std)
        for x in x_rest:
            x.sub_(train_mean).div_(train_std)

        return train_mean, train_std

    @staticmethod
    def load_mnist(train_valid_ratio=5 / 6):
        dataset_root = os.path.join(project_path, "datasets")
        mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
        mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

        x_train, y_train = mnist_train.data, mnist_train.targets
        x_test, y_test = mnist_test.data, mnist_test.targets
        x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

        #  might be better to shuffle first, but the loaded dataset is already shuffled so all good.
        N = x_train.shape[0]
        split_size = int(N * train_valid_ratio)
        x_train, x_valid = x_train.split([split_size, N - split_size])
        y_train, y_valid = y_train.split([split_size, N - split_size])

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
