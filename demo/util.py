import numpy as np
import os
import torch
import torch.optim as optim
from functools import reduce
from matplotlib import pyplot as plt

from utils.data import eval_perf_multi, graph_surface, graph_data, sample_gauss_2d, sample_gmm_2d


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
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
        """
        X = torch.from_numpy(X)
        probs = model.forward(X)
        return probs.detach().numpy()

    @staticmethod
    def log(X, Y_, Yoh_, params, pretty_printer, model, train_loss, special=[]):
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
            total_n_of_parameters += reduce(lambda a, b: a * b, parameter_shape)
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
