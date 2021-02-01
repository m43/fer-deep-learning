import os
import torch
import torch.optim as optim
import numpy as np
import pprint
import matplotlib.pyplot as plt

from model.pt_logreg import PTLogreg
from utils.data import sample_gauss_2d, sample_gmm_2d, class_to_onehot, eval_perf_multi, graph_data, graph_surface
from utils.util import get_str_formatted_time, ensure_dir, project_path

def train(model, X, Yoh_, epochs, eta, weight_decay):
    """
    Arguments:
    - X: model inputs [NxD], type: torch.Tensor
    - Yoh_: ground truth [NxC], type: torch.Tensor
    - epochs: number of training iterations
    - eta: learning rate
    - weight_decay: L2 regularization parameter
    """
    optimizer = optim.SGD(model.parameters(), lr=eta, weight_decay=weight_decay)

    for i in range(epochs):
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i<=20 or i%100==0:
            print(f"step:{i:06d}   model_loss:{loss}")

    return loss

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

def demo(params):
    print(f"params:")
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(params)

    # np.random.seed(72)
    # X, Y_ = sample_gmm_2d(4, 3, 200)
    # X, Y_ = sample_gmm_2d(4, 3, 200)

    # np.random.seed(100)
    # X, Y_ = sample_gauss_2d(2, 200)

    np.random.seed(100)
    X, Y_ = sample_gauss_2d(3, 100)

    Yoh_ = class_to_onehot(Y_)
    X, Yoh_ = torch.from_numpy(X), torch.from_numpy(Yoh_)

    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
    train_loss = train(ptlr, X, Yoh_, params["epochs"], params["eta"], params["lambda"])
    probs = eval(ptlr, X.numpy())
    Y = np.argmax(probs, axis=-1)

    accuracy, pr, conf = eval_perf_multi(Y, Y_)
    results = {}
    results["train_loss"] = train_loss
    results["accuracy"] = accuracy
    results["(recall_i, precision_i)"] = pr
    results["confusion matrix"] = conf
    results["w_mean"] = ptlr.w.mean()
    results["w_std"] = ptlr.w.std()
    pp.pprint(results)
    results["w"] = ptlr.w

    # Log to text file
    with open(os.path.join(params["save_dir"], "results.txt"), "w") as f:
        f.write("params:\n")
        f.write(pp.pformat(params))
        f.write("\n")

        f.write("results:")
        f.write(pp.pformat(results))
        f.write("\n")

    X, Yoh_ = X.numpy(), Yoh_.numpy()
    # PLOT TRAIN
    plt.title("TRAIN " + params["run_name"])

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    # graph_surface(lambda x: logreg_get_probs(x, w, b).max(axis=-1), rect)
    graph_surface(lambda x: np.argmax(eval(ptlr, x), axis=-1), rect)

    graph_data(X, Y_, Y, special=[])
    plt.savefig(os.path.join(params["save_dir"], "plot_train.png"), dpi=200)
    if params["show_plots"]:
        plt.show()
    plt.close()

if __name__ == "__main__":
    params = {"epochs": 10000, "eta": 0.1, "lambda": 0.01/2}
    params["show_plots"] = False

    run_name = f"pt_logreg_{get_str_formatted_time()}_eta={params['eta']}_lambda={params['lambda']}_epochs={params['epochs']}"
    params["run_name"] = run_name
    params["save_dir"] = os.path.join(project_path, f"imgs/pt_logreg/{run_name}")
    ensure_dir(params["save_dir"])

    demo(params)
