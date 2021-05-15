import argparse
import numpy as np
import os
import random
import torch

from demo.lab1.util import PTUtil
from model.ksvm_wrap import KSVMWrap
from utils.data import class_to_onehot
from utils.util import ensure_dir, project_path, get_str_formatted_time


def mnist_demo(c, gamma, kernel, seed=72):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PARAMETERS
    params = {
        "c": c,
        "gamma": gamma,
        "dataset": "mnist",
        "kernel": kernel,
        "show_plots": False,
    }

    run_name = f"ksvm_wrap" \
               f"_kernel={params['kernel']}" \
               f"_d={params['dataset']}" \
               f"_c={params['c']}" \
               f"_g={params['gamma']}"

    (x_train, y_train), (_, __), (x_test, y_test) = PTUtil.load_mnist(train_valid_ratio=1)  # SVM need no valid dataset
    x_train, x_test = x_train.flatten(-2), x_test.flatten(-2)
    y_train_oh, y_test_oh = class_to_onehot(y_train, True), class_to_onehot(y_test, True)

    N = x_train.shape[0]
    D = x_train.shape[1]
    C = y_train.max().add_(1).item()

    svm = KSVMWrap(x_train, y_train, params["c"], params["gamma"], decision_function_shape="ovo", kernel=kernel)

    class Adapter:
        def __init__(self, svm):
            self.svm = svm

        def forward(self, X):
            return torch.from_numpy(svm.get_scores(X))

        def get_loss(self, X, Yoh_, _):
            Y = self.forward(X)
            nLL = -(Yoh_ * torch.log(Y)).sum(axis=1).mean()
            return nLL

    model = Adapter(svm)

    current_results = {
        "c": c, "gamma": gamma, "kernel": kernel,
    }
    current_results.update(
        PTUtil.get_perf_multi(model, 0, x_train, y_train, y_train_oh, "train_", False))
    current_results.update(
        PTUtil.get_perf_multi(model, 0, x_test, y_test, y_test_oh, "test_", False))

    print(c, gamma, current_results)
    return current_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration parameters')
    parser.add_argument('--cs', dest='cs', nargs="+", type=float, default=[1])
    parser.add_argument('--gammas', dest='gammas', nargs="+", type=str, default=["auto"])
    parser.add_argument('--kernel', dest='kernel', type=str, default="rbf")
    args = parser.parse_args()
    assert (args.cs is not None)
    assert (len(args.cs) != 0)
    assert (args.gammas is not None)
    assert (len(args.gammas) != 0)

    save_dir = os.path.join(project_path, f"imgs/ksvm_wrap_mnist"
                                          f"/cs={'-'.join([str(x) for x in args.cs])}"
                                          f"_gammas={'-'.join([str(x) for x in args.gammas])}")
    ensure_dir(save_dir)
    print(os.path.abspath(save_dir))

    print(get_str_formatted_time())
    header = "kernel,c,gamma,test_accuracy,test_precision,test_recall,train_accuracy,test_loss,train_loss"

    # cs = [0.1, 0.5, 1, 2, 5, 10, 100, 1000]
    # cs = [1, 5, 10, 100, 1000]
    # gammas = ["auto", 3, 1, 0.1, 0.05, 0.01, 0.001, 0.0001]
    # gammas = ["auto", 3, 1, 0.1]

    args.gammas = ["auto" if g == "auto" else float(g) for g in args.gammas]

    results = [header]
    for c in args.cs:
        for gamma in args.gammas:
            r = mnist_demo(c, gamma, args.kernel)
            results.append(','.join([str(r[k]) for k in header.split(",")]))

    with open(os.path.join(save_dir, "results.csv"), "w") as f:
        for line in results:
            f.write(line + "\n")
        f.write("\n")
    with open(os.path.join(save_dir, "results.csv"), "r") as f:
        print(f.read())
