import numpy as np
import os
import pprint
import torch

from demo.lab1.util import PTUtil
from model.ksvm_wrap import KSVMWrap
from utils.data import class_to_onehot
from utils.util import get_str_formatted_time, ensure_dir, project_path


def demo(params, X, Y_, ):
    print(f"params:")
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(params)

    Yoh_ = class_to_onehot(Y_)
    if params["whiten input"]:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    X, Yoh_ = torch.from_numpy(X), torch.from_numpy(Yoh_)

    svm = KSVMWrap(X, Y_, params["c"], params["gamma"])
    support_indices = svm.support_indices

    class Adapter:
        def __init__(self, svm):
            self.svm = svm

        def forward(self, X):
            return torch.from_numpy(svm.get_scores(X))

        def get_loss(self, X, Yoh_, weight_decay):
            Y = self.forward(X)
            nLL = -(Yoh_ * torch.log(Y)).sum(axis=1).mean()
            return nLL

    model = Adapter(svm)
    train_loss = model.get_loss(X, Yoh_, 0)

    results = PTUtil.log_2d(X, Y_, Yoh_, params, pp, model, train_loss, special=support_indices)

    return results


def meta_demo(dataset_id, c, gamma):
    torch.manual_seed(72)
    np.random.seed(100)
    X, Y_ = PTUtil.load_dataset(dataset_id)

    # PARAMETERS
    params = {
        "c": c,
        "gamma": gamma,
        "whiten input": False,
        "dataset": dataset_id,
        "show_plots": False,
    }

    run_name = f"ksvm_wrap" \
               f"_{get_str_formatted_time()}" \
               f"_w={1 if params['whiten input'] else 0}" \
               f"_d={params['dataset']}" \
               f"_c={params['c']}" \
               f"_g={params['gamma']}"

    params["run_name"] = run_name
    params["save_dir"] = os.path.join(project_path, f"imgs/ksvm_wrap/{run_name}")
    ensure_dir(params["save_dir"])

    return demo(params, X, Y_)


if __name__ == "__main__":
    results = []

    # for dataset_id in [1, 2, 3]:
    #     for c in [0.1, 1, 10, 100]:
    #         for gamma in ["auto"]:
    #             r = meta_demo(dataset_id, c, gamma)
    #             results.append(f'{dataset_id},{c},{gamma},{r["accuracy"]:.5f},{r["train_loss"]:.7f}')

    for dataset_id in [1, 2, 3]:
        for c in [0.1, 0.5, 1, 2, 5, 10, 100, 1000]:
            for gamma in ["auto", 3, 1, 0.1, 0.05, 0.01, 0.001, 0.0001]:
                r = meta_demo(dataset_id, c, gamma)
                results.append(f'{dataset_id},{c},{gamma},{r["accuracy"]:.5f},{r["train_loss"]:.7f}')

    print("***RESULTS***")
    for result in results:
        print(result)
