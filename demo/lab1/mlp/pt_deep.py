import numpy as np
import os
import pprint
import torch

from demo.lab1.util import PTUtil
from model.pt_deep import PTDeep
from model.pt_deep import PTDeep2
from utils.data import class_to_onehot
from utils.util import get_str_formatted_time, ensure_dir, project_path


def demo(params, X, Y_):
    print(f"params:")
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(params)

    Yoh_ = class_to_onehot(Y_)
    if params["whiten input"]:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    X, Yoh_ = torch.from_numpy(X), torch.from_numpy(Yoh_)

    model = None
    if params["deep_version"] == 1:
        model = PTDeep(params["neurons_per_layer"], params["activation"])
    elif params["deep_version"] == 2:
        model = PTDeep2(params["neurons_per_layer"], params["activation"])
    model.to(X.dtype)

    train_loss = PTUtil.train(model, X, Yoh_, params["epochs"], params["eta"], params["lambda"])
    PTUtil.count_params(model)
    PTUtil.log_2d(X, Y_, Yoh_, params, pp, model, train_loss)


if __name__ == "__main__":
    torch.manual_seed(72)
    np.random.seed(100)
    dataset_id = 1
    X, Y_ = PTUtil.load_dataset(dataset_id)

    # PARAMETERS
    params = {
        "deep_version": 1,  # 1 or 2
        "whiten input": True,
        "activation": "relu",  # relu, sigmoid, tanh
        # "neurons_per_layer": [X.shape[1], Y_.max() + 1],  # [D, H_1, H_2, ..., H_N, C]
        # "neurons_per_layer": [X.shape[1], 10, Y_.max() + 1],
        # "neurons_per_layer": [X.shape[1], 10, 10, Y_.max() + 1],
        "neurons_per_layer": [X.shape[1], 10, 10, 10, Y_.max() + 1],
        "dataset": dataset_id,
        "epochs": 30000,
        "eta": 0.2,
        "lambda": 1e-4,
        "show_plots": False,
    }

    run_name = f"pt_deep" \
               f"_{get_str_formatted_time()}" \
               f"_w={1 if params['whiten input'] else 0}" \
               f"_d={params['dataset']}" \
               f"_v={params['deep_version']}" \
               f"_npl={params['neurons_per_layer']}" \
               f"_act={params['activation']}" \
               f"_eta={params['eta']}" \
               f"_lambda={params['lambda']}" \
               f"_epochs={params['epochs']}"

    params["run_name"] = run_name
    params["save_dir"] = os.path.join(project_path, f"imgs/pt_deep/{run_name}")
    ensure_dir(params["save_dir"])

    demo(params, X, Y_)
