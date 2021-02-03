import numpy as np
import os
import pprint
import torch

from demo.util import PTUtil
from model.pt_logreg import PTLogreg
from utils.data import sample_gauss_2d, class_to_onehot
from utils.util import get_str_formatted_time, ensure_dir, project_path


def demo(params, X, Y_):
    print(f"params:")
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(params)

    Yoh_ = class_to_onehot(Y_)
    X, Yoh_ = torch.from_numpy(X), torch.from_numpy(Yoh_)

    model = PTLogreg(X.shape[1], Yoh_.shape[1])
    train_loss = PTUtil.train(model, X, Yoh_, params["epochs"], params["eta"], params["lambda"])
    PTUtil.log(X, Y_, Yoh_, params, pp, model, train_loss)


if __name__ == "__main__":
    # DATASET

    # np.random.seed(72)
    # X, Y_ = sample_gmm_2d(4, 3, 200)
    # X, Y_ = sample_gmm_2d(4, 3, 200)

    # np.random.seed(100)
    # X, Y_ = sample_gauss_2d(2, 200)

    np.random.seed(100)
    X, Y_ = sample_gauss_2d(3, 100)

    # PARAMETERS
    params = {
        "epochs": 10000,
        "eta": 0.1,
        "lambda": 0.01,
        "show_plots": False,
    }

    run_name = f"pt_logreg_{get_str_formatted_time()}_eta={params['eta']}_lambda={params['lambda']}_epochs={params['epochs']}"
    params["run_name"] = run_name
    params["save_dir"] = os.path.join(project_path, f"imgs/pt_logreg/{run_name}")
    ensure_dir(params["save_dir"])

    demo(params, X, Y_)
