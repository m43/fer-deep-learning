import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint

from model.logreg import logreg_get_probs, logreg_train, logreg_loss
from utils.data import graph_surface, graph_data, sample_gmm_2d, eval_perf_multi, class_to_onehot, sample_gauss_2d
from utils.util import project_path, get_str_formatted_time, ensure_dir


def demo(params):
    print(f"params:")
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(params)

    # get data

    np.random.seed(72)
    # X, Y_ = sample_gmm_2d(4, 3, 200)
    # X, Y_ = sample_gmm_2d(4, 3, 200)

    # np.random.seed(100)
    # X, Y_ = sample_gauss_2d(2, 200)

    np.random.seed(100)
    X, Y_ = sample_gauss_2d(3, 100)

    Y_onehot_ = class_to_onehot(Y_)

    fig, ax = plt.subplots()
    plt.title(params["run_name"])
    ims = []
    rect = (np.min(X, axis=0), np.max(X, axis=0))

    def add_artist(iter, w, b):
        probs_train = logreg_get_probs(X, w, b)
        Y = np.argmax(probs_train, axis=-1)
        train_loss = logreg_loss(Y_onehot_, probs_train, w, params["lambda"])

        # gs = graph_surface(lambda x: logreg_get_probs(x, w, b).max(axis=-1), rect)
        gs = graph_surface(lambda x: np.argmax(
            logreg_get_probs(x, w, b), axis=-1), rect)

        d = graph_data(X, Y_, Y)
        txt_iter = plt.text(0.5, 0.9, f"i:{iter}", ha="center", va="bottom",
                            transform=ax.transAxes, fontsize="large",
                            bbox=dict(facecolor='red', alpha=0.3))
        txt_train_loss = plt.text(0.2, 0.15, f"train_loss:{train_loss:.7f}", ha="center", va="bottom",
                                  transform=ax.transAxes, fontsize="small",
                                  bbox=dict(facecolor='red', alpha=0.3))
        ims.append(gs + d + [txt_iter, txt_train_loss])

    # train the model
    w, b, train_loss = logreg_train(
        X, Y_, params["eta"], params["epochs"], params["lambda"],
        params["log_interval"], params["grad_check"], params["grad_check_epsilon"],
        add_artist if params["animate"] else None)

    # get the class predictions
    Y = np.argmax(logreg_get_probs(X, w, b), axis=-1)

    # evaluation
    accuracy, pr, conf = eval_perf_multi(Y, Y_)
    results["train_loss"] = train_loss
    results = {}
    results["train_loss"] = train_loss
    results["accuracy"] = accuracy
    results["(recall_i, precision_i)"] = pr
    results["confusion matrix"] = conf
    results["w"] = w
    results["w_mean"] = w.mean()
    results["w_std"] = w.std()
    pp.pprint(results)

    with open(os.path.join(params["save_dir"], "results.txt"), "w") as f:
        f.write("params:\n")
        f.write(pp.pformat(params))
        f.write("\n")

        f.write("results:")
        f.write(pp.pformat(results))
        f.write("\n")

    if params["animate"]:
        print("Processing animation")
        ani = animation.ArtistAnimation(
            fig, ims, interval=200, blit=True, repeat_delay=500)
        ani.save(os.path.join(params["save_dir"], "animation.mp4"), metadata={
                 'artist': 'm43'})
        print("Animation saved")
        if params["show_plots"]:
            plt.show()
        plt.close()

    # PLOT TRAIN
    plt.title("TRAIN " + params["run_name"])

    # graph the decision surface
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    # graph_surface(lambda x: logreg_get_probs(x, w, b).max(axis=-1), rect)
    graph_surface(lambda x: np.argmax(
        logreg_get_probs(x, w, b), axis=-1), rect)

    # graph the data points
    graph_data(X, Y_, Y, special=[])
    plt.savefig(os.path.join(params["save_dir"], "plot_train.png"), dpi=200)
    if params["show_plots"]:
        plt.show()
    plt.close()


if __name__ == "__main__":
    params = {"epochs": 10000, "eta": 0.1, "lambda": 1e-2}
    params["grad_check"] = True
    params["grad_check_epsilon"] = 1e-5
    params["log_interval"] = 100
    params["animate"] = True
    params["show_plots"] = False

    run_name = f"logreg_eta={params['eta']}_lambda={params['lambda']}_epochs={params['epochs']}_{get_str_formatted_time()}"
    params["run_name"] = run_name
    params["save_dir"] = os.path.join(project_path, f"imgs/logreg/{run_name}")
    ensure_dir(params["save_dir"])

    demo(params)
