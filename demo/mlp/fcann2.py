import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint

from model.fcann2 import fcann2_forward, fcann2_loss, fcann2_train
from utils.data import graph_surface, graph_data, sample_gmm_2d, eval_perf_multi, class_to_onehot, sample_gauss_2d
from utils.util import project_path, get_str_formatted_time, ensure_dir


def demo(params):
    print(f"params:")
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(params)

    # get data
    np.random.seed(100)
    X, Y_ = sample_gmm_2d(6, 2, 10)
    # X, Y_ = sample_gmm_2d(4, 3, 200)

    # np.random.seed(100)
    # X, Y_ = sample_gauss_2d(2, 200)

    # np.random.seed(100)
    # X, Y_ = sample_gauss_2d(3, 100)

    # data preprocessing
    # X = (X-X.mean(axis=0))/X.std(axis=0)
    Y_onehot_ = class_to_onehot(Y_)

    fig, ax = plt.subplots()
    plt.title(params["run_name"])
    ims = []
    rect = (np.min(X, axis=0), np.max(X, axis=0))

    def add_artist(iter, w1, b1, w2, b2):
        s1, h1, s2, probs_train = fcann2_forward(X, w1, b1, w2, b2)
        Y = np.argmax(probs_train, axis=-1)
        train_loss = fcann2_loss(
            Y_onehot_, probs_train, w1, w2, params["lambda"])

        # gs = graph_surface(lambda x: fcann2_forward(
        # x, w1, b1, w2, b2)[3].max(axis=-1), rect)
        gs = graph_surface(lambda x: np.argmax(
            fcann2_forward(x, w1, b1, w2, b2)[3], axis=-1), rect)

        d = graph_data(X, Y_, Y)
        txt_iter = plt.text(0.5, 0.9, f"i:{iter}", ha="center", va="bottom",
                            transform=ax.transAxes, fontsize="large",
                            bbox=dict(facecolor='red', alpha=0.3))
        txt_train_loss = plt.text(0.2, 0.15, f"train_loss:{train_loss:.7f}", ha="center", va="bottom",
                                  transform=ax.transAxes, fontsize="small",
                                  bbox=dict(facecolor='red', alpha=0.3))
        ims.append(gs + d + [txt_iter, txt_train_loss])

    # train the model
    w1, b1, w2, b2, train_loss = fcann2_train(
        X, Y_, params["H"], params["eta"], params["epochs"], params["lambda"],
        params["log_interval"], add_artist if params["animate"] else None)

    # get the class predictions
    s1, h1, s2, probs = fcann2_forward(X, w1, b1, w2, b2)
    Y = np.argmax(probs, axis=-1)

    # evaluation
    accuracy, pr, conf = eval_perf_multi(Y, Y_)
    results = {}
    results["train_loss"] = train_loss
    results["accuracy"] = accuracy
    results["(recall_i, precision_i)"] = pr
    results["confusion matrix"] = conf
    results["w1_mean"] = w1.mean()
    results["w1_std"] = w1.std()
    results["w2_mean"] = w2.mean()
    results["w2_std"] = w2.std()
    pp.pprint(results)
    results["w1"] = w1
    results["b1"] = b1
    results["w2"] = w2
    results["b2"] = b2

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
    graph_surface(lambda x: np.argmax(
        fcann2_forward(x, w1, b1, w2, b2)[3], axis=-1), rect)

    # graph the data points
    graph_data(X, Y_, Y, special=[])
    plt.savefig(os.path.join(params["save_dir"], "plot_train_1.png"), dpi=200)
    if params["show_plots"]:
        plt.show()
    plt.close()

    # graph the decision surface
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(lambda x: fcann2_forward(
        x, w1, b1, w2, b2)[3].max(axis=-1), rect)

    # graph the data points
    graph_data(X, Y_, Y, special=[])
    plt.savefig(os.path.join(params["save_dir"], "plot_train_2.png"), dpi=200)
    if params["show_plots"]:
        plt.show()
    plt.close()


if __name__ == "__main__":
    params = {"epochs": 100000, "eta": 0.05, "lambda": 1e-3, "H": 5}
    # params = {"epochs": 10000, "eta": 1e-2, "lambda": 0, "H": 100}
    params["log_interval"] = 1000
    params["animate"] = True
    params["show_plots"] = False

    run_name = f"fcann2_{get_str_formatted_time()}_H={params['H']}_eta={params['eta']}_lambda={params['lambda']}_epochs={params['epochs']}"
    params["run_name"] = run_name
    params["save_dir"] = os.path.join(project_path, f"imgs/fcann2/{run_name}")
    ensure_dir(params["save_dir"])

    demo(params)
