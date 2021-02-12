import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint

from model.binlogreg import binlogreg_classify, binlogreg_train, binlogreg_loss
from utils.data import sample_gauss_2d, graph_surface, graph_data, eval_perf_binary
from utils.util import project_path, get_str_formatted_time, ensure_dir


def generate_uniform_dataset():
    X_all, Y_all_ = sample_gauss_2d(2, 200)  # 400 points
    train_indices = (list(range(0, 100)) + list(range(200, 300)))
    test_indices = (list(range(100, 200)) + list(range(300, 400)))
    X_train, Y_train_ = X_all[train_indices], Y_all_[train_indices]
    X_test, Y_test_ = X_all[test_indices], Y_all_[test_indices]

    return X_all, Y_all_, X_train, Y_train_, X_test, Y_test_


def generate_unbalanced_dataset():
    X_all, Y_all_ = sample_gauss_2d(2, 300)  # 600 points
    train_indices = (list(range(0, 30)) + list(range(300, 470)))
    test_indices = (list(range(200, 300)) + list(range(500, 600)))
    X_train, Y_train_ = X_all[train_indices], Y_all_[train_indices]
    X_test, Y_test_ = X_all[test_indices], Y_all_[test_indices]

    return X_all, Y_all_, X_train, Y_train_, X_test, Y_test_


def demo(params):
    print(f"params:")
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(params)

    # get data
    np.random.seed(100)
    X_all, Y_all_, X_train, Y_train_, X_test, Y_test_ = generate_uniform_dataset()
    # X_all, Y_all_, X_train, Y_train_, X_test, Y_test_ = generate_unbalanced_dataset()

    fig, ax = plt.subplots()
    plt.title(params["run_name"])
    ims = []
    rect = (np.min(X_all, axis=0), np.max(X_all, axis=0))
    X_all = np.vstack((X_train, X_test))
    Y_all_ = np.hstack((Y_train_, Y_test_))
    test_indices_in_all_datapoints = list(range(len(Y_train_), len(Y_train_) + len(Y_test_)))

    def add_artist(iter, w, b):
        probs_train = binlogreg_classify(X_train, w, b)
        Y_train = probs_train > 0.5
        train_loss = binlogreg_loss(Y_train_, probs_train, w, params["lambda"])

        probs_test = binlogreg_classify(X_test, w, b)
        Y_test = probs_test > 0.5
        test_loss = binlogreg_loss(Y_test_, probs_test, w, params["lambda"])

        Y_all = binlogreg_classify(X_all, w, b) > 0.5

        if iter == 0:
            graph_data(X_all, Y_all_, Y_all, special=test_indices_in_all_datapoints)

        gs = graph_surface(lambda x: binlogreg_classify(x, w, b), rect, offset=None)
        d = graph_data(X_all, Y_all_, Y_all, special=test_indices_in_all_datapoints)
        txt_iter = plt.text(0.5, 0.9, f"i:{iter}", ha="center", va="bottom",
                            transform=ax.transAxes, fontsize="large",
                            bbox=dict(facecolor='red', alpha=0.3))
        txt_train_loss = plt.text(0.2, 0.15, f"train_loss:{train_loss:.7f}", ha="center", va="bottom",
                                  transform=ax.transAxes, fontsize="small",
                                  bbox=dict(facecolor='red', alpha=0.3))
        txt_test_loss = plt.text(0.2, 0.05, f"test__loss:{test_loss:.7f}", ha="center", va="bottom",
                                 transform=ax.transAxes, fontsize="small",
                                 bbox=dict(facecolor='red', alpha=0.3))

        ims.append(gs + d + [txt_iter, txt_train_loss, txt_test_loss])

    # train the model
    w, b, train_loss = binlogreg_train(
        X_train, Y_train_, params["eta"], params["epochs"], params["lambda"],
        params["log_interval"], params["grad_check"], params["grad_check_epsilon"],
        add_artist if params["animate"] else None)

    # get the class predictions
    Y_all = binlogreg_classify(X_all, w, b) > 0.5
    Y_train = binlogreg_classify(X_train, w, b) > 0.5

    probs_test = binlogreg_classify(X_test, w, b)
    Y_test = probs_test > 0.5
    test_loss = binlogreg_loss(Y_test_, probs_test, w, params["lambda"])

    # evaluation
    accuracy, recall, precision = eval_perf_binary(Y_test, Y_test_)
    # ap_score = eval_AP()  # TODO
    results = {}
    results["train_loss"] = train_loss
    results["test_loss"] = test_loss
    results["accuracy"] = accuracy
    results["precision"] = precision
    results["recall"] = recall
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
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=500)
        ani.save(os.path.join(params["save_dir"], "animation.mp4"), metadata={'artist': 'm43'})
        print("Animation saved")
        if params["show_plots"]:
            plt.show()
        plt.close()

    # PLOT TRAIN
    plt.title("TRAIN " + params["run_name"])
    # graph the decision surface
    rect = (np.min(X_all, axis=0), np.max(X_all, axis=0))
    graph_surface(lambda x: binlogreg_classify(x, w, b), rect, offset=None)
    # graph the data points
    graph_data(X_train, Y_train_, Y_train, special=[])
    plt.savefig(os.path.join(params["save_dir"], "plot_train.png"), dpi=200)
    if params["show_plots"]:
        plt.show()
    plt.close()

    # PLOT ALL DATAPOINTS
    plt.title("ALL " + params["run_name"])
    graph_surface(lambda x: binlogreg_classify(x, w, b), rect, offset=None)
    graph_data(X_all, Y_all_, Y_all, special=test_indices_in_all_datapoints)
    plt.savefig(os.path.join(params["save_dir"], "plot_all.png"), dpi=200)
    if params["show_plots"]:
        plt.show()
    plt.close()


if __name__ == "__main__":
    params = {"epochs": 1000, "eta": 1, "lambda": 1e-3}
    params["grad_check"] = True
    params["grad_check_epsilon"] = 1e-5
    params["log_interval"] = 100
    params["animate"] = True
    params["show_plots"] = False

    run_name = f"binlogreg_eta={params['eta']}_lambda={params['lambda']}_epochs={params['epochs']}_{get_str_formatted_time()}"
    params["run_name"] = run_name
    params["save_dir"] = os.path.join(project_path, f"imgs/binlogreg/{run_name}")
    ensure_dir(params["save_dir"])

    demo(params)
