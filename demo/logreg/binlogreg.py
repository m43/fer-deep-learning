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

"""
2.1 Provjerite poklapaju li se vrijednosti analitičkih gradijenata s njihovim numeričkim aproksimacijama (engl. 
gradient checking). 

>. I think they do. They only differ if the analytical gradients are too small, but otherwise are approximately equal.
The case when the analytical gradients are very small (for params={"epochs": 100000, "eta": 1, "lambda": 0}):
---> analytical gradients: [ 1.04877705e-13  1.16686283e-15 -4.63711292e-13]
---> numerical gradients: [ 7.97972799e-12  6.24500451e-12 -3.46944695e-13]
---> Gradcheck grad difference (should be below 1e-6): 0.9468835403689254
This is how i do grad_check:
dw = 1e-5 (for each w_i separately, the bias weight included)
numerical_grads = (loss(w+dw) - loss(w-dw)) / (2*dw) --> Dx1
diff = |grads-numerical_grads| / ( |grads| + |numerical_grads| )  # Andrew Ng, http://cs230.stanford.edu/files/C2M1_old.pdf
"Good" if (diff <= cca 1e-8) else "Bad" 


2.2 Dodajte regularizacijski gubitak u obliku L2 norme težina w pomnožene hiperparametrom param_lambda (parametre b 
najčešće nema smisla regularizirati). Ne zaboravite izraziti utjecaj te promjene na gradijente. 

>. Larger loss, but weights closer to zero:
before:
 'accuracy': 0.965,
 'loss': 0.0621127106986851,
 'w_mean': -0.9367516145531858,
 'w_std': 2.8487826797504705
after (lambda=0.01)
 'accuracy': 0.96,
 'loss': 0.11441632779277375,
 'w_mean': -0.23347257412606948,
 'w_std': 1.1868119294413713


2.3 Eksperimentirajte s različitim vrijednostima hiper-parametara epochs, eta i lambda. Pronađite kombinacije 
hiper-parametara za koje vaš program ne uspijeva pronaći zadovoljavajuće rješenje i objasnite što se događa. 

>.
name: ex01
params:
    {"epochs": 1000, "eta": 100, "lambda": 0}
results:
{'accuracy': 0.965,
 'loss': 4.336190654272568,
 'precision': 0.9603960396039604,
 'recall': 0.97,
 'w': array([-140.15251288,   82.91952849]),
 'w_mean': -28.616492194262364,
 'w_std': 111.53602068559123}
comment: "overshooting" the minimum, though not very bad as it does not diverge

name: ex02
params:
    {"epochs":10, "eta":0.1, "lambda":0}
results:
{'accuracy': 0.955,
 'loss': 0.15517174386013166,
 'precision': 1.0,
 'recall': 0.91,
 'w': array([-1.90737086,  1.05848016]),
 'w_mean': -0.4244453468708167,
 'w_std': 1.4829255107500334}
comment: not trained for enough epochs. compare to experiment below

name: ex03
params:
    {"epochs":10000, "eta":0.1, "lambda":0}
results:
{'accuracy': 0.965,
 'loss': 0.06524058382834898,
 'precision': 0.9696969696969697,
 'recall': 0.96,
 'w': array([-3.00666517,  2.09776666]),
 'w_mean': -0.4544492542413696,
 'w_std': 2.552215916333711}
 
name: ex04
params:
    {"epochs":10000, "eta":2, "lambda":0}
results:
{'accuracy': 0.965,
 'loss': 0.062069111366528436,
 'precision': 0.9696969696969697,
 'recall': 0.96,
 'w': array([-3.9072773 ,  1.90815627]),
 'w_mean': -0.9995605171867253,
 'w_std': 2.907716784270339}
comment: eta parameter closer to optimum value, makes the convergence faster (compared to ex03)

name: ex05
params:
    {"epochs":10000, "eta":1, "lambda":1}
results:
{'accuracy': 0.5,
 'loss': 832727537.2017386,
 'precision': nan,
 'recall': 0.0,
 'w': array([-16218.68821527, -23871.19071281]),
 'w_mean': -20044.939464037892,
 'w_std': 3826.251248771826}
comment: too much regularization


2.4 Smislite način za vizualiziranje plohe funkcije gubitka i napredovanje postupka optimizacije.

>. Added animations, set params["animation"]=True. Saved to params["save_dir"].

2.5 Procijenite pristranost (eng. bias) i varijancu vašeg postupka većim brojem eksperimenata na podatcima za 
testiranje. Podatke za testiranje dobijte uzorkovanjem iste podatkovne distribucije koja je korištena za dobivanje 
skupa za učenje. Možete li zadati distribuciju podataka za koju će pristranost klasifikatora biti vrlo velika? 

>. TODO
Move this task to another demo.
run many experiments with same config but different datapoints from the same, true distribution. variance should
be, hmmm.. The stddev of all individual weights, then summed? And the bias as the difference from many test datapoints?
TODO google how to estimate model bias and variance
"""
