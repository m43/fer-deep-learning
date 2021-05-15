import copy

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import random
import torch
import torch.optim as optim
from matplotlib.ticker import StrMethodFormatter
from tqdm import tqdm

from demo.util import PTUtil
from model.pt_deep import PTDeep, PTDeep2
from utils.data import class_to_onehot
from utils.util import ensure_dir, project_path


def train_and_eval(params, model, x_train, y_train, y_train_oh, x_valid, y_valid, y_valid_oh, x_test, y_test, y_test_oh,
                   device):
    scheduler = None
    if params["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=params["eta"])
    elif params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params["eta"])
    elif params["optimizer"] == "adam+ls":
        optimizer = optim.Adam(model.parameters(), lr=params["eta"])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1 - params["eta"])  # TODO mislim da nema smisla 1-eta

    else:
        raise RuntimeError(f"Invalid value for optimizer, got '{params['optimizer']}'")

    best_epoch, best_loss, best_state_dict = -1, None, None
    all_results = {}
    with open(os.path.join(params["save_dir"], "params.txt"), "w") as f:
        f.write(f"{pp.pformat(params)}\n")
    batch_size = params["batch_size"]
    N = x_train.shape[0]

    t = tqdm(range(1, params["epochs"] + 1), desc="", leave=True)
    for e in t:
        print()
        model.to(device)
        model.eval()

        x_train, y_train, y_train_oh = x_train.to(device), y_train.to(device), y_train_oh.to(device)

        if e == 1:
            log(0, all_results, device, model, params, x_train, y_train, y_train_oh, x_valid, y_valid,
                y_valid_oh, t)
            best_loss = all_results["valid_loss"][-1] if x_valid is not None else all_results["train_loss"][-1]
            best_epoch = 0
            best_state_dict = copy.deepcopy(model.state_dict())

        if params["shuffle"]:
            perm = torch.randperm(N)
            x_train_perm, y_train_oh_perm = x_train[perm], y_train_oh[perm]
        else:
            x_train_perm, y_train_oh_perm = x_train, y_train_oh

        model.train()
        for batch_idx in range(np.math.ceil(N / batch_size)):
            x_mb = x_train_perm[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            y_mb_oh = y_train_oh_perm[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            _loss = model.get_loss(x_mb, y_mb_oh, params["weight_decay"])
            _loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        model.eval()

        if e < 5 or e == 100 or e % params["log_interval"] == 0 or e == params["epochs"] - 1:
            log(e, all_results, device, model, params, x_train, y_train, y_train_oh,
                x_valid, y_valid, y_valid_oh, t)

            loss = all_results["valid_loss"][-1] if x_valid is not None else all_results["train_loss"][-1]
            if best_loss is None or loss < best_loss - params[
                "early_stopping_epsilon"]:  # Early stopping could be moved outside of log loop (faster here)
                best_epoch = e
                best_loss = loss
                best_state_dict = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), os.path.join(params["save_dir"], f"model_state_dict_best.pth"))
            elif e - best_epoch >= params["early_stopping"]:
                print("Early stopping...")
                break

        if e < 5 or e == 100 or e % params["log_images_interval"] == 0 or e == params["epochs"] - 1:
            log_images(e, all_results, device, model, params)

    # end model
    torch.save(model, os.path.join(params["save_dir"], f"model_best.pth"))
    torch.save(model, os.path.join(params["save_dir"], f"model_end_{device.type}.pth"))
    if device.type != "cpu":
        model.to("cpu")
        torch.save(model, os.path.join(params["save_dir"], f"model_end_cpu.pth"))
    torch.save(best_state_dict, os.path.join(params["save_dir"], f"model_end_state_dict.pth"))

    model.to(device)
    all_results.update(PTUtil.get_perf_multi(model, params["weight_decay"], x_train, y_train, y_train_oh, "end_train_"))
    if x_valid is not None:
        x_valid, y_valid, y_valid_oh = x_valid.to(device), y_valid.to(device), y_valid_oh.to(device)
        all_results.update(
            PTUtil.get_perf_multi(model, params["weight_decay"], x_valid, y_valid, y_valid_oh, "end_valid_"))
    x_test, y_test, y_test_oh = x_test.to(device), y_test.to(device), y_test_oh.to(device)
    all_results.update(PTUtil.get_perf_multi(model, params["weight_decay"], x_test, y_test, y_test_oh, "best_test_"))

    # best model
    model.load_state_dict(best_state_dict)
    torch.save(model, os.path.join(params["save_dir"], f"model_best_{device.type}.pth"))
    if device.type != "cpu":
        model.to("cpu")
        torch.save(model, os.path.join(params["save_dir"], f"model_best_cpu.pth"))
    torch.save(best_state_dict, os.path.join(params["save_dir"], f"model_best_state_dict.pth"))

    model.to(device)

    all_results["best_epoch"] = best_epoch
    best_train_results = PTUtil.get_perf_multi(model, params["weight_decay"], x_train, y_train, y_train_oh,
                                               "best_train_")
    all_results.update(best_train_results)
    pp.pprint(best_train_results)

    if x_valid is not None:
        x_valid, y_valid, y_valid_oh = x_valid.to(device), y_valid.to(device), y_valid_oh.to(device)
        best_valid_results = PTUtil.get_perf_multi(model, params["weight_decay"], x_valid, y_valid, y_valid_oh,
                                                   "best_valid_")
        all_results.update(best_valid_results)
        pp.pprint(best_valid_results)

    x_test, y_test, y_test_oh = x_test.to(device), y_test.to(device), y_test_oh.to(device)
    best_test_results = PTUtil.get_perf_multi(model, params["weight_decay"], x_test, y_test, y_test_oh, "best_test_")
    all_results.update(best_test_results)
    pp.pprint(best_test_results)

    print(f"Best epoch: {best_epoch}")
    log_images(e, all_results, device, model, params)
    return all_results, model


def update_all_results(all_results, new_results):
    for k, v in new_results.items():
        if k not in all_results:
            all_results[k] = []
        all_results[k].append(v)


def log(i, all_results, device, model, params, x_train, y_train, y_train_oh, x_valid,
        y_valid, y_valid_oh, t=None):
    results = {"i": i}

    results.update(PTUtil.get_loss_and_acc(model, params["weight_decay"], x_train, y_train, y_train_oh, "train_"))
    if x_valid is not None:
        x_valid, y_valid, y_valid_oh = x_valid.to(device), y_valid.to(device), y_valid_oh.to(device)
        results.update(PTUtil.get_loss_and_acc(model, params["weight_decay"], x_valid, y_valid, y_valid_oh, "valid_"))
    if t is not None:
        t.set_description(f"tloss:{results['train_loss']:.6f} tacc:{results['train_accuracy']:.4f} "
                          f"vloss:{results['valid_loss']:.6f} vacc:{results['valid_accuracy']:.4f}")

    update_all_results(all_results, results)


def log_images(i, all_results, device, model, params):
    with open(os.path.join(params["save_dir"], "results.txt"), "w") as f:
        f.write(f"{all_results}\n")

    model.to("cpu")
    if len(params["neurons_per_layer"]) == 2:
        w = None
        if params["deep_version"] == 1:
            w = model.weights[0].T
            w = w.clone().detach()
        elif params["deep_version"] == 2:
            w = model.seq.ll_1.weight.data

        for digit in range(10):
            plt.title(f"weights for digit {digit}, npl:{params['neurons_per_layer']}")
            plt.imshow(w[digit].reshape((28, 28)), cmap=plt.get_cmap('gray'))
            plt.savefig(os.path.join(params["save_dir"], f"digit_{i:06d}_{digit}.png"), dpi=300)
            plt.close()

    model.to(device)

    plt.figure(figsize=(6.4 * 1.3, 4.8 * 1.26))
    plt.title("Loss over epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(all_results["i"][2:], all_results["train_loss"][2:], color="green", label="train")
    plt.plot(all_results["i"][2:], all_results["valid_loss"][2:], color="red", label="valid")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(params["save_dir"], f"loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6.4 * 1.3, 4.8 * 1.26))
    plt.title("Log loss over epoch")
    plt.xlabel("epoch")
    plt.ylabel("log loss")
    plt.plot(all_results["i"][2:], np.log(np.array(all_results["train_loss"][2:])), color="green", label="train")
    plt.plot(all_results["i"][2:], np.log(np.array(all_results["valid_loss"][2:])), color="red", label="valid")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.3e}'))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(params["save_dir"], f"logloss.png"), dpi=300)
    plt.close()

    plt.title("Accuracy over epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0, 100)
    plt.plot(all_results["i"][1:], all_results["train_accuracy"][1:], color="green", label="train")
    plt.plot(all_results["i"][1:], all_results["valid_accuracy"][1:], color="red", label="valid")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(params["save_dir"], f"accuracy.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # load dataset
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = PTUtil.load_mnist()
    x_train, x_valid, x_test = x_train.flatten(-2), x_valid.flatten(-2), x_test.flatten(-2)
    y_train_oh, y_valid_oh, y_test_oh = class_to_onehot(y_train, True), class_to_onehot(y_valid, True), class_to_onehot(
        y_test, True)

    N = x_train.shape[0]
    D = x_train.shape[1]
    C = y_train.max().add_(1).item()

    # check available devices
    print(torch.cuda.device_count())
    device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
    device_ids = list(range(torch.cuda.device_count()))

    # parse command line arguments
    p = argparse.ArgumentParser(description='Configuration parameters')

    p.add_argument('--v', default=1, dest='deep_version', type=int, help='Version of PTDeep used')
    p.add_argument('--prefix', default="", dest='run_name_prefix', type=str, help='Run name prefix')
    p.add_argument('--device', default=device, dest='device', type=torch.device, help='cuda or cpu')
    p.add_argument('--seed', default=72, dest='seed', type=int, help='Reproducibility seed')

    p.add_argument('--init', default="kaiming_uniform", dest='init', type=str, help='kaiming_uniform or normal')
    p.add_argument('--activation', default="relu", dest='activation', type=str, help='relu, sigmoid or tanh')
    p.add_argument('--optimizer', default="sgd", dest='optimizer', type=str, help='sgd, adam or adam+ls')

    p.add_argument('--batch_norm', default=False, dest='batch_norm', type=bool, help='Use batch norm (True/False)')
    p.add_argument('--batch_norm_epsilon', default=1e-5, dest='batch_norm_epsilon', type=float)
    p.add_argument('--batch_norm_momentum', default=0.9, dest='batch_norm_momentum', type=float)

    p.add_argument('--epochs', default=50000, dest='epochs', type=int, help='Maximum number of epochs')
    p.add_argument('--eta', default=0.1, dest='eta', type=float, help='The value of the learning rate')
    p.add_argument('--wd', default=1e-3, dest='weight_decay', type=float, help='L2 regularization (weight decay)')
    p.add_argument('--bs', default=N, dest='batch_size', type=int, help='Batch size')
    p.add_argument('--es', default=2000, dest='early_stopping', type=int, help='Early stopping')
    p.add_argument('--es_eps', default=1e-7, dest='early_stopping_epsilon', type=float, help='Early stopping epsilon')

    p.add_argument('--whiten_input', default=True, dest='whiten_input', type=bool, help='standardize inputs?')
    p.add_argument('--shuffle', default=False, dest='shuffle', type=bool, help='To shuffle train before each epoch?')
    p.add_argument('--npl', default=[D, C], dest='neurons_per_layer', nargs="+", type=int, help="Neurons per layer")

    p.add_argument('--log_interval', default=1, dest='log_interval', type=int, help='Log interval for train/valid loss')
    p.add_argument('--log_images_interval', default=4000, dest='log_images_interval', type=int,
                   help='Log interval for plots')
    p.add_argument('--show_plots', default=False, dest='show_plots', type=bool, help='Whether to show log plots')

    args = p.parse_args()
    params = dict(vars(args).items())
    print(device_ids)
    print("Using device", device)

    # reproducibility
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pp = pprint.PrettyPrinter(width=100, compact=True)

    # run_name is an identifier of the experiment
    # run_name = f"{params['run_name_prefix']}mnist" \
    #            f"_{get_str_formatted_time()}" \
    #            f"_w={1 if params['whiten_input'] else 0}" \
    #            f"_v={params['deep_version']}" \
    #            f"_i={params['init']}" \
    #            f"_npl={params['neurons_per_layer']}" \
    #            f"_act={params['activation']}" \
    #            f"_eta={params['eta']}" \
    #            f"_wd={params['weight_decay']}" \
    #            f"_epochs={params['epochs']}"
    run_name = f"{params['run_name_prefix']}" \
               f"opt={params['optimizer']}" \
               f"_act={params['activation']}" \
               f"_i={params['init']}" \
               f"_h={len(params['neurons_per_layer'])}" \
               f"_mb={params['batch_size']}" \
               f"_epochs={params['epochs']}" \
               f"_es={params['early_stopping']}" \
               f"_eta={params['eta']}" \
               f"_wd={params['weight_decay']}" \
               f"_s={params['seed']}"

    params["run_name"] = run_name
    params["save_dir"] = os.path.join(project_path, f"imgs/mnist_batchnorm2/{run_name}")
    ensure_dir(params["save_dir"])

    print("params:")
    pp.pprint(params)

    if params["whiten_input"]:
        train_mean, train_std = PTUtil.standardize_inplace(x_train, [x_valid, x_test])

    model = None
    if params["deep_version"] == 1:
        model = PTDeep(params["neurons_per_layer"], params["activation"], params["init"],
                       params["batch_norm"], x_train, params["batch_norm_epsilon"], params["batch_norm_momentum"])
    elif params["deep_version"] == 2:
        if params["batch_norm"]:
            raise RuntimeError("Not implemented for v2!")
        model = PTDeep2(params["neurons_per_layer"], params["activation"], params["init"])
    PTUtil.count_params(model)

    # train and eval
    train_and_eval(params, model, x_train, y_train, y_train_oh, x_valid, y_valid, y_valid_oh, x_test, y_test, y_test_oh,
                   device)
