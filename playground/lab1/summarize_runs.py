import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import zipfile
from matplotlib.ticker import StrMethodFormatter
from pprint import PrettyPrinter
from shutil import copyfile

from demo.lab1.util import PTUtil
from model.pt_deep import PTDeep
from utils.data import class_to_onehot
from utils.util import ensure_dir, project_path

device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")

(x_train, y_train,), (x_valid, y_valid), (x_test, y_test) = PTUtil.load_mnist()
x_train, x_valid, x_test = x_train.flatten(-2), x_valid.flatten(-2), x_test.flatten(-2)
y_train_oh, y_valid_oh, y_test_oh = class_to_onehot(y_train, True), class_to_onehot(y_valid, True), class_to_onehot(
    y_test, True)

x_train, y_train, y_train_oh = x_train.to(device), y_train.to(device), y_train_oh.to(device)
x_valid, y_valid, y_valid_oh = x_valid.to(device), y_valid.to(device), y_valid_oh.to(device)
x_test, y_test, y_test_oh = x_test.to(device), y_test.to(device), y_test_oh.to(device)
PTUtil.standardize_inplace(x_train, [x_valid, x_test])

results = []
# mnist_results_path = "/home/user72/Desktop/imgs/mnist/mnist_best/"
mnist_results_path = "/home/user72/Desktop/imgs/mnist/mnist_batchnorm2/"
# mnist_results_path = os.path.join(project_path, "imgs/mnist_best")
results_path = os.path.join(project_path, "ma_results_beest")
ensure_dir(results_path)
for folder in os.listdir(mnist_results_path):
    print(folder)
    folder_path = os.path.join(mnist_results_path, folder)

    # if not folder.startswith("act=relu") and not folder.startswith("relu"):
    #     print("meh, next")
    #     continue

    with open(os.path.join(folder_path, "params.txt"), "r") as f:
        x = "".join(f.read().split("\n")).strip()
        x = x.replace("""'device': device(type='cuda'),""", "")
        x = x.replace("""'device': device(type='cpu'),""", "")
        # x = x.replace("""'neurons_per_layer': [784, 10],""", "")
        # x = x.replace("""'neurons_per_layer': [784, 100, 10],""", "")
        # x = x.replace("""'neurons_per_layer': [784, 100, 100, 10],""", "")

        x = x.split("}{")[-1]
        if not x.startswith("{"):
            x = "{" + x

        params = eval(x)

    # parts = folder.replace("-force", "").split("-", maxsplit=3)
    # params = {
    #     "activation": parts[0],
    #     "deep_version": parts[1][1:],
    #     "init": "normal" if parts[2] == "normal" else "kaiming_uniform",
    #     "weight_decay": float(parts[3].split("=")[1]),
    #     "neurons_per_layer": [784,10],
    # }

    with open(os.path.join(folder_path, "results.txt"), "r") as f:
        from numpy import array, int64, nan

        ae = array, int64, nan
        all_results = eval(f.read())

    # model = torch.load(os.path.join(folder_path, "model_best.pth"))
    # model = PTDeep(params["neurons_per_layer"], params["activation"], params["init"])
    model = PTDeep(params["neurons_per_layer"], params["activation"], params["init"],
                   params["batch_norm"], x_train, params["batch_norm_epsilon"], params["batch_norm_momentum"])
    model.load_state_dict(
        torch.load(os.path.join(folder_path, "model_best_state_dict.pth"), map_location=torch.device(device)))
    model.to(device)
    # model.eval()

    current_results = {
        "best_epoch": all_results["best_epoch"],
        # "best_epoch": -1,
    }
    current_results.update(
        PTUtil.get_perf_multi(model, params["weight_decay"], x_train, y_train, y_train_oh, "best_train_", False))
    current_results.update(
        PTUtil.get_perf_multi(model, params["weight_decay"], x_valid, y_valid, y_valid_oh, "best_valid_", False))
    current_results.update(
        PTUtil.get_perf_multi(model, params["weight_decay"], x_test, y_test, y_test_oh, "best_test_", False))

    results.append((folder, current_results))

    copyfile(f"{os.path.join(folder_path, 'loss.png')}",
             f"{os.path.join(results_path, 'loss_' + folder + '.png')}")

    # copyfile(f"{os.path.join(folder_path, 'accuracy.png')}",
    #          f"{os.path.join(results_path, 'accuracy_' + folder + '.png')}")

    plt.figure(figsize=(6.4 * 1.26, 4.8 * 1.26))
    plt.title("Loss over epoch")
    plt.xlabel("epoch")
    plt.ylabel("log loss")
    plt.plot(all_results["i"][2:], np.log(np.array(all_results["train_loss"][2:])), color="green", label="train")
    plt.plot(all_results["i"][2:], np.log(np.array(all_results["valid_loss"][2:])), color="red", label="valid")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.3e}'))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{os.path.join(results_path, 'loss2_' + folder + '.png')}", dpi=300)
    plt.close()

    plt.title("Accuracy over epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0, 100)
    plt.plot(all_results["i"][1:], np.array(all_results["train_accuracy"][1:]) * 100, color="green", label="train")
    plt.plot(all_results["i"][1:], np.array(all_results["valid_accuracy"][1:]) * 100, color="red", label="valid")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{os.path.join(results_path, 'accuracy2_' + folder + '.png')}", dpi=300)
    plt.close()

pp = PrettyPrinter(width=100, compact=True)
pp.pprint(results)
with open(f"{os.path.join(results_path, 'ma_mnist_results_pp.txt')}", "w") as f:
    f.write(pp.pformat(results))
    f.write("\n")
with open(f"{os.path.join(results_path, 'ma_mnist_results.txt')}", "w") as f:
    f.write(f"{results}")
    f.write("\n")

group_same_seed = {}
for name, result in results:
    group = name.split("_s=", maxsplit=2)[0]
    if group not in group_same_seed:
        group_same_seed[group] = []
    group_same_seed[group].append(result)

group_statistics = {}
for group, group_results in group_same_seed.items():
    group_statistics[group] = {}
    for k in group_results[0]:
        values = np.array([gr[k] for gr in group_results])
        group_statistics[group][k] = values.mean(), values.std()

with open(f"{os.path.join(results_path, 'ma_mnist_results.csv')}", "w") as csv_file:
    header = False
    for g, gs in group_statistics.items():
        if not header:
            header = True
            csv_file.write("run")
            for k in gs.keys():
                csv_file.write(f",{k} mean,{k} std")
            csv_file.write("\n")

        csv_file.write(f"{g}")
        for k in gs.keys():
            csv_file.write(f",{gs[k][0]},{gs[k][1]}")
        csv_file.write("\n")
with open(f"{os.path.join(results_path, 'ma_mnist_results.csv')}", "r") as f:
    print(f.read())

interesting_keys = ["best_epoch", "best_test_accuracy", "best_test_loss", "best_test_precision", "best_test_recall",
                    "best_train_accuracy", "best_train_loss"]
with open(f"{os.path.join(results_path, 'ma_interesting_mnist_results.csv')}", "w") as csv_file:
    header = False
    for g, gs in group_statistics.items():
        if not header:
            header = True
            csv_file.write("run")
            for k in interesting_keys:
                csv_file.write(f",{k} mean,{k} std")
            csv_file.write("\n")

        csv_file.write(f"{g}")
        for k in interesting_keys:
            csv_file.write(f",{gs[k][0]},{gs[k][1]}")
        csv_file.write("\n")
with open(f"{os.path.join(results_path, 'ma_mnist_results.csv')}", "r") as f:
    print(f.read())


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path,
                                                                                                        '../../demo')))


zipf = zipfile.ZipFile(results_path + ".zip", 'w', zipfile.ZIP_DEFLATED)
zipdir(results_path, zipf)
zipf.close()
