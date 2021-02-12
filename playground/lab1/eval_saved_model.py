import os
import zipfile
from pprint import PrettyPrinter
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import StrMethodFormatter

from demo.util import PTUtil
from utils.data import class_to_onehot
from utils.util import ensure_dir, zipdir, project_path
device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")

(x_train, y_train,), (x_valid, y_valid), (x_test, y_test) = PTUtil.load_mnist()
x_train, x_valid, x_test = x_train.flatten(-2), x_valid.flatten(-2), x_test.flatten(-2)
y_train_oh, y_valid_oh, y_test_oh = class_to_onehot(y_train, True), class_to_onehot(y_valid, True), class_to_onehot(
    y_test, True)

x_train, y_train, y_train_oh = x_train.to(device), y_train.to(device), y_train_oh.to(device)
x_valid, y_valid, y_valid_oh = x_valid.to(device), y_valid.to(device), y_valid_oh.to(device)
x_test, y_test, y_test_oh = x_test.to(device), y_test.to(device), y_test_oh.to(device)
PTUtil.standardize_inplace(x_train, [x_valid, x_test])

wd = "0.0"
# wd="0.001"
# relative_model_path = f"imgs/mnist_best/force_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=300000_es=1000000000_eta=0.1_wd={wd}_s=360"
relative_model_path = f"visuals/lab1/7_mnist_force/force_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=300000_es=1000000000_eta=0.1_wd={wd}_s=72"
model_location = os.path.join(project_path, relative_model_path)
results_path = os.path.join(project_path, f"ma_results_force/wd={wd}")
ensure_dir(results_path)

with open(os.path.join(model_location, "params.txt"), "r") as f:
    x = "".join(f.read().split("\n")).strip()
    x = x.replace("""'device': device(type='cuda'),""", "")
    # x = x.replace("""'neurons_per_layer': [784, 10],""", "")
    # x = x.replace("""'neurons_per_layer': [784, 100, 10],""", "")s
    # x = x.replace("""'neurons_per_layer': [784, 100, 100, 10],""", "")

    x = x.split("}{")[-1]
    if not x.startswith("{"):
        x = "{" + x

    params = eval(x)
    print(params)

with open(os.path.join(model_location, "results.txt"), "r") as f:
    from numpy import array, int64, nan

    ae = array, int64, nan
    all_results = eval(f.read())

model_best = torch.load(os.path.join(model_location, "model_best_cpu.pth"))
model_best.to(device)
model_best.eval()

model_end = torch.load(os.path.join(model_location, "model_end_cpu.pth"))
model_end.to(device)
model_end.eval()

wd = params["weight_decay"]
results = {
    "best_epoch": all_results["best_epoch"],
}
results.update(PTUtil.get_perf_multi(model_best, wd, x_train, y_train, y_train_oh, "best_train_", False))
results.update(PTUtil.get_perf_multi(model_best, wd, x_valid, y_valid, y_valid_oh, "best_valid_", False))
results.update(PTUtil.get_perf_multi(model_best, wd, x_test, y_test, y_test_oh, "best_test_", False))
results.update(PTUtil.get_perf_multi(model_end, wd, x_train, y_train, y_train_oh, "end_train_", False))
results.update(PTUtil.get_perf_multi(model_end, wd, x_valid, y_valid, y_valid_oh, "end_valid_", False))
results.update(PTUtil.get_perf_multi(model_end, wd, x_test, y_test, y_test_oh, "end_test_", False))

pp = PrettyPrinter(width=100, compact=True)
pp.pprint(results)

copyfile(f"{os.path.join(model_location, 'loss.png')}",
         f"{os.path.join(results_path, 'loss.png')}")

# copyfile(f"{os.path.join(model_location, 'accuracy.png')}",
#          f"{os.path.join(results_path, 'accuracy.png')}")

plt.figure(figsize=(6.4 * 1.3, 4.8 * 1.26))
plt.title("Log loss over epoch")
plt.xlabel("epoch")
plt.ylabel("log loss")
plt.plot(all_results["i"][2:], np.log(np.array(all_results["train_loss"][2:])), color="green", label="train")
plt.plot(all_results["i"][2:], np.log(np.array(all_results["valid_loss"][2:])), color="red", label="valid")
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.3e}'))
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(f"{os.path.join(results_path, 'loss2.png')}", dpi=300)
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
plt.savefig(f"{os.path.join(results_path, 'accuracy2.png')}", dpi=300)
plt.close()

with open(os.path.join(results_path, "performance.txt"), "w") as f:
    f.write(pp.pformat(results))
    f.write("\n")

zipf = zipfile.ZipFile(results_path + ".zip", 'w', zipfile.ZIP_DEFLATED)
zipdir(results_path, zipf)
zipf.close()
