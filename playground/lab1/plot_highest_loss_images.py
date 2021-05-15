import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import zipfile
from matplotlib import rcParams
from matplotlib.ticker import StrMethodFormatter
from pprint import PrettyPrinter
from shutil import copyfile

from demo.lab1.util import PTUtil
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

model_location = os.path.join(project_path, "visuals/lab1/7_mnist__best_model")
results_path = os.path.join(project_path, "ma_results_best")
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

model = torch.load(os.path.join(model_location, "model_best_cpu.pth"))
# model = PTDeep(params["neurons_per_layer"], params["activation"], params["init"])
# model.load_state_dict(torch.load(os.path.join(model_location, "model_best_state_dict.pth"), map_location=torch.device(device)))
model.to(device)
model.eval()

results = {
    "best_epoch": all_results["best_epoch"],
    # "best_epoch": -1,
}
results.update(PTUtil.get_perf_multi(model, params["weight_decay"], x_train, y_train, y_train_oh, "best_train_", False))
results.update(PTUtil.get_perf_multi(model, params["weight_decay"], x_valid, y_valid, y_valid_oh, "best_valid_", False))
results.update(PTUtil.get_perf_multi(model, params["weight_decay"], x_test, y_test, y_test_oh, "best_test_", False))
pp = PrettyPrinter(width=100, compact=True)
pp.pprint(results)

copyfile(f"{os.path.join(model_location, 'loss.png')}",
         f"{os.path.join(results_path, 'loss.png')}")

copyfile(f"{os.path.join(model_location, 'accuracy.png')}",
         f"{os.path.join(results_path, 'accuracy.png')}")

plt.figure(figsize=(6.4 * 1.26, 4.8 * 1.26))
plt.title("Loss over epoch")
plt.xlabel("epoch")
plt.ylabel("log loss")
plt.plot(all_results["i"][2:], np.log(np.array(all_results["train_loss"][2:])), color="green", label="train")
plt.plot(all_results["i"][2:], np.log(np.array(all_results["valid_loss"][2:])), color="red", label="valid")
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.3e}'))
plt.legend(loc="best")
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
plt.savefig(f"{os.path.join(results_path, 'accuracy2.png')}", dpi=300)
plt.close()


def highest_loss_images(model, weight_decay, x, y, y_oh, prefix="", grid=(4, 4)):
    plt.style.use('ggplot')

    losses = [(i, model.get_loss(x[i:i + 1], y_oh[i:i + 1], weight_decay).item()) for i in range(len(x))]
    losses.sort(key=lambda x: -x[1])

    top_losses = losses[:grid[0] * grid[1]]
    top_indices = [x[0] for x in top_losses]
    top_probs = model.forward(x[top_indices]).clone().detach().cpu() * 100

    def plt_grid(i, ax):
        index_i = top_indices[i]

        loss_i = top_losses[i][1]
        probs_i = top_probs[i].tolist()
        y_i = y[index_i]
        x_i = x[index_i]

        # prepare colors for each digit
        colors_map = {"green": "#2ecc71", "cyan": "#00c4cc", "red": "#e24a33", "orange": "#f39c12"}
        colors = [colors_map["orange"]] * 10
        guess = np.argmax(probs_i).item()
        colors[y_i] = colors_map["green"]
        if guess != y_i:
            colors[guess] = colors_map["red"]

        # Save the chart so we can loop through the bars below.
        bars = ax.bar(
            x=np.arange(10),
            height=probs_i,
            tick_label=np.arange(10),
            color=colors
        )
        ax.set_ylim([0, 100])

        # Add text annotations to the top of the bars.
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                round(bar.get_height(), 1),
                horizontalalignment='center',
                color=bar.get_facecolor(),
                weight='bold'
            )

        # Add labels and a title. Note the use of `labelpad` and `pad` to add some
        # extra space between the text and the tick labels.
        ax.set_xlabel('Digit', labelpad=5, color='#333333')
        ax.set_ylabel('Probability given by model', labelpad=15, color='#333333')
        ax.set_title(f'{prefix.replace("_", ":").title()} True digit = {y_i}, Loss = {loss_i:.6f}', pad=15,
                     color='#333333', weight='bold')

        im = x[index_i].reshape((28, 28))
        # fig.figimage(im, 0, fig.bbox.ymax - 28, resize=2)

    def plt_corner(i, fig, ax):
        index_i = top_indices[i]

        im = x[index_i].reshape((28, 28))

        ax_pos = ax.get_position()
        corner_ax_rect = [ax_pos.x0 + 0.8 * ax_pos.width, ax_pos.y0 + 0.8 * ax_pos.height, ax_pos.width * 0.2,
                          ax_pos.height * 0.2]

        newax = fig.add_axes(corner_ax_rect, anchor='NE', zorder=0)  # [x0, y0, width, height]
        newax.imshow(im)
        newax.axis('off')

    for i in range(len(top_losses)):
        index_i = top_indices[i]
        loss_i = top_losses[i][1]
        y_i = y[index_i]

        fig, ax = plt.subplots()
        plt_grid(i, ax)
        fig.tight_layout()
        plt_corner(i, fig, ax)
        plt.savefig(
            os.path.join(results_path, f"{prefix}{i:d}_i={index_i:d}_loss={loss_i:.6f}_true={y_i}.png"), dpi=150)
        plt.close()

    figsize = rcParams["figure.figsize"]
    figsize = (figsize[0] * grid[0], figsize[1] * grid[1])
    fig, axs = plt.subplots(grid[0], grid[1], figsize=figsize)
    for i in range(len(top_losses)):
        plt_grid(i, axs[i // grid[1], i % grid[1]])
    fig.tight_layout()
    for i in range(len(top_losses)):
        plt_corner(i, fig, axs[i // grid[1], i % grid[1]])
    # fig.tight_layout()
    plt.savefig(os.path.join(results_path, f"{prefix}grid.png"), dpi=150)
    plt.close()


highest_loss_images(model, params["weight_decay"], x_train, y_train, y_train_oh, "train_")
highest_loss_images(model, params["weight_decay"], x_valid, y_valid, y_valid_oh, "valid_")
highest_loss_images(model, params["weight_decay"], x_test, y_test, y_test_oh, "test_")

zipf = zipfile.ZipFile(results_path + ".zip", 'w', zipfile.ZIP_DEFLATED)
zipdir(results_path, zipf)
zipf.close()
