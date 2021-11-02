import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib import rcParams
from torch import softmax
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.utils import make_grid

from demo.lab2.util import load_cifar10
from model.conv_model import SimpleConvolutionalModel, CifarSimpleConvolutionalModel
from trainer import SimpleConvTrainer
from utils.metric import accuracy
from utils.util import project_path, ensure_dirs, setup_torch_reproducibility, setup_torch_device


def loss(logits, y):
    logits = logits - logits.max()
    lse = torch.logsumexp(logits)
    correct = logits[y]
    return lse - correct


config = {}
config['max_epochs'] = 300
config['batch_size'] = 50
config['seed'] = 72  # np.random.seed(int(time.time() * 1e6) % 2 ** 31)
# config['weight_decay'] = 0
config['weight_decay'] = 1e-4
config['conv1_channels'] = 16
config['conv2_channels'] = 32
config['fc1_width'] = 256
config['fc2_width'] = 128
config['lr'] = 1e-3/2
config["lr_scheduler_decay"] = 0.95
config['log_step'] = 100
config['val_split'] = 0.9
config['run_name'] = f"cifar10_conv_" \
                     f"_e={config['max_epochs']}" \
                     f"_bs={config['batch_size']}" \
                     f"_wd={config['weight_decay']}" \
                     f"_lr={config['lr']}" \
                     f"_lr_scheduler_decay={config['lr_scheduler_decay']}" \
                     f"_seed={config['seed']}"

DATA_DIR = os.path.join(project_path, 'datasets')
SAVE_DIR = os.path.join(project_path, f'saved')
ensure_dirs([DATA_DIR, SAVE_DIR])
print(f"DATA_DIR:{DATA_DIR}\nSAVE_DIR:{SAVE_DIR}\n")
config['save_dir'] = SAVE_DIR

setup_torch_reproducibility(config["seed"])
device = setup_torch_device()

train, test, valid = load_cifar10(DATA_DIR, config['val_split'])
train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(valid, batch_size=config['batch_size'])
test_loader = DataLoader(test, batch_size=config['batch_size'])

# net = SimpleConvolutionalModel(
#     in_channels=3,
#     image_height=32,
#     image_width=32,
#     conv1_channels=config['conv1_channels'],
#     conv2_channels=config['conv2_channels'],
#     fc1_width=config['fc1_width'],
#     # fc2_width=config['fc2_width'],
#     class_count=10
# )
net = CifarSimpleConvolutionalModel(
    in_channels=3,
    image_height=32,
    image_width=32,
    conv1_channels=config['conv1_channels'],
    conv2_channels=config['conv2_channels'],
    fc1_width=config['fc1_width'],
    fc2_width=config['fc2_width'],
    class_count=10
)
print("summary for model as floats:")
summary(net, input_size=train[0][0].shape, device=device.type)
print(f"CONFIG:{config}")

criterion = torch.nn.CrossEntropyLoss(reduction='mean')  # mean is default anyways
optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config["lr_scheduler_decay"])
metric_ftns = [accuracy, ]

trainer = SimpleConvTrainer(
    run_name=config['run_name'],
    model=net,
    criterion=criterion,
    metric_ftns=metric_ftns,
    optimizer=optimizer,
    device=device,
    device_ids=[],
    epochs=config['max_epochs'],
    save_folder=config['save_dir'],
    monitor="max val accuracy",
    log_step=config['log_step'],
    early_stopping=100000000,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    lr_scheduler=lr_scheduler
)
trainer.train_epoch()


def highest_loss_images(top_losses, top_images, top_probs, top_targets, n, m, save_dir, prefix=""):
    plt.style.use('ggplot')
    grid = (n, m)

    def plt_grid(i, ax):

        loss_i = top_losses[i]
        probs_i = top_probs[i].tolist()
        y_i = top_targets[i]
        x_i = top_images[i]

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
            # tick_label=np.arange(10),
            tick_label=["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
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

        # im = x_i.reshape((28, 28))
        # fig.figimage(im, 0, fig.bbox.ymax - 28, resize=2)

    def plt_corner(i, fig, ax):
        # im = top_images[i].reshape((28, 28))
        im = top_images[i]

        ax_pos = ax.get_position()
        corner_ax_rect = [ax_pos.x0 + 0.8 * ax_pos.width, ax_pos.y0 + 0.8 * ax_pos.height, ax_pos.width * 0.2,
                          ax_pos.height * 0.2]

        newax = fig.add_axes(corner_ax_rect, anchor='NE', zorder=0)  # [x0, y0, width, height]
        newax.imshow(im.permute(1, 2, 0))
        newax.axis('off')

    for i in range(len(top_losses)):
        loss_i = top_losses[i]
        y_i = top_targets[i]

        fig, ax = plt.subplots()
        plt_grid(i, ax)
        fig.tight_layout()
        plt_corner(i, fig, ax)
        plt.savefig(
            os.path.join(save_dir, f"{prefix}{i:d}_i={i:d}_loss={loss_i:.6f}_true={y_i}.png"), dpi=150)
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
    grid_location = os.path.join(save_dir, f"{prefix}grid.png")
    plt.savefig(grid_location, dpi=150)
    plt.close()

    return grid_location


def top_NxM_pictures_with_highest_loss(name, dataloder, n, m, save_dir, mean=None, std=None):
    n_total_pics = n * m

    top_losses = []
    top_images = []
    top_probs = []
    top_targets = []

    trainer.model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloder):
            data, target = data.to(trainer.device), target.to(trainer.device)

            for i in range(len(data)):
                output_i = trainer.model(data[i:i + 1])
                loss = trainer.criterion(output_i, target[i:i + 1])
                probs_i = softmax(output_i[0], -1) * 100

                if len(top_losses) < n_total_pics:
                    top_losses.append(loss)
                    top_images.append(data[i])
                    top_probs.append(probs_i)
                    top_targets.append(target[i])

                min_idx = top_losses.index(min(top_losses))
                if loss >= top_losses[min_idx]:
                    top_losses[min_idx] = loss
                    top_images[min_idx] = data[i]
                    top_probs[min_idx] = probs_i
                    top_targets[min_idx] = target[i]

                # break

    top_images = [x for _, x in sorted(zip(top_losses, top_images), key=lambda x: -x[0])]
    top_probs = [x for _, x in sorted(zip(top_losses, top_probs), key=lambda x: -x[0])]
    top_targets = [x for _, x in sorted(zip(top_losses, top_targets), key=lambda x: -x[0])]
    top_losses = sorted(top_losses, key=lambda x: -x)

    if mean is not None and std is not None:
        for i in range(len(top_images)):
            top_images[i] *= std[:, None, None]
            top_images[i] += mean[:, None, None]

    trainer.writer.add_image(
        f"{name} -- hardest examples (at end) uint",
        make_grid(top_images, nrow=m, normalize=True, padding=2), 0
    )

    highest_loss_images(top_losses, top_images, top_probs, top_targets, n, m, save_dir, prefix=name)


norm_transform = trainer.train_loader.dataset.dataset.transform.transforms[-1]
mean, std = norm_transform.mean, norm_transform.std
top_NxM_pictures_with_highest_loss(trainer.test_metrics.get_name(), trainer.test_loader, 4, 4, trainer.logs_folder,
                                   mean, std)
top_NxM_pictures_with_highest_loss(trainer.valid_metrics.get_name(), trainer.val_loader, 4, 4, trainer.logs_folder,
                                   mean, std)
top_NxM_pictures_with_highest_loss(trainer.train_metrics.get_name(), trainer.train_loader, 4, 4, trainer.logs_folder,
                                   mean, std)

print(trainer.logs_folder)
