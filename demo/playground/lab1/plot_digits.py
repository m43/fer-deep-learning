import matplotlib.pyplot as plt
import os
import torch

from utils.util import project_path, ensure_dir

v = 1
act = "relu"
npl = [784, 10]

model_path = "imgs/mnist/mnist_2021.02.05_20.46.43_w=1_v=1_i=normal_npl=[784, 10]_act=relu_eta=0.1_wd=0.0_epochs=100000/model_best.pth"
model_path = os.path.join(project_path, model_path)
plots_path = os.path.join(project_path, "imgs/mnist/layers/ae_relu_init=noraml_uniform_wd=0")
ensure_dir(plots_path)

model = torch.load(model_path)
model.to("cpu")

w = None
if v == 1:
    w = model.weights[0].T
    w = w.detach()
elif v == 2:
    w = model.seq.ll_1.weight.data

for i in range(10):
    title = f"weights for digit {i}, npl:{npl}"
    print(title)
    plt.title(title)
    plt.imshow(w[i].reshape((28, 28)), cmap=plt.get_cmap('gray'))
    plt.savefig(os.path.join(plots_path, f"{i}.png"), dpi=300)
    plt.close()

    title = f"weights for digit {i}, npl:{npl}"
    print(title)
    plt.title(title)
    plt.imshow(w[i].reshape((28, 28)))
    plt.savefig(os.path.join(plots_path, f"g_{i}.png"), dpi=300)
    plt.close()

# (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()
# train_mean, train_std = standardize_inplace(x_train, [x_valid, x_test])
#
# for i in range(50):
#     plt.title(f"i:{i} digit {y_train[i]}")
#     plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
#     plt.savefig(os.path.join(plots_path, f"s_MNIST_TRAIN_{i}.png"), dpi=300)
#     # plt.show()
#     plt.close()
