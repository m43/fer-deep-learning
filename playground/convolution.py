import numpy as np
import os
from matplotlib import rcParams, pyplot as plt

from utils.util import project_path

COLORS_MAP = {
    "green": "#2ecc71",
    "cyan": "#00c4cc",
    "red": "#e24a33",
    "orange": "#f39c12",
    "yellow": "#f1c40f", }

np.random.seed(72)

x = np.linspace(0, 24, 1000)
y = np.sin(x) + 2. * (x > np.pi * 4.5)
yn = y + np.random.normal(0, 0.1, x.shape)

relu = lambda x: np.maximum(x, 0)

w = 0.05 * (relu(x) - relu(2 * (x - 0.5)) + relu((x - 1)))
# w_reversed = w[::-1]
# print(f"w_reversed.base is w: {w_reversed.base is w}")  # True

ma_cross_correlation = lambda x, y: np.array(
    [np.sum([x[tau] * y[(t + tau) % len(x)] for tau in range(len(x))]) for t in range(len(x))])
h = ma_cross_correlation(w, yn)

g = ma_cross_correlation(yn, w)[::-1]
# g = np.convolve(yn, w, "full")  # TODO kako ovako dobiti?

rows = 5
figsize = rcParams["figure.figsize"]
figsize = (figsize[0] * 1, figsize[1] * rows / 2)

fig, axs = plt.subplots(rows, 1, figsize=figsize)
axs[0].plot(x, y, c=COLORS_MAP["green"])
axs[1].plot(x, yn, c=COLORS_MAP["red"])
axs[2].plot(x, w, c=COLORS_MAP["cyan"])
axs[3].plot(x, h, c=COLORS_MAP["yellow"])
axs[4].plot(np.arange(g.shape[0]), g, c=COLORS_MAP["yellow"])

fig.savefig(os.path.join(project_path, "imgs/laser_beam_1.png"), dpi=300)
fig.tight_layout()
fig.savefig(os.path.join(project_path, "imgs/laser_beam_2.png"), dpi=300)
plt.show()
plt.close()
