import os
import pathlib
from datetime import datetime

import numpy as np

horse = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;;;;;'. ;   |          )   \\ | ;;;;;;
          ,;*;;;;\\/   |.        /   /` | ';;;*;
           ;;;;;;/    |/       /   /__/   ';;;
           '*jgs/     |       /    |      ;*;
                `""""`        `""""`     ;'"""  # Why jgs?

project_path = pathlib.Path(__file__).parent.parent


def get_str_formatted_time() -> str:
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def ensure_dirs(dirs):
    for dir_ in dirs:
        ensure_dir(dir_)


def sigmoid(x, beta=1):
    return 1. / (1. + np.exp(-beta * x))


def softmax(X):
    X_stable = X - np.expand_dims(X.max(axis=-1), axis=-1)
    exp_X_stable = np.exp(X_stable)
    result = exp_X_stable / np.expand_dims(exp_X_stable.sum(axis=-1), axis=-1)
    return result


def relu(x):
    return np.maximum(0, x)


def zipdir(path, ziph):
    """
    Usage example:
    zipf = zipfile.ZipFile(results_path + ".zip", 'w', zipfile.ZIP_DEFLATED)
    zipdir(results_path, zipf)
    zipf.close()

    Source: https://stackoverflow.com/questions/41430417/using-zipfile-to-create-an-archive

    :param path: Path to dir to zip
    :param ziph: zipfile handle
    :return:
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))
