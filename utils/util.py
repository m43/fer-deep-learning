import numpy as np
import pathlib
from datetime import datetime

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
    return datetime.now().strftime('%Y-%m-%d--%H.%M.%S')


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def ensure_dirs(dirs):
    for dir_ in dirs:
        ensure_dir(dir_)


def sigmoid(x, beta=1):
    return 1. / (1. + np.exp(-beta * x))
