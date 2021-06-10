import argparse
import copy

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, RNN, GRU
from torch.utils.data import DataLoader

from demo.lab3.dataset import NLPDataset, load_sst_dataset, Word2Vec
from demo.lab3.rnn import MaRNN
from utils.util import setup_torch_reproducibility, setup_torch_device


def train_epoch(model, dataloader, optimizer, criterion, args):
    model.train()
    loss_sum = correct = total = 0
    for batch_num, (x, y, lengths) in enumerate(dataloader):
        model.zero_grad()
        logits = model(x, lengths).reshape_as(y)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        y_predicted = (logits > 0).clone().detach().type(torch.int).reshape(y.shape)
        correct += (y_predicted == y).sum()
        total += len(x)

        if batch_num % 100 == 0:
            print(f"{batch_num} --> Loss: {loss:3.5f}")
        loss_sum += loss
    print(f"TRAIN --> avg_loss={loss_sum / (batch_num + 1)}\tacc={correct / total}")


def evaluate(model, dataloader, criterion, args):
    model.eval()
    with torch.no_grad():
        losses = []
        tp = fp = tn = fn = 0
        for batch_num, (x, y, lengths) in enumerate(dataloader):
            logits = model(x, lengths).reshape_as(y)
            loss = criterion(logits, y)
            losses.append(loss)

            y_predicted = (logits > 0).clone().detach().type(torch.int).reshape(y.shape)
            assert list(y_predicted.shape) == list(y.shape)

            tp += torch.logical_and(y_predicted == y, y == 1).sum()
            fp += torch.logical_and(y_predicted != y, y == 0).sum()
            tn += torch.logical_and(y_predicted == y, y == 0).sum()
            fn += torch.logical_and(y_predicted != y, y == 1).sum()

        assert sum([tp, tn, fp, fn]) == len(dataloader.dataset)

        results = {}
        results["loss"] = sum(losses) / len(losses)
        results["acc"] = (tp + tn) / (tp + tn + fp + fn)
        results["pre"] = tp / (tp + fp)
        results["rec"] = tp / (tp + fn)
        results["f1"] = 2 * results["pre"] * results["rec"] / (results["pre"] + results["rec"])
        results["confmat"] = [[tp, fp], [fn, tn]]

    return results


def main(args, model_cls, word2vec_fn):
    setup_torch_reproducibility(args.seed)
    train_dataset, valid_dataset, test_dataset = load_sst_dataset()

    ordered_vocab_tokens = [token for _, token in sorted(train_dataset.text_vocab.itos.items())]
    pad_idx = train_dataset.text_vocab.get_pad_idx()
    word2vec = word2vec_fn(ordered_vocab_tokens, pad_idx)

    def collate_fn(x):
        x, y, z = NLPDataset.pad_collate_fn(x, pad_idx)
        return word2vec(x).to(args.device), y.to(args.device), z.to(args.device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = model_cls().to(args.device)

    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_model = best_loss = best_epoch = None
    for epoch in range(args.epochs):
        epoch += 1
        train_epoch(model, train_dataloader, optimizer, criterion, args)
        valid_results = evaluate(model, valid_dataloader, criterion, args)
        print(f"[{epoch}/{args.epochs}] VALID RESULTS\n{valid_results}")

        if best_loss is None or best_loss > valid_results["loss"] + args.early_stopping_epsilon:
            best_epoch = epoch
            best_loss = valid_results["loss"]
            best_model = model_cls()
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))

        if args.early_stopping_iters != -1 and epoch - best_epoch >= args.early_stopping_iters:
            print("EARLY STOPPING")
            model = best_model
            model.to(args.device)
            break

    test_results = evaluate(model, test_dataloader, criterion, args)
    print(f"[FINITO] TEST RESULTS\n{test_results}")
    return test_results


def print_results(results, path_to_save_csv="./output.txt"):
    print(f"# ~~~~~~ All results -- dict ~~~~~~ #\n{results}\n\n")

    print(f"# ~~~~~~ All results -- table with second moment measures ~~~~~~ #\n")
    metrics = list(results.items())[0][1][0].keys()
    csv_lines = []
    for metric in metrics:
        for model_name, values in results.items():
            if metric == "confmat": continue
            x = np.array([v[metric].item() for v in values])
            print(f"[{model_name}] [{metric.upper()}] avg:{x.mean()} std:{x.std()}")
            csv_lines.append(f"{model_name},{metric.upper()},{x.mean()},{x.std()}\n")

    print("CSV FORMAT")
    print("\n".join(csv_lines))

    with open(path_to_save_csv, "a") as csv:
        csv.writelines(csv_lines)
        csv.write("\n\n")


def create_wrapper_around_ma_rnn(rnn_cell_class, **kwargs):
    class RNNWrapper(MaRNN):
        def __init__(self):
            super().__init__(rnn_class=rnn_cell_class, **kwargs)

        def forward(self, x, l):
            return super().forward(x.transpose(1, 0), l)

    return RNNWrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--max_vocab_size", type=int, default=-1)
    # parser.add_argument("--min_token_freq_in_vocab", type=int, default=0)
    parser.add_argument("--seed", "-s", type=int, default=7052020)
    parser.add_argument("--device", default=setup_torch_device())
    parser.add_argument("--n_runs", "-n", type=int, default=10)
    parser.add_argument("--batch_size", "-bs", type=int, default=10)
    parser.add_argument("--epochs", "-e", type=int, default=300)
    # parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--clip", "-c", type=float, default=0.25)
    parser.add_argument("--early_stopping_epsilon", "-ese", type=float, default=1e-7)
    parser.add_argument("--early_stopping_iters", "-esi", type=int, default=20)

    args = parser.parse_args(args=[])
    print(f"{args}\n\n")

    # Hyperparameter search config
    # Setup parameters
    max_vocab_size_list = [-1, 100, 200, 1000]
    min_token_freq_in_vocab_list = [0, 20, 200, 2000]
    learning_rate_list = [1e-4]  # [1e-4, 1, 1e-2, 1e-5]
    word2vec_fn_list = [Word2Vec.load_glove]  # [Word2Vec.load_glove, Word2Vec.generate_random_gauss_matrix]
    # RNN cell parameters
    dropout_values_list = [0]  # [0, 0.5, 0.9]
    n_layers_list = [2]  # [2, 3, 5, 20]
    hid_dim_list = [150]  # [150, 50, 300, 1000]
    rnm_cell_list = [("GRU", GRU)]  # [("GRU", GRU), ("LSTM", LSTM), ("RNN", RNN)]
    models_to_test = []  # + [("BASELINE", Baseline)]
    for rnn_cell_name, rnn_cell in rnm_cell_list:
        for n_layers in n_layers_list:
            for hid_dim in hid_dim_list:
                for dropout_value in dropout_values_list:
                    model_name = f"{rnn_cell_name}" \
                                 f"_n={n_layers}" \
                                 f"_h={hid_dim}" \
                                 f"_d={dropout_value}"
                    wrapped_model_class = create_wrapper_around_ma_rnn(rnn_cell, hid_dim_1=hid_dim, hid_dim_2=hid_dim,
                                                                       num_layers=n_layers, dropout=dropout_value)
                    models_to_test.append((model_name, wrapped_model_class))

    for max_vocab_size in max_vocab_size_list:
        for min_token_freq_in_vocab in min_token_freq_in_vocab_list:
            for learning_rate in learning_rate_list:
                for word2vec_fn in word2vec_fn_list:
                    args.max_vocab_size = max_vocab_size
                    args.min_token_freq_in_vocab = min_token_freq_in_vocab
                    args.learning_rate = learning_rate

                    test_results = {}
                    for model_name, model_class in models_to_test:
                        model_name = f"{model_name}" \
                                     f"__mvs={max_vocab_size}" \
                                     f"_mtiv={min_token_freq_in_vocab}" \
                                     f"_lr={learning_rate}" \
                                     f"_w2v={'g' if word2vec_fn == Word2Vec.load_glove else 'r'}"
                        for i in range(args.n_runs):
                            print(f"# ~~~~~~ {model_name} -- run {i} ~~~~~~ #")
                            args.seed = i
                            assert args.seed == i
                            results = main(args, model_class, word2vec_fn)
                            test_results[model_name] = test_results.get(model_name, []) + [results]
                            print("\n\n")
                        print("TEMP RESULTS (in case of sudden halt)")
                        print_results(test_results)

    print("All done. Final results below")
    print_results(test_results)
