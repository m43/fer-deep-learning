import torch


def accuracy(logits, target):
    assert len(logits) == len(target)

    with torch.no_grad():
        pred = logits.argmax(1)
        correct = (pred.to(target.dtype) == target).sum().item()

    return correct / len(target)


def accuracy_onehot(logits, onehot_target):
    assert len(logits) == len(onehot_target)

    with torch.no_grad():
        correct = (logits.argmax(1) == onehot_target.argmax(1)).sum().item()

    return correct / len(onehot_target)
