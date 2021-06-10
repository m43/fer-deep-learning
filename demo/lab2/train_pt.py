import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

from demo.lab2.util import load_mnist
from model.conv_model import SimpleConvolutionalModel
from trainer.trainer import SimpleConvTrainer
from utils.metric import accuracy
from utils.util import project_path, ensure_dirs, setup_torch_reproducibility, setup_torch_device

for wd in [0, 0.001, 0.01, 0.1]:
    config = {}
    config['max_epochs'] = 8
    config['batch_size'] = 50
    config['seed'] = 72  # np.random.seed(int(time.time() * 1e6) % 2 ** 31)
    config['weight_decay'] = wd
    config['run_name'] = f"test2____pt_conv_" \
                         f"_e={config['max_epochs']}" \
                         f"_bs={config['batch_size']}" \
                         f"_wd={config['weight_decay']}" \
                         f"_seed={config['seed']}"
    config['conv1_channels'] = 16
    config['conv2_channels'] = 32
    config['fc1_width'] = 512
    config['lr'] = 1e-1
    config['lr_scheduler_step'] = 2
    config['lr_scheduler_decay'] = 0.1
    config['log_step'] = 100

    DATA_DIR = os.path.join(project_path, 'datasets')
    SAVE_DIR = os.path.join(project_path, f'saved')
    ensure_dirs([DATA_DIR, SAVE_DIR])
    print(f"DATA_DIR:{DATA_DIR}\nSAVE_DIR:{SAVE_DIR}\n")
    config['save_dir'] = SAVE_DIR

    setup_torch_reproducibility(config["seed"])
    device = setup_torch_device(True)

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_mnist(DATA_DIR)
    train_loader, val_loader, test_loader = [
        DataLoader(
            TensorDataset(torch.tensor(x, dtype=torch.double), torch.tensor(y.argmax(1), dtype=torch.long)),
            batch_size=config['batch_size'], shuffle=shuffle
        )
        for x, y, shuffle in [(train_x, train_y, True), (valid_x, valid_y, False), (test_x, test_y, False)]
    ]

    net = SimpleConvolutionalModel(1, train_x[0].shape[-2], train_x[0].shape[-1], config['conv1_channels'],
                                   config['conv2_channels'],
                                   config['fc1_width'], class_count=train_y.shape[-1])
    print("summary for model as floats:")
    summary(net, input_size=train_x[0].shape, device=device.type)
    net.double()
    print(f"CONFIG:{config}")

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_scheduler_step'], config['lr_scheduler_decay'])
    metric_ftns = [accuracy]

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
        early_stopping=100000000,  # No early stopping.
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr_scheduler=lr_scheduler
    )
    trainer.train_epoch()
