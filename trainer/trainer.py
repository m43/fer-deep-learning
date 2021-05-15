# -*- coding: utf-8 -*-
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from model.nn import draw_conv_filters
from utils.util import MetricTracker, Object


class SimpleConvTrainer(BaseTrainer):

    def __init__(self, run_name, model, criterion, metric_ftns, optimizer, device, device_ids,
                 epochs, save_folder, monitor, log_step, early_stopping, train_loader, val_loader=None,
                 test_loader=None, start_epoch=1, lr_scheduler=None, ):

        super().__init__(run_name, model, criterion, metric_ftns, optimizer, device, device_ids, epochs, save_folder,
                         monitor, start_epoch, early_stopping)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader  # TODO

        self.lr_scheduler = lr_scheduler
        self.len_epoch = len(self.train_loader)
        self.log_step = log_step

        self.train_metrics = MetricTracker("train", 'loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker("val", 'loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker("test", 'loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            step = (epoch - 1) * self.len_epoch + batch_idx
            self.train_metrics.update('loss', loss.item())
            self.writer.add_scalar(f"{self.train_metrics.get_name()} loss", loss.item(), step)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
                self.writer.add_scalar(f"{self.train_metrics.get_name()} {met.__name__}", met(output, target), step)

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))
                self.writer.add_image(f"{self.train_metrics.get_name()} conv1 filters (v1)",
                                      make_grid(self.model.conv1.weight, nrow=8, normalize=True, padding=0), step)
                self.writer.add_image(f"{self.train_metrics.get_name()} conv1 filters (v2)",
                                      make_grid(self.model.conv1.weight, nrow=8, normalize=True, padding=1), step)
                self.writer.add_image(f"{self.train_metrics.get_name()} conv1 filters (v3)",
                                      self.draw_conv_filters(epoch, step, self.model.conv1.weight.clone().detach().numpy(),
                                                             None, False), step, dataformats="HWC")

                conv1_wrapper = Object()
                conv1_wrapper.name = "conv1"
                conv1_wrapper.C = self.model.conv1.weight.shape[1]
                conv1_wrapper.weights = self.model.conv1.weight.clone().detach().numpy()
                conv1_wrapper.weights = conv1_wrapper.weights.reshape(conv1_wrapper.weights.shape[0], -1)
                self.writer.add_image(f"{self.train_metrics.get_name()} conv1 filters (v0)",
                                      np.expand_dims(draw_conv_filters(epoch, step, conv1_wrapper, None, False), 0),
                                      step)

                # self.writer.add_image(f"{self.train_metrics.get_name()} input", make_grid(
                #     [make_grid(data.cpu()[0], nrow=8, normalize=True),
                #      make_grid(data.cpu()[1], nrow=8, normalize=True)], nrow=2, normalize=True), step)

        log = self.train_metrics.result()
        self.writer.add_scalar(f"{self.train_metrics.get_name()} epoch loss", log["loss"], epoch)
        for met in self.metric_ftns:
            self.writer.add_scalar(f"{self.train_metrics.get_name()} epoch {met.__name__}", log[met.__name__],
                                   epoch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, epoch, bins='auto')

        if self.val_loader is not None:
            val_log = self._evaluate(epoch, self.valid_metrics, self.val_loader)
            log.update(**{f"{self.valid_metrics.get_name()} {k}": v for k, v in val_log.items()})

        if self.test_loader is not None:
            test_log = self._evaluate(epoch, self.test_metrics, self.test_loader)
            log.update(**{f"{self.test_metrics.get_name()} {k}": v for k, v in test_log.items()})

        if self.lr_scheduler is not None:
            self.writer.add_scalar(f"epoch lr", self.lr_scheduler.get_lr()[0], epoch)
            self.lr_scheduler.step()

        return log

    def _evaluate(self, epoch, metrics, dataloader):
        metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                self.model.eval()
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                metrics.update('loss', loss.item())

                output = self.model(data)
                for met in self.metric_ftns:
                    m = met(output, target)
                    metrics.update(met.__name__, m)

        result = metrics.result()
        self.writer.add_scalar(f"{metrics.get_name()} epoch loss", result["loss"], epoch)
        for met in self.metric_ftns:
            self.writer.add_scalar(f"{metrics.get_name()} epoch {met.__name__}", result[met.__name__], epoch)
        return result

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def draw_conv_filters(self, epoch, step, w, save_dir, save_images=False):
        num_filters = w.shape[0]
        num_channels = w.shape[1]
        k = w.shape[2]
        assert w.shape[3] == w.shape[2]
        w = w.transpose(2, 3, 1, 0)
        w -= w.min()
        w /= w.max()
        border = 1
        cols = 8
        rows = math.ceil(num_filters / cols)
        width = cols * k + (cols - 1) * border
        height = rows * k + (rows - 1) * border
        img = np.zeros([height, width, num_channels])
        for i in range(num_filters):
            r = int(i / cols) * (k + border)
            c = int(i % cols) * (k + border)
            img[r:r + k, c:c + k, :] = w[:, :, :, i]
        filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
        if save_images:
            ski.io.imsave(os.path.join(save_dir, filename), img)
        return img
