import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_processing.load_data_idxs import get_dataloaders_with_index
from models.wideresnet import WideResNet
from src.data_processing.transform_data import Augment


class FullySupervisedTrainer:

    def __init__(self, batch_size, model_params, n_steps, optimizer, adam,
                 sgd, steps_validation, steps_checkpoint, dataset, save_path):

        self.n_steps = n_steps
        self.start_step = 0
        self.steps_validation = steps_validation
        self.steps_checkpoint = steps_checkpoint
        self.num_labeled = 50000
        self.train_loader, _, self.val_loader, self.test_loader, self.lbl_idx, _, self.val_idx = \
            get_dataloaders_with_index(path='../data',
                                       batch_size=batch_size,
                                       num_labeled=self.num_labeled,
                                       which_dataset=dataset,
                                       validation=False)
        print('Labeled samples: ' + str(len(self.train_loader.sampler)))

        self.batch_size = self.train_loader.batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        depth, k, n_out = model_params
        self.model = WideResNet(depth=depth, k=k, n_out=n_out, bias=True).to(self.device)
        self.ema_model = WideResNet(depth=depth, k=k, n_out=n_out, bias=True).to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()

        if optimizer == 'adam':
            self.lr, self.weight_decay = adam
            self.momentum, self.lr_decay = None, None
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.ema_optimizer = WeightEMA(self.model, self.ema_model, self.lr, alpha=0.999)

        else:
            self.lr, self.momentum, self.weight_decay, self.lr_decay = sgd
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
            self.ema_optimizer = None

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracies, self.train_losses, = [], []
        self.val_accuracies, self.val_losses, = [], []
        self.best_acc = 0

        self.augment = Augment(K=1)

        self.path = save_path
        self.writer = SummaryWriter()

    def train(self):
        iter_train_loader = iter(self.train_loader)

        for step in range(self.n_steps):
            self.model.train()
            # Get next batch of data
            try:
                x_input, x_labels, _ = iter_train_loader.next()
                # Check if batch size has been cropped for last batch
                if x_input.shape[0] < self.batch_size:
                    iter_train_loader = iter(self.train_loader)
                    x_input, x_labels, _ = iter_train_loader.next()
            except:
                iter_train_loader = iter(self.train_loader)
                x_input, x_labels, _ = iter_train_loader.next()

            # Send to GPU
            x_input = x_input.to(self.device)
            x_labels = x_labels.to(self.device)

            # Augment
            x_input = self.augment(x_input)
            x_input = x_input.reshape((-1, 3, 32, 32))

            # Compute X' predictions
            x_output = self.model(x_input)

            # Compute loss
            loss = self.criterion(x_output, x_labels)

            # Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.ema_optimizer:
                self.ema_optimizer.step()

            # Decaying learning rate. Used in with SGD Nesterov optimizer
            if not self.ema_optimizer and step in self.lr_decay:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.2

            # Evaluate model
            self.model.eval()
            if step > 0 and not step % self.steps_validation:
                val_acc, is_best = self.evaluate_loss_acc(step)
                if is_best:
                    self.save_model(step=step, path=f'{self.path}/best_checkpoint.pt')

            # Save checkpoint
            if step > 10000 and not step % self.steps_checkpoint:
                self.save_model(step=step, path=f'{self.path}/checkpoint_{step}.pt')

        # --- Training finished ---
        test_val, test_acc = self.evaluate(self.test_loader)
        print("Training done!!\t Test loss: %.3f \t Test accuracy: %.3f" % (test_val, test_acc))

        self.writer.flush()

    # --- support functions ---

    def evaluate_loss_acc(self, step):
        val_loss, val_acc = self.evaluate(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        train_loss, train_acc = self.evaluate(self.train_loader)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

        is_best = False
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            is_best = True

        print("Step %d.\tLoss train_lbl/valid  %.2f  %.2f\t Accuracy train_lbl/valid  %.2f  %.2f \tBest acc %.2f \t%s" %
              (step, train_loss, val_loss, train_acc, val_acc, self.best_acc, time.ctime()))

        self.writer.add_scalar("Loss train_label", train_loss, step)
        self.writer.add_scalar("Loss validation", val_loss, step)
        self.writer.add_scalar("Accuracy train_label", train_acc, step)
        self.writer.add_scalar("Accuracy validation", val_acc, step)
        return val_acc, is_best

    def evaluate(self, dataloader):
        correct, total, loss = 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                if self.ema_optimizer:
                    outputs = self.ema_model(inputs)
                else:
                    outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            loss /= dataloader.__len__()

        acc = correct / total * 100
        return loss, acc

    def save_model(self, step=None, path='../models/model.pt'):
        if not step:
            step = self.n_steps  # Training finished

        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_train': self.train_losses,
            'loss_val': self.val_losses,
            'acc_train': self.train_accuracies,
            'acc_val': self.val_accuracies,
            'steps': self.n_steps,
            'batch_size': self.batch_size,
            'num_labels': self.num_labeled,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'lr_decay': self.lr_decay,
            'lbl_idx': self.lbl_idx,
            'val_idx': self.val_idx,
        }, path)

    def load_checkpoint(self, model_name):
        saved_model = torch.load(f'../models/{model_name}')
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.ema_model.load_state_dict(saved_model['ema_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        self.start_step = saved_model['step']

        self.train_loader, _, self.val_loader, self.test_loader, self.lbl_idx, _, self.val_idx = \
            get_dataloaders_with_index(path='../data',
                                       batch_size=self.batch_size,
                                       num_labeled=self.num_labeled,
                                       which_dataset='cifar10',
                                       lbl_idxs=saved_model['lbl_idx'],
                                       unlbl_idxs=[],
                                       valid_idxs=saved_model['val_idx'])
        print('Model ' + model_name + ' loaded.')


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.ema_model.eval()
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                # Update Exponential Moving Average parameters
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # Apply Weight Decay
                param.mul_(1 - self.wd)  # Beware that this "param" affects the main model. It is passed by reference
