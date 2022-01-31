import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_processing.load_data_idxs import get_dataloaders_with_index
from models.wideresnet import WideResNet
from mixmatch import MixMatch
import numpy as np
import pdb


class MixMatchTrainer:

    def __init__(self, batch_size, num_lbls, model_params, n_steps, K, lambda_u, optimizer, adam,
                 sgd, steps_validation, steps_checkpoint, dataset, save_path, use_pseudo, tau):

        self.validation_set = False

        self.n_steps = n_steps
        self.start_step = 0
        self.K = K
        self.steps_validation = steps_validation
        self.steps_checkpoint = steps_checkpoint
        self.num_labeled = num_lbls
        self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader, self.lbl_idx, self.unlbl_idx, self.val_idx \
            = get_dataloaders_with_index(path='../data',
                                         batch_size=batch_size,
                                         num_labeled=num_lbls,
                                         which_dataset=dataset,
                                         validation=self.validation_set)
        print('Labeled samples: ' + str(len(self.labeled_loader.sampler)) + '\tUnlabeled samples: ' + str(len(self.unlabeled_loader.sampler)))
        self.targets_list = np.array(self.labeled_loader.dataset.targets)
        self.batch_size = self.labeled_loader.batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # -- Model --
        depth, k, n_out = model_params
        self.model = WideResNet(depth=depth, k=k, n_out=n_out, bias=False).to(self.device)
        self.ema_model = WideResNet(depth=depth, k=k, n_out=n_out, bias=False).to(self.device)
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

        self.lambda_u_max, self.step_top_up = lambda_u
        self.loss_mixmatch = Loss(self.lambda_u_max, self.step_top_up)
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracies, self.train_losses, = [], []
        self.val_accuracies, self.val_losses, = [], []
        self.best_acc = 0

        self.mixmatch = MixMatch(self.model, self.batch_size, self.device)

        self.writer = SummaryWriter()
        self.path = save_path

        # -- Pseudo label --
        self.use_pseudo = use_pseudo
        self.steps_pseudo_lbl = 5000
        self.tau = tau  # confidence threshold

        self.min_unlbl_samples = 1000
        # Make a deep copy of original unlabeled loader
        _, self.unlabeled_loader_original, _, _, _, _, _ \
            = get_dataloaders_with_index(path='../data',
                                         batch_size=batch_size,
                                         num_labeled=num_lbls,
                                         which_dataset=dataset,
                                         lbl_idxs=self.lbl_idx,
                                         unlbl_idxs=self.unlbl_idx,
                                         valid_idxs=self.val_idx,
                                         validation=self.validation_set)

    def train(self):

        iter_labeled_loader = iter(self.labeled_loader)
        iter_unlabeled_loader = iter(self.unlabeled_loader)

        for step in range(self.start_step, self.n_steps):

            # Get next batch of data
            self.model.train()
            try:
                x_imgs, x_labels, _ = iter_labeled_loader.next()
                # Check if batch size has been cropped for last batch
                if x_imgs.shape[0] < self.batch_size:
                    iter_labeled_loader = iter(self.labeled_loader)
                    x_imgs, x_labels, _ = iter_labeled_loader.next()
            except:
                iter_labeled_loader = iter(self.labeled_loader)
                x_imgs, x_labels, _ = iter_labeled_loader.next()

            try:
                u_imgs, _, _ = iter_unlabeled_loader.next()
                if u_imgs.shape[0] < self.batch_size:
                    iter_unlabeled_loader = iter(self.unlabeled_loader)
                    u_imgs, _, _ = iter_unlabeled_loader.next()
            except:
                iter_unlabeled_loader = iter(self.unlabeled_loader)
                u_imgs, _, _ = iter_unlabeled_loader.next()

            # Send to GPU
            x_imgs = x_imgs.to(self.device)
            x_labels = x_labels.to(self.device)
            u_imgs = u_imgs.to(self.device)

            # MixMatch algorithm
            x, u = self.mixmatch.run(x_imgs, x_labels, u_imgs)
            x_input, x_targets = x
            u_input, u_targets = u
            u_targets.detach_()  # stop gradients from propagation to label guessing

            # Compute X' predictions
            x_output = self.model(x_input)

            # Compute U' predictions. Separate in batches
            u_batch_outs = []
            for k in range(self.K):
                u_batch = u_input[k * self.batch_size:(k + 1) * self.batch_size]
                u_batch_outs.append(self.model(u_batch))
            u_outputs = torch.cat(u_batch_outs, dim=0)

            # Compute loss
            loss = self.loss_mixmatch(x_output, x_targets, u_outputs, u_targets, step)

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
                if is_best and step > 10000:
                    self.save_model(step=step, path=f'{self.path}/best_checkpoint.pt')

            # Save checkpoint
            if step > 0 and not step % self.steps_checkpoint:
                self.save_model(step=step, path=f'{self.path}/checkpoint_{step}.pt')

            # Generate Pseudo-labels
            if self.use_pseudo and step >= 50000 and not step % self.steps_pseudo_lbl:
                # matrix columns: [index, confidence, pseudo_label, true_label, is_ground_truth]
                matrix = self.get_pseudo_labels()
                self.print_threshold_comparison(matrix)

                # Generate pseudo set based on threshold (same for all classes)
                if self.tau != -1:
                    matrix = self.generate_pseudo_set(matrix)

                # Generate pseudo set balanced (top 90% guesses of each class)
                else:
                    matrix = self.generate_pseudo_set_balanced(matrix)

                iter_labeled_loader = iter(self.labeled_loader)
                iter_unlabeled_loader = iter(self.unlabeled_loader)

                # Save
                torch.save(matrix, f'{self.path}/pseudo_matrix_balanced_{step}.pt')

        # --- Training finished ---
        test_val, test_acc = self.evaluate(self.test_loader)
        print("Training done!!\t Test loss: %.3f \t Test accuracy: %.3f" % (test_val, test_acc))

        self.writer.flush()

    # --- support functions ---
    def evaluate_loss_acc(self, step):
        val_loss, val_acc = self.evaluate(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        train_loss, train_acc = self.evaluate(self.labeled_loader)
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


    def get_losses(self):
        return self.loss_mixmatch.loss_list, self.loss_mixmatch.lx_list, self.loss_mixmatch.lu_list, self.loss_mixmatch.lu_weighted_list

    def save_model(self, step=None, path=f'../model.pt'):
        loss_list, lx, lu, lu_weighted = self.get_losses()
        if not step:
            step = self.n_steps     # Training finished

        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_train': self.train_losses,
            'loss_val': self.val_losses,
            'acc_train': self.train_accuracies,
            'acc_val': self.val_accuracies,
            'loss_batch': loss_list,
            'lx': lx,
            'lu': lu,
            'lu_weighted': lu_weighted,
            'steps': self.n_steps,
            'batch_size': self.batch_size,
            'num_labels': self.num_labeled,
            'lambda_u_max': self.lambda_u_max,
            'step_top_up': self.step_top_up,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'lr_decay': self.lr_decay,
            'lbl_idx': self.lbl_idx,
            'unlbl_idx': self.unlbl_idx,
            'val_idx': self.val_idx,
        }, path)

    def load_checkpoint(self, path_checkpoint):
        saved_model = torch.load(path_checkpoint)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.ema_model.load_state_dict(saved_model['ema_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        self.start_step = saved_model['step']
        self.train_losses = saved_model['loss_train']
        self.val_losses = saved_model['loss_val']
        self.train_accuracies = saved_model['acc_train']
        self.val_accuracies = saved_model['acc_val']
        self.batch_size = saved_model['batch_size']
        self.num_labeled = saved_model['num_labels']
        self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader, self.lbl_idx, self.unlbl_idx, self.val_idx = \
            get_dataloaders_with_index(path='../data',
                                       batch_size=self.batch_size,
                                       num_labeled=self.num_labeled,
                                       which_dataset='cifar10',
                                       lbl_idxs=saved_model['lbl_idx'],
                                       unlbl_idxs=saved_model['unlbl_idx'],
                                       valid_idxs=saved_model['val_idx'],
                                       validation=self.validation_set)
        self.unlabeled_loader_original = self.unlabeled_loader
        print('Model ' + path_checkpoint + ' loaded.')

    def get_pseudo_labels(self):
        matrix = torch.tensor([], device=self.device)

        # Iterate through unlabeled loader
        for batch_idx, (data, target, idx) in enumerate(self.unlabeled_loader_original):
            with torch.no_grad():
                # Get predictions for unlabeled samples
                out = self.model(data.to(self.device))
                p_out = torch.softmax(out, dim=1)  # turn into probability distribution
                confidence, pseudo_lbl = torch.max(p_out, dim=1)
                pseudo_lbl_batch = torch.vstack((idx.to(self.device), confidence, pseudo_lbl)).T
                # Append to matrix
                matrix = torch.cat((matrix, pseudo_lbl_batch), dim=0)  # (n_unlabeled, 3)

        n_unlabeled = matrix.shape[0]
        indices = matrix[:, 0].cpu().numpy().astype(int)
        ground_truth = self.targets_list[indices]
        matrix = torch.vstack((matrix.T, torch.tensor(ground_truth, device=self.device))).T   # (n_unlabeled, 4)
        matrix = torch.vstack((matrix.T, torch.zeros(n_unlabeled, device=self.device))).T   # (n_unlabeled, 5)

        # matrix columns: [index, confidence, pseudo_label, true_label, is_ground_truth]

        # Check if pseudo label is ground truth
        for i in range(n_unlabeled):
            if matrix[i, 2] == matrix[i, 3]:
                matrix[i, 4] = 1
        return matrix

    def generate_pseudo_set(self, matrix):

        unlbl_mask1 = (matrix[:, 1] < self.tau)
        # unlbl_mask2 = (matrix[:, 1] >= 0.99)
        pseudo_mask = (matrix[:, 1] >= self.tau)
        # pseudo_mask = (matrix[:, 1] >= self.tau) & (matrix[:, 1] < 0.99)

        # unlbl_indices = torch.cat((matrix[unlbl_mask1, 0], matrix[unlbl_mask2, 0]))
        unlbl_indices = matrix[unlbl_mask1, 0]
        matrix = matrix[pseudo_mask, :]
        indices = matrix[:, 0]

        new_lbl_idx = np.int_(torch.cat((torch.tensor(self.lbl_idx, device=self.device), indices)).tolist())
        new_unlbl_idx = np.int_(unlbl_indices.tolist())
        self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader, _, _, new_val_idx = \
            get_dataloaders_with_index(path='../data', batch_size=self.batch_size, num_labeled=self.num_labeled,
                                       which_dataset='cifar10', lbl_idxs=new_lbl_idx, unlbl_idxs=new_unlbl_idx,
                                       valid_idxs=self.val_idx, validation=self.validation_set)

        assert np.allclose(self.val_idx, new_val_idx), 'error'
        assert (len(self.labeled_loader.sampler) + len(self.unlabeled_loader.sampler) == 50000), 'error'

        # Change real labels for pseudo labels
        for i in range(matrix.shape[0]):
            index = int(matrix[i, 0].item())
            assert int(matrix[i, 3]) == self.labeled_loader.dataset.targets[index]
            pseudo_labels = int(matrix[i, 2].item())
            self.labeled_loader.dataset.targets[index] = pseudo_labels

        correct = torch.sum(matrix[:, 4]).item()
        pseudo_acc = correct / matrix.shape[0] * 100 if matrix.shape[0] > 0 else 0
        print('Generated labels: %d\t Correct: %d\t Accuracy: %.2f' % (matrix.shape[0], correct, pseudo_acc))
        print('Training with Labeled / Unlabeled / Validation samples\t %d %d %d' % (len(new_lbl_idx),
                                                                                     len(new_unlbl_idx),
                                                                                     len(self.val_idx)))

        return matrix

    def generate_pseudo_set_balanced(self, matrix_all):

        unlbl_indices = torch.tensor([], device=self.device)

        # Get top 10% confident guesses for each class
        matrix = torch.tensor([], device=self.device)
        for i in range(10):
            matrix_label = matrix_all[matrix_all[:, 2] == i, :]
            threshold = torch.quantile(matrix_label[:, 1], 0.9)     # returns prob in the percentile 90
            unlbl_idxs = matrix_label[matrix_label[:, 1] < threshold, 0]
            matrix_label = matrix_label[matrix_label[:, 1] >= threshold, :]
            matrix = torch.cat((matrix, matrix_label), dim=0)
            unlbl_indices = torch.cat((unlbl_indices, unlbl_idxs))
        indices = matrix[:, 0]

        new_lbl_idx = np.int_(torch.cat((torch.tensor(self.lbl_idx, device=self.device), indices)).tolist())
        new_unlbl_idx = np.int_(unlbl_indices.tolist())
        self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader, _, _, new_val_idx = \
            get_dataloaders_with_index(path='../data', batch_size=self.batch_size, num_labeled=self.num_labeled,
                                       which_dataset='cifar10', lbl_idxs=new_lbl_idx, unlbl_idxs=new_unlbl_idx,
                                       valid_idxs=self.val_idx, validation=self.validation_set)

        assert np.allclose(self.val_idx, new_val_idx), 'error'
        assert (len(self.labeled_loader.sampler) + len(self.unlabeled_loader.sampler) == 50000), 'error'

        # Change real labels for pseudo labels
        for i in range(matrix.shape[0]):
            index = int(matrix[i, 0].item())
            assert int(matrix[i, 3]) == self.labeled_loader.dataset.targets[index]
            pseudo_labels = int(matrix[i, 2].item())
            self.labeled_loader.dataset.targets[index] = pseudo_labels

        correct = torch.sum(matrix[:, 4]).item()
        pseudo_acc = correct / matrix.shape[0] * 100 if matrix.shape[0] > 0 else 0
        print('Generated labels: %d\t Correct: %d\t Accuracy: %.2f' % (matrix.shape[0], correct, pseudo_acc))
        print('Training with Labeled / Unlabeled / Validation samples\t %d %d %d' % (len(new_lbl_idx),
                                                                                     len(new_unlbl_idx),
                                                                                     len(self.val_idx)))
        return matrix

    def print_threshold_comparison(self, matrix):
        m2 = matrix[matrix[:, 1] >= 0.9, :]
        for i, tau in enumerate([0.9, 0.95, 0.97, 0.99, 0.999]):
            pseudo_labels = m2[m2[:, 1] >= tau, :]
            total = pseudo_labels.shape[0]
            correct = torch.sum(pseudo_labels[:, 4]).item()
            print('Confidence threshold %.3f\t Generated / Correct / Precision\t %d\t%d\t%.2f '
                  % (tau, total, correct, correct / (total + np.finfo(float).eps) * 100))


class Loss(object):

    def __init__(self, lambda_u_max, step_top_up):
        self.lambda_u_max = lambda_u_max
        self.step_top_up = step_top_up
        self.mse_loss = nn.MSELoss()
        self.lx_list = []
        self.lu_list = []
        self.lu_weighted_list = []
        self.loss_list = []

    def __call__(self, x_output, x_target, u_output, u_target, step):
        lambda_u = self.ramp_up_lambda(step)
        u_output = torch.softmax(u_output, dim=1)

        lx = - torch.mean(torch.sum(x_target * torch.log_softmax(x_output, dim=1), dim=1))
        # lx = torch.clamp(lx, min=0, max=2)  # Try clamping lx to
        lu = self.mse_loss(u_output, u_target)
        loss = lx + lu * lambda_u

        self.lx_list.append(lx.item())
        self.lu_list.append(lu.item())
        self.lu_weighted_list.append(lu.item() * lambda_u)
        self.loss_list.append(loss.item())
        return loss

    def ramp_up_lambda(self, step):
        if step > self.step_top_up:
            return self.lambda_u_max
        else:
            return self.lambda_u_max * step / self.step_top_up



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
                param.mul_(1 - self.wd) # Beware that this "param" affects the main model. It is passed by reference
