import torch
import torch.nn as nn
import numpy as np
from src.data_processing.transform_data import Augment


class MixMatch(object):

    def __init__(self, model, batch_size, device, T=0.5, K=2, alpha=0.75):
        self.T = T
        self.K = K
        self.batch_size = batch_size
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=1)
        self.model = model
        self.device = device
        self.n_labels = 10  # Warning! hardcoded
        self.beta = torch.distributions.beta.Beta(alpha, alpha)

    def run(self, x_imgs, x_labels, u_imgs):
        # One hot encoding
        x_labels = self.one_hot_encoding(x_labels)
        x_labels.to(self.device)

        # Augment
        augment_once = Augment(K=1)
        augment_k = Augment(K=self.K)

        x_hat = augment_once(x_imgs)  # shape (1, batch_size, 3, 32, 32)
        u_hat = augment_k(u_imgs)  # shape (K, batch_size, 3, 32, 32)

        # Generate guessed labels
        q_bar = self.guess_label(u_hat)
        q = self.sharpen(q_bar)  # shape (K, batch_size, 10)

        x_hat = x_hat.reshape((-1, 3, 32, 32))  # shape (batch_size, 3, 32, 32)
        u_hat = u_hat.reshape((-1, 3, 32, 32))  # shape (K*batch_size, 3, 32, 32)
        q = q.repeat(self.K, 1, 1).reshape(-1, 10)  # shape (K*batch_size, 10)

        # Concat and shuffle
        w_imgs = torch.cat((x_hat, u_hat))
        w_labels = torch.cat((x_labels, q))
        w_imgs, w_labels = self.shuffle_matrices(w_imgs, w_labels)

        # Apply MixUp
        x_prime, p_prime = self.mixup(x_hat, w_imgs[:self.batch_size], x_labels, w_labels[:self.batch_size])
        u_prime, q_prime = self.mixup(u_hat, w_imgs[self.batch_size:], q, w_labels[self.batch_size:])

        return (x_prime, p_prime), (u_prime, q_prime)

    def mixup(self, x1, x2, p1, p2):
        n_samples = x1.shape[0]
        lambda_rand = self.beta.sample([n_samples, 1, 1, 1]).to(self.device)  # one lambda per sample
        lambda_prime = torch.max(lambda_rand, 1 - lambda_rand).to(self.device)
        x_prime = lambda_prime * x1 + (1 - lambda_prime) * x2
        lambda_prime = lambda_prime.reshape(-1, 1)
        p_prime = lambda_prime * p1 + (1 - lambda_prime) * p2
        return x_prime, p_prime

    def sharpen(self, q_bar):
        #q_bar = q_bar.numpy()
        q = torch.pow(q_bar, 1 / self.T) / torch.sum(torch.pow(q_bar, 1 / self.T), dim=1)[:, np.newaxis]
        return q

    def guess_label(self, u_hat):
        # Do not change model to eval mode! label guessing must be done in train mode
        with torch.no_grad():
            q_bar = torch.zeros([self.batch_size, self.n_labels], device=self.device)
            for k in range(self.K):
                q_bar += self.softmax(self.model(u_hat[k]))
            q_bar /= self.K
        return q_bar

    def one_hot_encoding(self, labels):
        shape = (labels.shape[0], self.n_labels)
        one_hot = torch.zeros(shape, dtype=torch.float32, device=self.device)
        rows = torch.arange(labels.shape[0])
        one_hot[rows, labels] = 1
        return one_hot

    # shuffles along the first axis (axis 0)
    def shuffle_matrices(self, m1, m2):
        n_samples = m1.shape[0]
        rand_indexes = torch.randperm(n_samples)
        m1 = m1[rand_indexes]
        m2 = m2[rand_indexes]
        return m1, m2
