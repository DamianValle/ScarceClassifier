import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def get_dataloaders_with_index(path="../../data", batch_size=64, num_labeled=250,
                        lbl_idxs=None, unlbl_idxs=None, valid_idxs=None, which_dataset='cifar10', validation=True):
    """
    Returns data loaders for Semi-Supervised Learning
    Split between train_labeled, train_unlabeled, validation and test
    """

    # Define transform to normalize data
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if which_dataset == 'cifar10':
        train_set = CustomCIFAR10(root=path, train=True, transform=transform)
        test_set = CustomCIFAR10(root=path, train=False, transform=transform)
    elif which_dataset == 'svhn':
        train_set = datasets.SVHN(root=path, split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root=path, split='test', download=True, transform=transform)
    else:
        raise Exception('Not supported yet')


    # Split indexes between labeled, unlabeled and validation
    if which_dataset == 'cifar10':
        training_labels = train_set.targets
    elif which_dataset == 'svhn':
        training_labels = train_set.labels
    else :
        training_labels = train_set.targets

    if validation:
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = labeled_unlabeled_val_split(training_labels, int(num_labeled / 10))
    else:
        train_labeled_idxs, train_unlabeled_idxs = labeled_unlabeled_split(training_labels, int(num_labeled / 10))
        val_idxs = []

    # If indexes are provided, use them
    if lbl_idxs is not None:
        train_labeled_idxs = lbl_idxs
        train_unlabeled_idxs = unlbl_idxs
        val_idxs = valid_idxs

    # Define samplers using indexes
    train_labeled_sampler = SubsetRandomSampler(train_labeled_idxs)
    train_unlabeled_sampler = SubsetRandomSampler(train_unlabeled_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    # Create data loaders
    train_labeled_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_labeled_sampler, num_workers=0)
    train_unlabeled_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_unlabeled_sampler, num_workers=0)
    val_loader = DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    if not validation:
        val_loader = test_loader

    return train_labeled_loader, train_unlabeled_loader, val_loader, test_loader, train_labeled_idxs, train_unlabeled_idxs, val_idxs


def labeled_unlabeled_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])    # 5000 validation samples
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def labeled_unlabeled_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


class CustomCIFAR10(Dataset):
    """
    Returns triplet (data, target, index) in __getitem__()
    """
    def __init__(self, root, train, transform):
        self.cifar10 = datasets.CIFAR10(root=root,
                                        download=True,
                                        train=train,
                                        transform=transform)
        self.targets = self.cifar10.targets

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


if __name__ == "__main__":

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    samples = np.array([1,2,3,4,5,6])
    sampler = SubsetRandomSampler(samples)
    dataset = CustomCIFAR10("../../data", transform)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)

    for batch_idx, (data, target, idx) in enumerate(loader):
        print('Batch idx {}, dataset index {}'.format(
            batch_idx, idx))


