import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_dataloaders_ssl(path="../../data", batch_size=64, num_labeled=250,
                        lbl_idxs=None, unlbl_idxs=None, valid_idxs=None, which_dataset='cifar10'):
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
        train_set = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    elif which_dataset == 'svhn':
        train_set = datasets.SVHN(root=path, split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root=path, split='test', download=True, transform=transform)
    else:
        train_set = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)


    # Split indexes between labeled, unlabeled and validation
    if which_dataset == 'cifar10':
        training_labels = train_set.targets
    elif which_dataset == 'svhn':
        training_labels = train_set.labels
    else :
        training_labels = train_set.targets

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = labeled_unlabeled_val_split(training_labels,
                                                                                     int(num_labeled / 10))
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
    train_labeled_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                       sampler=train_labeled_sampler, num_workers=0)
    train_unlabeled_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                         sampler=train_unlabeled_sampler, num_workers=0)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

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


# -------- This function is deprecated ---------
def get_dataloaders_validation(path="../../data", batch_size=64, shuffle=False, augment=False,
                               train_size=45000, val_size=5000):
    """
    Include split in train and valdiation set
    """
    assert ((train_size >= 0) and (val_size >= 0) and (train_size + val_size <= 50000))

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Need to load dataset twice in case train data is augmented but not validation
    train_set = datasets.CIFAR10(root=path, train=True,
                                 download=True, transform=transform_train)

    val_set = datasets.CIFAR10(root=path, train=True,
                               download=True, transform=transform_val)

    test_set = datasets.CIFAR10(root=path, train=False,
                                download=True, transform=transform_test)

    indices = list(range(train_size + val_size))
    if shuffle:
        np.random.seed(1)  # Hardcoded seed for the moment
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[:train_size], indices[train_size:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader


# -------- This function is deprecated ---------
def get_dataloaders(path='../../data', batch_size=64):
    transform_train = transforms.Compose(
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = batch_size

    trainset = datasets.CIFAR10(root=path, train=True,
                                download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=path, train=False,
                               download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader


if __name__ == "__main__":
    _, _, _, _ = get_dataloaders_ssl()
    print('foo')

