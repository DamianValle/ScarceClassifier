import matplotlib.pyplot as plt
import numpy as np

from d02_data.load_data import get_dataloaders, get_dataloaders_validation
from d03_processing.transform_data import TransformTwice
from d07_visualization.visualize_cifar10 import show_img, show_grid

# Mover un directorio parriba pa que chute

if __name__ == '__main__':
    augment_unlabeled = TransformTwice(K=4)
    train_loader, val_loader, test_loader = get_dataloaders_validation(path='../data', batch_size=1)

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0], data[1]

        i1, i2, i3, i4 = augment_unlabeled(inputs)

        show_grid([inputs[0], i1[0], i2[0], i3[0], i4[0]])

        break