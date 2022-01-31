import numpy as np
import matplotlib.pyplot as plt


def plot_training_loss(loss_train, loss_test, step=1):
    x = np.linspace(1, len(loss_train)*step, len(loss_train))
    plt.plot(x, loss_train, label='Training')
    plt.plot(x, loss_test, label='Validation')
    plt.title('Loss function')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../models/loss.png', format='png')
    #plt.show()


def plot_acc(acc_train, acc_test, step=1):
    x = np.linspace(1, len(acc_train)*step, len(acc_train))
    plt.plot(x, acc_train, label='Training')
    plt.plot(x, acc_test, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('%')
    plt.legend()
    plt.savefig('../models/accuracy.png', format='png')
    #plt.show()


def plot_accuracies(acc_train, acc_test, step=1):
    x = np.linspace(1, len(acc_train)*step, len(acc_train))
    plt.plot(x, acc_train, label='Training')
    plt.plot(x, acc_test, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('%')
    plt.legend()
    plt.savefig('../models/accuracy.png', format='png')
    #plt.show()



def plot_losses(loss, lx, lu, lu_weighted):
    x = np.linspace(1, len(loss), len(loss))
    plt.plot(x, loss, label='Total loss')
    plt.plot(x, lx, label='Lx')
    plt.plot(x, lu, label='Lu')
    plt.plot(x, lu_weighted, label='Lu * lambda_u')
    plt.title('Batch Loss')
    plt.xlabel('Steps')
    plt.legend()
    plt.savefig('../models/losses_mixmatch.png', format='png')
    #plt.show()