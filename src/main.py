from pathlib import Path
import yaml
import time
import datetime
import torch

from viz.viz_training import plot_acc, plot_training_loss, plot_losses
from mixmatch_trainer import MixMatchTrainer
from fs_trainer import FullySupervisedTrainer

if __name__ == '__main__':
    print("Starting main...")
    configuration = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

    load_checkpoint = configuration['load_checkpoint']
    save_path = configuration['save_path']
    training_type = configuration['training_type']
    config = configuration['MixMatchTrainer']
    params = configuration['WideResNet']
    adam = config['adam']
    sgd = config['sgd']
    lambda_u = config['lambda_u']

    dataset = config['dataset']

    batch_size = configuration[dataset]['batch_size']
    num_labeled = configuration[dataset]['num_labeled']
    n_steps = config['n_steps']
    K = config['K']
    lambda_u_params = lambda_u['lambda_u_max'], lambda_u['step_top_up']
    steps_validation = config['steps_validation']
    steps_checkpoint = config['steps_checkpoint']
    use_pseudo = config['use_pseudo']
    tau = config['tau']
    optimizer = config['optimizer']
    adam_params = adam['lr'], adam['weight_decay']
    sgd_params = sgd['lr'], sgd['momentum'], sgd['weight_decay'], sgd['lr_decay_steps']

    wideresnet_params = (params['depth'], params['k'], params['n_out'])

    if training_type == 'mixmatch':
        trainer = MixMatchTrainer(batch_size, num_labeled, wideresnet_params, n_steps, K, lambda_u_params, optimizer,
                              adam_params, sgd_params, steps_validation, steps_checkpoint, dataset, save_path, use_pseudo, tau)
    else:
        trainer = FullySupervisedTrainer(batch_size, wideresnet_params, n_steps, optimizer, adam_params, sgd_params, steps_validation, steps_checkpoint, dataset, save_path)

    start_time = time.time()

    if load_checkpoint != '':
        trainer.load_checkpoint(load_checkpoint)

    trainer.train()

    seconds = time.time() - start_time
    print("Time elapsed: " + str(datetime.timedelta(seconds=seconds)))

    trainer.save_model(step=n_steps, path=f'{save_path}/last_one.pt')

    # plot_training_loss(trainer.train_losses, trainer.val_losses)
    # plot_acc(trainer.train_accuracies, trainer.val_accuracies)
    # plot_losses(*trainer.get_losses())
