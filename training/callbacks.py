import sys
import os.path as path
from collections import defaultdict

import numpy as np
import torch

from .abstract_callback import AbstractCallback


class StepwiseLearningRateReduction(AbstractCallback):
    """
    Reduces the learning rate of the optimizer every N epochs.

    Args:
        epoch_steps (int): number of epochs after which learning rate is reduced
        reduction_factor (float): multiplicative factor for learning rate reduction
        min_lr (float): lower bound for learning rate
    """

    def __init__(self, epoch_steps, reduction_factor, min_lr=None):
        self._epoch_steps = epoch_steps
        self._reduction_factor = reduction_factor
        self._min_lr = min_lr

    def __call__(self, epoch, step, performance_measures, context):

        # execute at the beginning of every Nth epoch
        if epoch > 0 and step==0 and epoch%self._epoch_steps==0:

            # reduce lr for each param group (necessary for e.g. Adam)
            for param_group in context.optimizer.param_groups:
                new_lr = param_group['lr'] * self._reduction_factor

                if self._min_lr is not None and new_lr < self._min_lr:
                    continue

                param_group['lr'] = new_lr
                print("Epoch {}: Reducing learning rate to {}".format(epoch, new_lr))

class ScheduledLearningRateReduction(AbstractCallback):
    """
    Reduces the learning rate of the optimizer for every scheduled epoch.

    Args:
        epoch_schedule (list of int): defines at which epoch the learning rate will be reduced
        reduction_factor (float): multiplicative factor for learning rate reduction
        min_lr (float): lower bound for learning rate
    """

    def __init__(self, epoch_schedule, reduction_factor, min_lr=None):
        self._epoch_schedule = sorted(epoch_schedule)
        self._reduction_factor = reduction_factor
        self._min_lr = min_lr

    def __call__(self, epoch, step, performance_measures, context):

        if not self._epoch_schedule: # stop if schedule is empty
            return

        next_epoch_step = self._epoch_schedule[0]
        if epoch >= next_epoch_step and step==0:

            # reduce lr for each param group (necessary for e.g. Adam)
            for param_group in context.optimizer.param_groups:
                new_lr = param_group['lr'] * self._reduction_factor

                if self._min_lr is not None and new_lr < self._min_lr:
                    continue

                param_group['lr'] = new_lr
                print("Epoch {}: Reducing learning rate to {}".format(epoch, new_lr))

            self._epoch_schedule.pop(0)

class HistoryRecorder(AbstractCallback):
    """ Records all losses and metrics during training. """

    def __init__(self):
        self.history = defaultdict(list)

    def __call__(self, epoch, step, performance_measures, context):

        if step != len(context.train_data_loader): # only record at end of epoch
            return

        for key, value in performance_measures.items():
            if type(value) == torch.Tensor:
                value = value.item()
            self.history[key].append(value)

class ModelCheckpoint(AbstractCallback):
    """
    Saves the model and optimizer state at the point with lowest validation error throughout training.

    Args:
        output_path (string): path to directory where the checkpoint will be saved to
        model_name (string): name of the checkpoint file
        target_metric (string): metric based on which checkpoints are generated
        smallest (bool): indicates if smaller or bigger values are considered better
    """

    def __init__(self, output_path, model_name='model_checkpoint.pt', target_metric='val_loss', smallest=True):
        self.output_path = path.join(output_path, model_name)
        self.target_metric = target_metric
        self.best_score = sys.float_info.max if smallest else  sys.float_info.min
        self.smallest = smallest

    def __call__(self, epoch, step, performance_measures, context):
        if not self.target_metric in performance_measures:
            return

        if self._compare_measures(performance_measures[self.target_metric], self.best_score):
            self.best_score = performance_measures[self.target_metric]
            self._save_checkpoint(context.model, context.optimizer, epoch)

    def _compare_measures(self, new_val, best_val):
        if self.smallest:
            return new_val < best_val
        return new_val > best_val

    def _save_checkpoint(self, model, optimizer, epoch):
        print("Saving model at checkpoint.")
        model.eval()
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        torch.save({'arch' : model.__class__.__name__,
                    'epoch' : epoch,
                    'model_state_dict' : model_state_dict,
                    'optimizer_state_dict' : optimizer_state_dict
                    }, self.output_path)
        model.train()

class LayerwiseGradientNorm(AbstractCallback):
    """ Collects the layer-wise gradient norms for each epoch. """

    def __init__(self):
        self.layer_grads = dict()
        self._batch_layer_grads = dict()

    def __call__(self, epoch, step, performance_measures, context):
        """
        Store gradient norms for each batch and compute means after the
        epoch's last batch.
        """
        self._store_batch_layer_grads(context.model)

        if step == len(context.train_data_loader): # end of epoch
            self._store_layer_grads()
            self._batch_layer_grads = dict()

    def _store_batch_layer_grads(self, model):
        """ Store gradient norm of each layer for current batch. """
        for name, param in model.named_parameters():

            if not param.requires_grad or param.grad is None:
                continue

            if not name in self._batch_layer_grads:
                self._batch_layer_grads[name] = []

            grad_norm = torch.sqrt(torch.sum(param.grad**2)).item()
            self._batch_layer_grads[name].append(grad_norm)

    def _store_layer_grads(self):
        """ Compute mean of all batch steps in epoch. """
        for name, grads in self._batch_layer_grads.items():

            if not name in self.layer_grads:
                self.layer_grads[name] = []

            layer_epoch_grad = np.mean(grads)
            self.layer_grads[name].append(layer_epoch_grad)

class EarlyStopping(AbstractCallback):
    """
    Early Stopping to terminate training early if the monitored metric did not improve
    over a number of epochs.

    Args:
        monitor (string): name of the relevant loss or metric (usually 'val_loss')
        min_delta (float): minimum change in monitored metric to qualify as an improvement
        patience (int): number of epochs to wait for an improvement before terminating the training
    """

    def __init__(self, monitor='val_loss', min_delta=0, patience = 5):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.last_best = sys.float_info.max
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, epoch, step, performance_measures, context):

        if step != len(context.train_data_loader): # only continue at end of epoch
            return

        if not self.monitor in performance_measures:
            return

        current_loss = performance_measures[self.monitor]
        if (self.last_best - current_loss) >= self.min_delta:
            self.last_best = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            context._stop_training = True # make ModelTrainer stop
            print('\nEarly stopping after epoch {}' % (epoch))

class ReduceLROnPlateau(AbstractCallback):
    """
    Reduce the learning rate if the train or validation loss plateaus.

    Args:
        monitor (string): name of the relevant loss or metric (usually 'val_loss')
        factor (float): factor by which the lr is decreased at each step
        patience (int): number of epochs to wait on plateau for loss improvement before reducing lr
        min_delta (float): minimum improvement necessary to reset patience
        cooldown (int): number of epochs to cooldown after a lr reduction
        min_lr (float): minimum value the learning rate can decrease to
        verbose (bool): print to console
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10, min_delta=0, cooldown=0, min_lr=0, verbose=False):
        self.monitor = monitor
        if factor >= 1.0 or factor < 0:
            raise ValueError('ReduceLROnPlateau does only support a factor in [0,1[.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best_loss = sys.float_info.max

    def __call__(self, epoch, step, performance_measures, context):

        if not self.monitor in performance_measures:
            return

        if step != len(context.train_data_loader): # only continue at end of epoch
            return

        if self.cooldown_counter > 0: # in cooldown phase
            self.cooldown_counter -= 1
            self.wait = 0

        current_loss = performance_measures[self.monitor]
        if (self.best_loss - current_loss) >= self.min_delta: # loss improved, save and reset wait counter
            self.best_loss = current_loss
            self.wait = 0
        elif self.cooldown_counter <= 0: # no improvement and not in cooldown

            if self.wait >= self.patience: # waited long enough, reduce lr
                for param_group in context.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr * self.factor
                    if new_lr >= self.min_lr: # only decrease if there is still enough buffer space
                        if self.verbose:
                            print("Epoch {}: Reducing learning rate from {} to {}".format(epoch, old_lr, new_lr)) #TODO print per param group?
                        param_group['lr'] = new_lr
                self.cooldown_counter = self.cooldown # new cooldown phase after lr reduction
                self.wait = 0
            else:
                self.wait += 1
