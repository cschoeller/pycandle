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
                print("Reducing learning rate to (epoch {}): {}".format(epoch, new_lr))

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
                print("Reducing learning rate to (epoch {}): {}".format(epoch, new_lr))

            self._epoch_schedule.pop(0)

class HistoryRecorder(AbstractCallback):
    """ Records all losses and metrics during training. """

    def __init__(self):
        self.history = defaultdict(list)

    def __call__(self, epoch, step, performance_measures, context):

        if step != len(context.train_data_loader)-1: # only record at end of epoch
            return

        for key, value in performance_measures.items():
            if type(value) == torch.Tensor:
                value = value.item()
            self.history[key].append(value)

class ModelCheckpoint(AbstractCallback):
    """
    Save the model with smallest validation error during training. Requires
    that valdiation data has been provided to callling ModelTrainer.

    Args:
        output_path (string): 
        model_name (string): 
    """

    def __init__(self, output_path, model_name='model_checkpoint.pt'):
        self._output_path = path.join(output_path, model_name)
        self._best_val_score = sys.float_info.max

    def __call__(self, epoch, step, performance_measures, context):
        if not 'val_loss' in performance_measures:
            return

        if performance_measures['val_loss'] < self._best_val_score:
            self._best_val_score = performance_measures['val_loss']
            self._save_model(context.model)

    def _save_model(self, model):
        print("Saving model at checkpoint.")
        model.eval() # save in eval mode to avoid mistakes during deployment
        state_dict = model.state_dict()
        torch.save({'arch' : model.__class__.__name__, 'state_dict' : state_dict}, self._output_path)
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

        if step == (len(context.train_data_loader)-1):
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