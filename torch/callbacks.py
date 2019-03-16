import sys
import os.path as path
from collections import defaultdict

import numpy as np
import torch

from .abstract_callback import AbstractCallback


# class EarlyStopping(AbstractCallback):

#     def __init__(self, min_steps, metric='val_loss'):
#         self.min_steps = min_steps
#         self.metric = metric
#         self.last_best = sys.float_info.max
#         self.counter = 0

#     def __call__(self, epoch, step, performance_measures, context):
#         if not self.metric in performance_measures:
#             raise KeyError('Metric not in performance_measures.')

#         measure = performance_measures[self.metric]
#         if measure < self.last_best:
#             self.last_best = measure
#             self.counter = 0
#         else:
#             self.counter += 1

#         if self.counter >= self.min_steps:
#             #TODO stop training


class StepwiseLearningRateReduction(AbstractCallback):
    """ Callback that reduces the learning rate of the optimizer every n epochs. """

    def __init__(self, num_epochs, reduction_factor, min_lr=None):
        self.num_epochs = num_epochs
        self.reduction_factor = reduction_factor
        self.min_lr = min_lr

    def __call__(self, epoch, step, performance_measures, context):
        if epoch > 0 and epoch%(self.num_epochs - 1)==0 and step==0:
            for param_group in context.optimizer.param_groups:
                new_lr = param_group['lr'] * self.reduction_factor

                if self.min_lr is not None and new_lr < self.min_lr:
                    continue

                param_group['lr'] = new_lr
                print("Reducing learning rate to (epoch {}): {}".format(epoch, new_lr))

class ScheduledLearningRateReduction(AbstractCallback):
    """ Callback that reduces the learning rate of the optimizer every n epochs. """

    def __init__(self, epoch_schedule, reduction_factor, min_lr=None):
        self.epoch_schedule = sorted(epoch_schedule)
        self.reduction_factor = reduction_factor
        self.min_lr = min_lr

    def __call__(self, epoch, step, performance_measures, context):
        if not self.epoch_schedule: # Stop if schedule is empty.
            return
        next_epoch_step = self.epoch_schedule[0]

        if epoch >= next_epoch_step and step==0:

            for param_group in context.optimizer.param_groups:
                new_lr = param_group['lr'] * self.reduction_factor

                if self.min_lr is not None and new_lr < self.min_lr:
                    continue

                param_group['lr'] = new_lr
                print("Reducing learning rate to (epoch {}): {}".format(epoch, new_lr))

            self.epoch_schedule.pop(0)

class HistoryRecorder(AbstractCallback):
    """ Records all losses and metrics during training. """

    def __init__(self):
        self.history = defaultdict(list)

    def __call__(self, epoch, step, performance_measures, context):
        if step == len(context.train_data_loader) - 1: # Only record at end of epoch.
            for key, value in performance_measures.items():
                if type(value) == torch.Tensor:
                    value = value.item()
                self.history[key].append(value)

class ModelCheckpoint(AbstractCallback):
    """ Saves the model with lowest validation error throughout training. """

    def __init__(self, output_path, model_name='model_checkpoint.pt'):
        self.output_path = path.join(output_path, model_name)
        self.best_val_score = sys.float_info.max

    def __call__(self, epoch, step, performance_measures, context):
        if not 'val_loss' in performance_measures:
            return

        if performance_measures['val_loss'] < self.best_val_score:
            self.best_val_score = performance_measures['val_loss']
            self._save_model(context.model)

    def _save_model(self, model):
        print("Saving model at checkpoint.")
        model.eval()
        state_dict = model.state_dict()
        torch.save({'arch' : model.__class__.__name__, 'state_dict' : state_dict}, self.output_path)
        model.train()


class LayerwiseGradientNorm(AbstractCallback):
    """ Collects the layer-wise gradient norms for each epoch. """

    def __init__(self):
        self.layer_grads = dict()
        self.batch_layer_grads = dict()

    def __call__(self, epoch, step, performance_measures, context):
        """ Stores gradient norms for each batch and computes means after last batch. """
        self.store_batch_layer_grads(context.model)

        if step == (len(context.train_data_loader)-1):
            self.store_layer_grads()
            self.batch_layer_grads = dict()

    def store_batch_layer_grads(self, model):
        """ Store gradient norm of each layer for current batch. """
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            if not name in self.batch_layer_grads:
                self.batch_layer_grads[name] = []

            grad_norm = torch.sqrt(torch.sum(param.grad**2)).item()
            self.batch_layer_grads[name].append(grad_norm)
        
    def store_layer_grads(self):
        """ Compute mean of all batch steps in epoch. """
        for name, grads in self.batch_layer_grads.items():
            if not name in self.layer_grads:
                self.layer_grads[name] = []
            layer_epoch_grad = np.mean(grads)
            self.layer_grads[name].append(layer_epoch_grad)