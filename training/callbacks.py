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
    """ Saves the model with lowest validation error throughout training. """

    def __init__(self, output_path, model_name='model_checkpoint.pt'):
        self.output_path = path.join(output_path, model_name)
        self.best_val_score = sys.float_info.max

    def __call__(self, epoch, step, performance_measures, context):
        if not 'val_loss' in performance_measures:
            return

        if performance_measures['val_loss'] < self.best_val_score:
            self.best_val_score = performance_measures['val_loss']
            self._save_model(context.model, context.optimizer)

    def _save_model(self, model, optimizer):
        print("Saving model at checkpoint.")
        model.eval()
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        """
        When saving a general checkpoint, to be used for either inference or resuming training,
        you must save more than just the model’s state_dict.
        It is important to also save the optimizer’s state_dict, as this
        contains buffers and parameters that are updated as the model train
        """
        torch.save({'arch' : model.__class__.__name__,
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
            
class EarlyStopping(AbstractCallback):
    """
    Early Stopping to terminate training early if monitored metric is not improved
    after a number of epochs defined by patience argument.

    Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvement before terminating.
            the counter be reset after each improvement

    """
    def __init__(self, monitor='val_loss', min_delta=0, patience = 5):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.last_best = sys.float_info.max
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, epoch, step, performance_measures, context):
        #Run callback only at end of epoch.
        if step == len(context.train_data_loader) - 1:

            if not self.monitor in performance_measures:
                return
            current_loss = performance_measures[self.monitor]
            if (current_loss - self.last_best) < -self.min_delta:
                self.last_best = current_loss
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience:
                self.stopped_epoch = epoch + 1
                #context is a reference to the ModelTrainer
                context._stop_training = True
                print('\nTerminated Training for Early Stopping at Epoch %04i' % (self.stopped_epoch))
                
                
class ReduceLROnPlateau(AbstractCallback):
    """
    Reduce the learning rate if the train or validation loss plateaus
    """

    def __init__(self,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 epsilon=0,
                 cooldown=0,
                 min_lr=0,
                 verbose=0):
        """
        Reduce the learning rate if the train or validation loss plateaus
        Arguments
        ---------
        monitor : string in {'loss', 'val_loss'}
            which metric to monitor
        factor : floar
            factor to decrease learning rate by
        patience : integer
            number of epochs to wait for loss improvement before reducing lr
        epsilon : float
            how much improvement must be made to reset patience
        cooldown : integer
            number of epochs to cooldown after a lr reduction
        min_lr : float
            minimum value to ever let the learning rate decrease to
        verbose : integer
            whether to print reduction to console
        """
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best_loss = 1e15
        self._reset()
        super(ReduceLROnPlateau, self).__init__()

    def _reset(self):
        """
        Reset the wait and cooldown counters
        """
        self.monitor_op = lambda a, b: (a - b) < -self.epsilon
        self.best_loss = 1e15
        self.cooldown_counter = 0
        self.wait = 0


    def __call__(self, epoch, step, performance_measures, context):

        if not self.monitor in performance_measures:
            return
        #Run callback only at end of epoch.
        if step == len(context.train_data_loader) - 1:
            current_loss = performance_measures[self.monitor]
            # if in cooldown phase
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.wait = 0
            # if loss improved, grab new loss and reset wait counter
            if self.monitor_op(current_loss, self.best_loss):
                self.best_loss = current_loss
                self.wait = 0
            # loss didnt improve, and not in cooldown phase
            elif not (self.cooldown_counter > 0):
                if self.wait >= self.patience:
                    for param_group in context.optimizer.param_groups:
                        old_lr = param_group['lr']
                        if old_lr > self.min_lr + 1e-4:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing lr from %0.3f to %0.3f' %
                                    (epoch, old_lr, new_lr))
                            param_group['lr'] = new_lr
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                    self.wait += 1          
