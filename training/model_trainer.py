from collections import defaultdict
import torch
import torch.nn.utils.clip_grad as Grads
from .utils import recursive_to_cuda


class ModelTrainer:
    """
    This class handles the training of a pytorch model. It provides convenience
    functionality to add metrics and callbacks and is inspired by the keras API.

    Args:
        model (nn.Module): model to be trained
        optimizer (optim.Optimizer): optimizer used for training, e.g. torch.optim.Adam
        loss (function): loss function that either accepts (model_output, label) or (input, label, model) if custom_model_eval is true
        epochs (int): epochs to train
        train_data_loader (utils.data.DataLoader): training data
        val_data_loader (utils.data.DataLoader, optional): validation data
        custom_model_eval (boolean, optional): enables training mode where the model is passed to the loss function and evaluated manually
        device (int, optional): if not set training runs on cpu, otherwise an int is expected that determines the training gpu
        clip_grads (float, optional): if set training gradients will be clipped at specified norm

    Example:
        >>> model_trainer = ModelTrainer(model, optimizer, F.nll_loss, num_epochs, train_loader, gpu=0)
        >>> model_trainer.start_training()
    """

    def __init__(self, model, optimizer, loss, epochs, train_data_loader, val_data_loader=None, 
                custom_model_eval=False, device=None, clip_grads=None, scheduler=None):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.loss = loss
        self._epochs = epochs
        self._metrics = []
        self._callbacks = []
        self._device = device
        self._custom_model_eval = custom_model_eval
        self._clip_grads = clip_grads
        self.scheduler = scheduler
        self._stop_training = False # used stop training externally
        
    def set_metrics(self, metrics):
        """
        Set metric functions that receive y_pred and y_true. Metrics are expected to return
        a basic numeric type like float or int.
        """
        self._metrics = metrics
    
    def add_metric(self, metric):
        self._metrics.append(metric)

    def set_callbacks(self, callbacks):
        """
        Set callbacks that are callable functionals and receive epoch, step, loss, context.
        Context is a pointer to the ModelTrainer instance. Callbacks are called after each
        processed batch.
        """
        self._callbacks = callbacks

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def start_training(self):
        self.model.train() # train mode
        for epoch in range(1, self._epochs + 1):
            self._epoch_step(epoch)

            if self._stop_training:
                break
        
        self._close_callbacks()

    def _epoch_step(self, epoch):
        """ Execute one training epoch. """
        running_batch_loss = 0
        running_metrics = defaultdict(float)
        
        for step, batch in enumerate(self.train_data_loader):
            batch = recursive_to_cuda(batch, self._device) # move to GPU

            # compute training batch
            loss, model_output, grad_norm = self._train_on_batch(batch)
            running_batch_loss += loss.item()

            # compute metrics
            self._compute_running_metrics(model_output, batch, running_metrics)
            running_metrics['gradient_norm'] += grad_norm # add grad norm to metrics

            # evaluate validation set at end of epoch
            if self.val_data_loader and step == (len(self.train_data_loader) - 1):
                self._compute_validation_error(running_metrics)

            # print current loss and metrics and provide it to callbacks
            performance_measures = self._construct_performance_dict(step, running_batch_loss, running_metrics)
            self._print_step_info(epoch, step, performance_measures)
            self._apply_callbacks(epoch, step, performance_measures)

    def _comp_gradient_norm(self):
            """ Compute the gradient norm for all model parameters. """
            grad_sum = 0
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_sum += torch.sum(param.grad**2)
            grad_norm = torch.sqrt(grad_sum).item()
            return grad_norm

    def _train_on_batch(self, batch):
            """ Compute loss, compute gradients and apply optimization step for given batch. """
            # run lr scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # evaluate loss
            if self._custom_model_eval: # custom evaluation
                loss, model_output = self.loss(batch, self.model)
            else: # regular supervised learning
                batch_x, batch_y = batch
                model_output = self.model(batch_x)
                loss = self.loss(model_output, batch_y)

            self.optimizer.zero_grad() # reset gradients
            loss.backward() # backpropagation

            # gradient clipping
            if self._clip_grads is not None:
                Grads.clip_grad_norm(self.model.parameters(), self._clip_grads)

            grad_norm = self._comp_gradient_norm() # compute average gradient norm

            self.optimizer.step() # apply optimization step
            return loss, model_output, grad_norm

    def _compute_validation_error(self, running_metrics):
        """ Evaluate the model's validation error. """
        running_val_loss = 0

        self.model.eval()
        for batch in self.val_data_loader:
            batch = recursive_to_cuda(batch, self._device)
            
            # evaluate loss
            batch_x, batch_y = batch
            if self._custom_model_eval: # e.g. used for sequences and other complex model evaluations
                val_loss, model_output = self.loss(batch, self.model)
            else:
                model_output = self.model(batch_x)
                val_loss = self.loss(model_output, batch_y)

            # compute running validation loss and metrics. add 'val_' prefix to all measures.
            running_val_loss += val_loss.item()
            self._compute_running_metrics(model_output, batch, running_metrics, prefix='val_')
        self.model.train()

        # add loss to metrics and normalize all validation measures
        running_metrics['val_loss'] = running_val_loss
        for key, value in running_metrics.items():
            if not 'val_' in key:
                continue
            running_metrics[key] = value / len(self.val_data_loader)

    def _compute_running_metrics(self, y_pred, batch, running_metrics, prefix=''):
        """
        Computes all metrics based on predictions and batches and adds them to the metrics
        dictionary. Allows to prepend a prefix to the metric names in the dictionary.
        """
        for metric in self._metrics:
            if self._custom_model_eval:
                metric_result = metric(y_pred, batch)
            else:
                batch_y = batch[1]
                metric_result = metric(y_pred, batch_y)

            # convert to float if metric returned tensor
            if type(metric_result) == torch.Tensor:
                metric_result = metric_result.item()
            
            running_metrics[prefix + metric.__name__] += metric_result

    def _construct_performance_dict(self, train_step, running_batch_loss, running_metrics):
        """
        Constructs a combined dictionary of losses and metrics for callbacks based on
        the current running averages.
        """
        performance_dict = defaultdict()
        for key, value in running_metrics.items():

            if not 'val_' in key:
                performance_dict[key] = value / (train_step + 1.)
            else:
                performance_dict[key] = value # validation metrics, already normalized

        performance_dict['loss'] = running_batch_loss / (train_step + 1.)
        return performance_dict

    def _apply_callbacks(self, epoch, step, performance_measures):
        """ Call all registered callbacks with current batch information. """
        for callback in self._callbacks:
            callback(epoch, step, performance_measures, self)

    def _close_callbacks(self):
        """ Signal callbacks training is finished. """
        for callback in self._callbacks:
            callback.close()

    def _print_step_info(self, epoch, step, performance_measures):
        """ Print running averages for loss and metrics during training. """
        output_message = "epoch {}   batch {}/{}".format(epoch, step, len(self.train_data_loader) - 1)
        delim = "   "
        for metric_name in sorted(list(performance_measures.keys())):
            if metric_name == 'gradient_norm':
                continue
            output_message += delim + "{}: {:.6f}".format(metric_name, performance_measures[metric_name])
        print(output_message)
