from collections import defaultdict
import torch
import torch.nn.utils.clip_grad as Grads


class ModelTrainer:
    """
    This class handles the training of a pytorch model. It provides convenience
    functionality to add metrics and callbacks and is inspired by the keras API.
    """

    def __init__(self, model, optimizer, loss, epochs, train_data_loader, val_data_loader=None, custom_model_eval=False, gpu=None, clip_grads=None):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.metrics = []
        self.callbacks = []
        self.gpu = gpu
        self.custom_model_eval = custom_model_eval # pass batch_x instad model out to loss.
        self.clip_grads = clip_grads
        self.epoch_grad_history = []

    def set_metrics(self, metrics):
        """ Set metric functions which receive y_pred and y_true. """
        self.metrics = metrics
    
    def add_metric(self, metric):
        self.metrics.append(metric)

    def set_callbacks(self, callbacks):
        """
        Set callbacks, which must be callable functionals and receive
        epoch, step, loss, context. Callbacks are called after each epoch.
        """
        self.callbacks = callbacks

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def start_training(self):
        self.model.train()
        for epoch in range(self.epochs):
            self._epoch_step(epoch)
        self._close_callbacks()

    def _epoch_step(self, epoch):
        # epoch for training set.
        running_batch_loss = 0
        running_metrics = defaultdict(float)
        
        for step, (batch_x, batch_y) in enumerate(self.train_data_loader):
            batch_x, batch_y = self._recursive_to_cuda(batch_x), self._recursive_to_cuda(batch_y) # move to GPU
            loss, model_output, grad_norm = self._train_on_batch(batch_x, batch_y)
            running_batch_loss += loss.item()
            self._compute_running_metrics(model_output, batch_y, running_metrics)
            running_metrics['gradient_norm'] += grad_norm

            if self.val_data_loader and step == (len(self.train_data_loader) - 1):
                self._compute_validation_set(running_metrics)

            performance_measures = self._construct_performance_dict(step, running_batch_loss, running_metrics)
            self._print_step_info(epoch, step, performance_measures)
            self._apply_callbacks(epoch, step, performance_measures)

    def _comp_gradients(self):
            grad_sum = 0
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_sum += torch.sum(param.grad**2)
            grad_norm = torch.sqrt(grad_sum).item()
            return grad_norm

    def _train_on_batch(self, batch_x, batch_y):
            """ Compute loss depending on settings, compute gradients and apply optimization step. """
            if self.custom_model_eval: # e.g. used for sequences and other complex model evaluations
                loss, model_output = self.loss(batch_x, batch_y, self.model)
            else:
                model_output = self.model(batch_x)
                loss = self.loss(model_output, batch_y)
            self.optimizer.zero_grad() # reset gradients
            loss.backward() # backpropagation

            if self.clip_grads is not None: # Clip grads if requested
                Grads.clip_grad_norm(self.model.parameters(), self.clip_grads)
            grad_norm = self._comp_gradients() # Compute average grad

            self.optimizer.step() # apply gradients
            return loss, model_output, grad_norm

    def _compute_validation_set(self, running_metrics):
        running_val_loss = 0
        self.model.eval()
        for (batch_x, batch_y) in self.val_data_loader:
            batch_x, batch_y = self._recursive_to_cuda(batch_x), self._recursive_to_cuda(batch_y) # move to GPU
            
            if self.custom_model_eval: # e.g. used for sequences and other complex model evaluations
                val_loss, model_output = self.loss(batch_x, batch_y, self.model)
            else:
                model_output = self.model(batch_x)
                val_loss = self.loss(model_output, batch_y)

            running_val_loss += val_loss.item()
            self._compute_running_metrics(model_output, batch_y, running_metrics, prefix='val_')
        self.model.train()

        # normalize metrics and add validation loss
        running_metrics['val_loss'] = running_val_loss
        for key, value in running_metrics.items():
            if not 'val_' in key:
                continue
            running_metrics[key] = value / len(self.val_data_loader)

    def _compute_running_metrics(self, y_pred, y_true, running_metrics, prefix=''):
        if not self.custom_model_eval: # user must handle detachment himself in custom case
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

        for metric in self.metrics:
            running_metrics[prefix + metric.__name__] += metric(y_pred, y_true)

    def _construct_performance_dict(self, train_step, running_batch_loss, running_metrics):
        """ Constructs a combined dictionary of losses and metrics for callbacks. """
        performance_dict = defaultdict()
        for key, value in running_metrics.items():
            if not 'val_' in key:
                performance_dict[key] = value / (train_step + 1.)
            else:
                performance_dict[key] = value # validation metrics, already normalized.

        performance_dict['loss'] = running_batch_loss / (train_step + 1.)
        return performance_dict

    def _apply_callbacks(self, epoch, step, performance_measures):
        for callback in self.callbacks:
            callback(epoch, step, performance_measures, self)

    def _close_callbacks(self):
        for callback in self.callbacks:
            callback.close()

    def _print_step_info(self, epoch, step, performance_measures):
        output_message = "epoch {}   batch {}/{}".format(epoch+1, step, len(self.train_data_loader) - 1)
        delim = "   "
        for metric_name in sorted(list(performance_measures.keys())):
            if metric_name == 'gradient_norm':
                continue
            output_message += delim + "{}: {:.6f}".format(metric_name, performance_measures[metric_name])
        print(output_message)

    def _recursive_to_cuda(self, tensors):
        """
        Recursively iterates nested lists in depth-first order and transfers
        all pytorch tensors to specified cuda device.
        """
        if self.gpu is None: # keep on cpu
            return tensors

        if type(tensors) != list: # not only for torch.Tensor
            return tensors.to(device=self.gpu)

        for i in range(len(tensors)):
            tensors[i] = self._recursive_to_cuda(tensors[i])
        return tensors
