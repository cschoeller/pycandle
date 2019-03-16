import torch
import torch.nn as nn


class ConvMinGRU(nn.Module):
    """ 
    Convolutional Minimal Gated Recurrent Unit Layer. Assumes input size to be equal to output size
    and applies same padding.
    MinGRU is a simpler and smaller version of the GRU and shows equivalent performance with fewer
    parameters (https://arxiv.org/pdf/1603.09420.pdf).
    """

    def __init__(self, channels, kernel_size):
        super(ConvMinGRU, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.same_padding = kernel_size // 2 # same padding

        self.forget_gate_x = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)
        self.forget_gate_h = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)
        self.out_gate_x = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)
        self.out_gate_h = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)

        # initialize biases with zeros, no initial forgetting
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.bias.data.fill_(1.) #TODO verify correct
    
    def forward(self, x, last_hidden_state=None):
        if last_hidden_state is None: # initialize hidden state
            last_hidden_state = torch.zeros_like(x)

        f = torch.sigmoid(self.forget_gate_x(x) + self.forget_gate_h(last_hidden_state))
        hidden_state_changes = torch.tanh(self.out_gate_x(x) + self.out_gate_h(f * last_hidden_state))
        new_hidden_state = f * last_hidden_state + (1. - f) * hidden_state_changes
        return new_hidden_state


class ConvGRU(nn.Module):
    """
    Convolutional Gated Recurrent Unit Layer. Assumes input size to be equal to output size and applies
    same padding.
    """

    def __init__(self, channels, kernel_size):
        super(ConvGRU, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.same_padding = kernel_size // 2 # same padding

        self.update_gate_x = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)
        self.update_gate_h = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)
        self.reset_gate_x = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)
        self.reset_gate_h = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)
        self.out_gate_x = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)
        self.out_gate_h = nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.same_padding)

        # initialize biases with zeros, no initial forgetting
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.bias.data.fill_(1.) #TODO verify correct

    def forward(self, x, last_hidden_state=None):
        if last_hidden_state is None: # initialize hidden state
            last_hidden_state = torch.zeros_like(x) # automatically on cuda if x on cuda

        r = torch.sigmoid(self.reset_gate_x(x) + self.reset_gate_h(last_hidden_state)) # reset gate
        hidden_state_modification = torch.tanh(self.out_gate_x(x) + self.out_gate_h(r * last_hidden_state)) # state modification
        z = torch.sigmoid(self.update_gate_x(x) + self.update_gate_h(last_hidden_state)) # update gate
        new_hidden_state = z * last_hidden_state + (1. - z) * hidden_state_modification # apply update
        return new_hidden_state