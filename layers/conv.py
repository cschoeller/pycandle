import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConv2d(nn.Module):
    """
    Adaptive convolutional layer, takes an additional input which forms the convolution kernel and then applies it
    to primary input. In this case the filter manifold network which generates the kernel is fixed
    to a size of 3 layers which double in the number of output neurons until the final output dimension is
    reached.

    The implementation with grouped convolutions is based on this answer on StackOverflow:
    https://stackoverflow.com/questions/42068999/tensorflow-convolutions-with-different-filter-for-each-sample-in-the-mini-batch

    Further relevant links to understand the implementation:
    https://discuss.pytorch.org/t/conv2d-certain-values-for-groups-and-out-channels-dont-work/14228
    https://github.com/pytorch/pytorch/issues/3653

    Literature: Kang et al., NIPS 2017, https://papers.nips.cc/paper/6976-incorporating-side-information-by-adaptive-convolution

    Parameters:
    channels_in - 2D input tensor
    channels_out - 2D output tensor
    features_in - 1D input tensor, the adaptive features
    kernel_size - scalar or tuple, if tuple its assumed to be of quadratic shape, e.g. (3,3), not (3,2)

    Notes:
    At the moment only supports same padding. To support batch-wise computation grouped convolutions are exploited as explained in the
    links.
    """

    def __init__(self, channels_in, channels_out, features_in, kernel_size):
        super(AdaptiveConv2d, self).__init__()

        # convert kernel tuple to scalar
        if type(kernel_size) == tuple:
            assert(len(kernel_size) == 2)
            assert(kernel_size[0] == kernel_size[1])
            kernel_size = kernel_size[0]

        self.features_in = features_in
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.same_padding = self.kernel_size // 2

        # represents the filter manifold network from the paper
        self.num_kernel_weights = channels_in * channels_out * kernel_size**2
        self.num_bias_weights = channels_out
        num_neurons_out = self.num_kernel_weights + self.num_bias_weights # last sum for bias
        layer_1_neurons = 20#(num_neurons_out - features_in) // 4
        layer_2_neurons = 30#(num_neurons_out - features_in) // 2

        self.dense1 = nn.Linear(features_in, layer_1_neurons)
        self.dense2 = nn.Linear(layer_1_neurons, layer_2_neurons)
        self.dense3 = nn.Linear(layer_2_neurons, num_neurons_out)

    def forward(self, x, z):
        """
        Takes a tensor x and an auxillary input z. This is implemented as a hack, exploiting
        grouped convolutions to process a whole input batch at once while applying multiple convolutions
        (channels and kernels) to each batch-instance without a loop.
        """
        # run filter manifold network to get convolution weights
        z = F.relu(self.dense1(z))
        z = F.relu(self.dense2(z))
        adaptive_weights = F.relu(self.dense3(z))

        batch_size, x_c_in, x_h, x_w = x.shape
        x = x.view(1, batch_size * x_c_in, x_h, x_w)

        # initialize convolution only here as weights are assigned new at each pass
        conv = nn.Conv2d(batch_size*self.channels_in, batch_size*self.channels_out, self.kernel_size, \
                        padding=self.same_padding, groups=batch_size)

        # assign computed weights to convolution
        conv_kernel_weights = adaptive_weights[:,:self.num_kernel_weights].contiguous().view(batch_size*self.channels_out, self.channels_in, self.kernel_size, self.kernel_size)
        conv_bias_weights = adaptive_weights[:,-self.num_bias_weights:].contiguous().view(-1) #TODO be aware of contiguous (problematic?)
        conv._parameters['weight'] = conv_kernel_weights
        conv._parameters['bias'] = conv_bias_weights

        # apply convolution on input and reshape in desired output
        x = F.relu(conv(x))
        x = x.view(-1, self.channels_out, x_h, x_w)
        return x
