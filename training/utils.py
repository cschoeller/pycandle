
def recursive_to_cuda(tensors, device):
    """
    Recursively iterates nested lists in depth-first order and transfers all tensors
    to specified cuda device.

    Parameters:
        tensors (list or Tensor): objects to move to specified device (can be nested)
    """
    if device is None: # keep on cpu
        return tensors

    if type(tensors) != list: # not only for torch.Tensor
        return tensors.to(device=device)

    for i in range(len(tensors)):
        tensors[i] = recursive_to_cuda(tensors[i], device)
    return tensors