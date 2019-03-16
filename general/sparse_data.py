import numpy as np
from scipy import sparse

def to_sparse_sample(sample):
    """
    Converts a sample (multiple grid maps) to a sparse representation.
    """
    sparse_sample = []
    for grid_map in sample:
        sparse_sample.append(sparse.csr_matrix(grid_map))
    return np.array(sparse_sample)

def to_dense_sample(sparse_sample):
    """
    Converts a sample (multiple grid maps) to a dense representation.
    """
    dense_sample = []
    for sparse_grid_map in sparse_sample:
        dense_grid_map = sparse.csr_matrix.todense(sparse_grid_map)
        dense_sample.append(dense_grid_map)
    return np.array(dense_sample)

def to_dense_batch(sparse_batch):
    """
    Converts a batch of shape [data, labels] to dense grid maps.
    """
    dense_data, dense_labels = [], []
    for i in range(len(sparse_batch[0])):
        dense_sample = to_dense_sample(sparse_batch[0][i])
        dense_label = to_dense_sample(sparse_batch[1][i])
        dense_data.append(dense_sample)
        dense_labels.append(dense_label)
    return [dense_data, dense_labels]

def make_dense(sparse_data):
    """
    Converts a list of sparse samples to a dense numpy array.
    """
    dense_data = []
    for sparse_sample in sparse_data:
        dense_sample = sparse.csr_matrix.todense(sparse_sample)
        dense_data.append(dense_sample)
    return np.array(dense_data)