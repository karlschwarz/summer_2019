import torch
import numpy as np
import matplotlib.pyplot as plt

def one_hot(Y, index):
    """
    encode the label of the training set as one hot vector.
    
    Arguments:
    Y -- labels of the samples, numpy vector with dimension as (m, 1)
    
    Return:
    Y_onehot -- labels of the samples, pytorch tensor with dimension as (m, index)
    """
    m = Y.shape[0]
    Y_onehot = torch.zeros(m, index, dtype=torch.float32).scatter_(1, torch.LongTensor(Y.reshape(m, 1)), torch.ones(m, index, dtype=torch.float32))
    
    return Y_onehot

def tensor_initialize(parameters):
    """
    Initialize the tensor network.
    
    Arguments:
    paramters -- python dictionary containing:
                    m -- Number of samples in one batch
                    n -- Number of features
                    Dmax -- Bond dimensions
                    l -- Sites of the output tensor
    
    Returns:
    tensors -- MPS, list of pytorch tensors
    """
    torch.manual_seed(1)
    m, n, Dmax, l, index = parameters['m'], parameters['n'], parameters['Dmax'], parameters['l'], parameters['index']
    bond_dims = [Dmax for i in range(n - 1)] + [1]
    tensors = []
    for i in range(n):
        if i != l:
            tensors.append(torch.randn(bond_dims[i-1], 2, bond_dims[i]))
        else:
            tensors.append(torch.randn(bond_dims[i-1], 2, index, bond_dims[i]))
    parameters['tensors'] = tensors
    
    return parameters

def img_show(dataset, n_row, n_column):
    
    fig = plt.figure()
    for index in range(1, 1+n_row*n_column):
        plt.subplot(n_row, n_column, index)
        plt.axis('off')
        plt.imshow(dataset[index][0].squeeze(), cmap='gray_r')
        


