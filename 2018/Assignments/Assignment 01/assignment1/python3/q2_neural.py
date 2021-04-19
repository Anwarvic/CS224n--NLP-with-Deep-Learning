#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- (M x Dx) matrix, where each row is a training example.
    labels -- (M x Dy) matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    M = data.shape[0]
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))   #shape is (Dx x H)
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))        #shape is (1 x H)
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))  #shape is (H x Dy)
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))      #shape is (1 x Dy)

    ### YOUR CODE HERE: forward propagation
    #First Layer
    Z1 = np.dot(data, W1) + b1                          #shape(M x Dx) x shape(Dx x H) = shape(M x H)
    A1 = sigmoid(Z1)                                    #shape(M x H)
    #Second Layer
    Z2 = np.dot(A1, W2) + b2                            #shape(M x H) x shape(H x Dy) = shape(M x Dy)
    A2 = softmax(Z2)                                    #shape(M x Dy)  ---> A2 is the same as y_hat
    assert A2.shape == labels.shape

    #cross entropy as a loss function
    cost = -np.sum(np.log(A2) * labels) 
    ### END YOUR CODE

    


    ### YOUR CODE HERE: backward propagation
    delta = A2 - labels
    gradZ2 = delta                                      #shape(M, Dy)
    gradW2 = np.dot(A1.T, gradZ2)                       #shape(H x M) x shape(M x Dy) = shape(H x Dy)
    gradb2 = np.sum(gradZ2, axis=0, keepdims=True)      #shape(1, Dy)
    assert gradW2.shape == W2.shape and gradb2.shape == b2.shape

    gradZ1 = np.dot(gradZ2, W2.T) * sigmoid_grad(A1)    #shape(M x Dy) x shape(Dy x H) * shape(M x H) = shape(M x H)
    gradW1 = np.dot(data.T, gradZ1)                     #shape(Dx x M) x shape(M x H) = shape(Dx x H)
    gradb1 = np.sum(gradZ1, axis=0, keepdims=True)      #shape(1, H)
    assert gradW1.shape == W1.shape and gradb1.shape == b1.shape
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(
        lambda parameters: forward_backward_prop(data, labels, parameters, dimensions), 
        params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
