#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    # 前向传播计算h和预测的y_hat

    A0 = X                      # shape(样本，特征) 20 * 10
    z1 = np.dot(A0, W1) + b1    # shape(样本，节点) 20 * 5
    A1 = sigmoid(z1)            # shape(样本，节点) 20 * 5
    z2 = np.dot(A1, W2) + b2    # shape(样本，类别) 20 * 10
    A2 = softmax(z2)            # shape(样本，列别) 20 * 10

    h = A1
    y_hat = A2


    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # 损失函数
    cost = np.sum(-np.log(y_hat[labels == 1])) / X.shape[0]

    # 新反向传播
    N = X.shape[0]  # N = 20
    y = labels                          # 维度=(20,10)=(样本，分类)
    # dCE/dz2
    # //TODO 为什么这里要除以一个N
    delta3 = (y_hat - y) / N            # 维度=(20,10)=(样本，分类)
    # dCE/dh
    delta2 = np.dot(delta3, W2.T)       # 维度=(20,5)=(样本，隐藏层节点)
    # dCE/dz1
    delta1 = delta2 * sigmoid_grad(h)   # 维度=(20,5)=(样本，隐藏层节点)
    # dCE/dx
    delta0 = np.dot(delta1, W1.T)       # 维度=(20,10)=(样本，分类)

    # dCE/dw2
    gradW2 = np.dot(h.T, delta3)        # 维度=(5,10)=(隐藏层节点，分类)
    # dCE/db2
    gradb2 = np.sum(delta3, 0, keepdims=True)  # 维度=(1,10)=(，分类)
    # dCE/dw1
    gradW1 = np.dot(X.T, delta1)        # 维度=(10,5)=(分类，隐藏层节点)
    # dCE/db1
    gradb1 = np.sum(delta1, 0)          # 维度=(1,5)=(，分类)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    # 这里是一个连接操作
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
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
            dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    print("my check pass...")
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
