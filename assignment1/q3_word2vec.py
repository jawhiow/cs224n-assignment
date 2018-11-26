#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    ### YOUR CODE HERE
    # 求单位向量
    res = np.apply_along_axis(lambda x: np.sqrt(x.T.dot(x)), 1, x)
    x /= res[:, None]
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print(x)
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models
     # 注意这里是对一个单词的损失和梯度计算

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v_c} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    # predicted shape=(样本数量,)(3, ) 预测的单词
    # target = integer 目标单词的索引
    # outputVectors shape=(5, 3) 词库
    # dataset 这里没用上

    # 变成我们熟悉的内容
    # forward propagation
    # z = wx + b
    # 这里的w1表示的比如skip-gram中的中心词，对应的是公式中的vc
    W1 = predicted        # shape=(3,1)=(特征，)
    # X表示的是整个词表的单词，也就是对应了公式中的U
    X = outputVectors     # shape=(5,3)=(样本，特征)
    Z1 = np.dot(X, W1)    # shape=(5,1)=(样本，)
    # y_hat = softmax(z)
    y_hat = softmax(Z1)   # shape=(5,1)=(样本，)

    # compute cost
    # 这里交叉熵为一个单词的损失函数，所以真实值为y=1
    cost = -np.log(y_hat[target])   #shape=是一个标量

    # backward propagation
    # 这里的y_hat - y = y_hat - 1.0,代表的是交叉熵的损失函数的梯度
    # dCE / dz
    delta3 = y_hat.copy()
    delta3[target] -= 1.0    # shape=(5,1)=(样本，)
    # dCE / dW1  (在实际的公式中对应的是dCE/dvc)
    delta2 = np.dot(X.T, delta3)  # shape=(3,1)=(特征，)
    # grad (在实际公式中对应的是dCE/dU)
    dZ1 = np.outer(delta3, W1)  # shape=(5, 3) = (5, 1) * (3, 1).T

    ### END YOUR CODE
    gradPred = delta2
    grad = dZ1

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE

    W1 = predicted   # shape=(节点，1)  (3， 1)
    # forward propagation
    X = outputVectors  # shape=(样本, 特征) (5, 3)
    Z1 = np.dot(X, W1) # shape=(5, 1)=(样本, )
    y_hat = softmax(Z1)   # shape=(5, 1)

    # cost function
    cost = -np.log(y_hat[target])

    # backward propagation
    delta3 = y_hat.copy()      # shape=(5,1)=(样本，)
    delta3[target] -= 1.0
    # dCE / dW1
    delta2 = np.dot(X.T, y_hat)   # shape = (3, 5)*(5, 1) = (特征,)(3, 1)
    dZ1 = np.outer(y_hat, W1)     # shape=(5, 3) = (5, 1) * (3, 1).T
    gradPred = delta2
    grad = dZ1

    for k in range(K):
        # sigmod(-x) = 1 - sigmod(x)
        # 这里是负采样推导的梯度
        out2 = sigmoid(-1 * outputVectors[indices[k + 1]].dot(predicted))
        cost += -np.log(out2)
        grad[indices[k + 1]] += - (out2 - 1) * predicted
        gradPred += - (out2 - 1) * outputVectors[indices[k + 1]]

    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word 具体中str类型的单词
    C -- integer, context size  窗口大小
    contextWords -- list of no more than 2*C strings, the context words 窗口单词
    tokens -- a dictionary that maps words to their indices in
              the word vector list  存放了每个单词对应的索引
    inputVectors -- "input" word vectors (as rows) for all tokens  通过token中的索引，可以找到对应的单词向量
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE

    # 提取出中心词的索引
    currentWordIdx = tokens[currentWord]
    # 提取出来中心词的向量
    v_c = inputVectors[currentWordIdx]

    for j in contextWords:
        u_idx = tokens[j]
        c_cost, c_grad_in, c_grad_out = \
            word2vecCostAndGradient(v_c, u_idx, outputVectors, dataset)
        cost += c_cost
        gradIn += c_grad_in
        gradOut += c_grad_out

    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # 提取出中心词的索引
    currentWordIdx = tokens[currentWord]

    v_hat = 0.0

    # 提取出来窗口窗口词
    for j in contextWords:
        v_j_idx = tokens[j]
        v_j = inputVectors[v_j_idx]
        v_hat += v_j

    cost, gradIn, gradOut = \
        word2vecCostAndGradient(v_hat, currentWordIdx, outputVectors, dataset)

    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    middle = int(N / 2)
    inputVectors = wordVectors[:middle, :]
    outputVectors = wordVectors[middle:, :]
    for i in range(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:middle, :] += gin / batchsize / denom
        grad[middle:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
