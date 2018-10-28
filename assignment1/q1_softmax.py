import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        # x = e_x / np.sum(e_x)
        # 要点1：使用分子分母同除以e的max次方的方法，来防止数超过范围。 也就是下面的 x-max
        # 要点2：使用np.apply_along_axis来对特定维度进行特点方法的操作，方法可以使用lambda表达式代替
        e_x = np.apply_along_axis(lambda t: np.exp(t - np.max(t)), 1, x)
        x = np.apply_along_axis(lambda t: t * 1.0 / np.sum(t), 1, e_x)
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        e_x = np.exp(x - np.max(x))
        x = e_x * 1.0 / np.sum(e_x)
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


# from q1_softmax import softmax
# return 20 if softmax(测试值) == 正确值 else 0


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR CODE HERE
    res = softmax(np.array([[1001, 1004], [3, 4]]))
    print("res is :")
    print(res)
    ans = np.array([[0.04742587, 0.95257413],
                    [0.26894142, 0.73105858]])
    print("ans is :")
    print(ans)
    assert np.allclose(res, ans, rtol=1e-05, atol=1e-06)
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
