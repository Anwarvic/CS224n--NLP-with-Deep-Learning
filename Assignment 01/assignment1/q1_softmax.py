import numpy as np
from numpy import newaxis


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- An N-dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape
    ### YOUR CODE HERE:
    if len(x.shape) == 1:
        # Vector
        #let's use the softmax property which states that softmax(x)==softmax(x+c).
        #I'm going to use c as c = -max(x)
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    elif len(x.shape) == 2:
        #vector but casted as matrix
        if x.shape[1] == 1:
            x = x[:, 0]
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
            #change the shape from (a, ) to (a, 1)
            x = x[:, newaxis]
        # Matrix
        else:
            x = x - np.max(x, axis=1, keepdims=True)
            x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    else:
    	print("Matrix needs to be only 2D")
    ### END YOUR CODE
    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06), test1
    print("\tPassed the first test")

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06), test2
    print("\tPassed the second test")

    test3 = softmax(np.array([[-1001,-1002]]))
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06), test3
    print("\tPassed the third test")



def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR CODE HERE
    test = softmax(np.array([[1],[2]]))
    ans = [[0.26894142], 
           [0.73105858]]
    assert np.allclose(test, ans, rtol=1e-05, atol=1e-06), test
    print("\tPassed your test!!")
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
