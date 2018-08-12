#!/usr/bin/env python
import random
import numpy as np


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """
    #the following two lines are used to set the seed
    rndstate = random.getstate()
    random.setstate(rndstate)
    _, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index #index
        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        """
        Check Gradient Checking by Andrew Ng. from [here](https://www.youtube.com/watch?v=P6EtCVrvYPU).
        In summary, we have to calculate four parameters:
        -> x+h
        -> x-h
        -> f(x+h)
        -> f(x-h)
        Then, calculate;
                   f(x+h) - f(x-h)
        numgrad = -----------------; where h-> 0
                         2h
        Let's do that now:
        """
        original_x = x[ix] #original value
        #-----calculate f(x+h)-----
        x[ix] += h                 #x+h
        random.setstate(rndstate)
        #notice, we are calculating f() over the whole matrix
        plus_h = f(x)[0]         #f(x+h)

        #-----calculate f(x-h)-----
        x[ix] = original_x - h     #x-h
        random.setstate(rndstate)
        minus_h = f(x)[0]        # f(x-h)

        #verify
        numgrad = (plus_h - minus_h)/ (2*h)
        #reset x
        x[ix] = original_x
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "\tFirst gradient error found at index %s" % str(ix)
            print "\tTrue gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
