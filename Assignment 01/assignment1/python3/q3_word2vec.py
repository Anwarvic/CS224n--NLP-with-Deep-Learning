#!/usr/bin/env python
import random
import numpy as np
from numpy import newaxis

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    ### END YOUR CODE
    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06), x
    print("passed")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (v_hat in
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
    V = predicted[:, newaxis]                      #shape(H x 1) where H is the number of neurons in the hidden layer
    U = outputVectors                              #shape(C x H) where C is the vocabulary size
    model_predictions = softmax(np.dot(U, V))      #shape(C x H) x shape(H x 1) = shape(C x 1)

    #cross entropy as a loss function
    cost = -np.log(model_predictions[target])      #shape(C x 1)

    #update
    model_predictions[target] -= 1
    
    #now, let's return the gradients
    gradPred = np.dot(U.T, model_predictions)      #shape(H x C) x shape(C x 1) = shape(H x 1)
    assert gradPred.shape == V.shape
    grad = np.dot(model_predictions, V.T)          #shape(C x 1) x shape(1 x H) = shape(C x LH)
    assert grad.shape == U.shape
    #Change the shape from (H x 1) to (H,)
    gradPred = gradPred[:, 0]
    ### END YOUR CODE

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


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
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
    indices.extend(getNegativeSamples(target, dataset, K)) #list of length = K+1

    ### YOUR CODE HERE
    V = predicted[:, newaxis]                          #shape(H x 1) where H is the number of neurons in the hidden layer
    U = outputVectors                                  #shape(C x H) where C is the vocabulary size
    U_target = outputVectors[target][newaxis]          #shape(1 x H)
    model_prediction = sigmoid(np.dot(U_target, V))    #shape(1 x H) x shape(H x 1) = shape(1 x 1)
    #cross entropy cost
    cost = -np.log(model_prediction)                   #shape(1 x 1)
    #update
    model_prediction -= 1
    #gradients
    gradPred = np.dot(U_target.T, model_prediction)    #shape(H x 1) * shape(1 x 1) = shape(H x 1)
    assert gradPred.shape == V.shape, gradPred.shape
    grad = np.zeros(U.shape)                           #shape(C x H)
    grad[target] = np.dot(model_prediction, V.T)       #shape(1 x 1) x shape(1 x H) = shape(1 x H)
    assert grad[target][newaxis].shape == U_target.shape and grad.shape == U.shape
    #----- up till here, it's the same as before -----
    for k in range(1, K+1): #we skipped the first as it's the target
        context_idx = indices[k]
        # print(context_idx)
        U_context = U[context_idx][newaxis]                     #shape(1 x H)
        model_prediction = sigmoid(np.dot(U_context, V))        #shape(1 x 1)
        #cost (subtracted 1 because they are negative samples)
        cost -= np.log(1 - model_prediction)                    #shape(1 x 1)
        #update gradients
        gradPred += np.dot(U_context.T, model_prediction)       #shape(H x 1)
        grad[context_idx][newaxis] +=np.dot(model_prediction, V.T)

    #Change the shape from (H x 1) to (H,)
    gradPred = gradPred[:, 0]
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words (keys) to their indices in
              the word vector list (value)
    inputVectors -- "input" word vectors (as rows) for all tokens
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
    gradIn = np.zeros(inputVectors.shape)             #shape(C x H)
    gradOut = np.zeros(outputVectors.shape)           #shape(C x H)
    ### YOUR CODE HERE
    U = outputVectors                                 #shape(C x H)
    center_idx = tokens[currentWord]
    for word in contextWords:
        target_idx = tokens[word]
        V = inputVectors[center_idx]                  #shape(H x 1)
        tmp_cost, tmp_gradPred, tmp_gradOut = \
                    word2vecCostAndGradient(V, target_idx, U, dataset)
        cost += tmp_cost
        gradIn[center_idx] += tmp_gradPred
        gradOut += tmp_gradOut
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
    U = outputVectors                                 #shape(C x H)
    context_indices = [tokens[word] for word in contextWords]
    V = inputVectors[context_indices]                 #shape(W x H) where W is the Context-Window size
    target_idx = tokens[currentWord]
    #reduce V's shape to (H,) to be valid for 'word2vecCostAndGradient' methods
    V = np.sum(V, axis=0)
    tmp_cost, tmp_gradPred, tmp_gradOut = \
                word2vecCostAndGradient(V, target_idx, U, dataset)
    cost += tmp_cost
    #update only the context words
    for i in context_indices:
        gradIn[i] += tmp_gradPred
    gradOut += tmp_gradOut
    """CAUTION: 
    Don't update 'gradIn' like this:
    >>> gradIn[context_indices] += tmp_gradPred
    Actually, I don't know why, but it didn't pass the tests.. 
    I have spent more than two hours in this piece-of-sh*t line.
    """
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
    inputVectors = wordVectors[:int(N/2),:]
    outputVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:int(N/2), :] += gin / batchsize / denom
        grad[int(N/2):, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        #returns random integer in the range [0, 4]
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
