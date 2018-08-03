import numpy as np
import utils.glove as glove
from utils.treebank import StanfordSentiment



def getSentenceFeatures(tokens, wordVectors, sentence):
    sentVector = np.zeros((wordVectors.shape[1],))

    ### YOUR CODE HERE
    word_indices = [tokens[word] for word in sentence]
    #average word vectors/embeddings
    sentVector = np.mean(np.take(wordVectors, indices=word_indices, axis=0), axis=0)
    ### END YOUR CODE

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector


def getRegularizationValues():
    values = None   # Assign a list of floats in the block below
    ### YOUR CODE HERE
    values = 
    ### END YOUR CODE
    return sorted(values)




if __name__ == "__main__":
    # Load the dataset
    dataset = StanfordSentiment()
    tokens = dataset.tokens()

    # Using pre-trained vectors
    wordVectors = glove.loadWordVectors(tokens)
    dimVectors = wordVectors.shape[1]

    # Load the train set
    trainset = dataset.getTrainSentences()
    nTrain = len(trainset)
    trainFeatures = np.zeros((nTrain, dimVectors))
    trainLabels = np.zeros((nTrain,), dtype=np.int32)
    for i in xrange(nTrain):
        words, trainLabels[i] = trainset[i]
        trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)
        break
    print trainFeatures[0, :]
    # We will save our results from each run
    results = []
    regValues = getRegularizationValues()
    for reg in regValues:
        print "Training for reg=%f" % reg
        # Note: add a very small number to regularization to please the library
        clf = LogisticRegression(C=1.0/(reg + 1e-12))
        clf.fit(trainFeatures, trainLabels)

        # Test on train set
        pred = clf.predict(trainFeatures)
        trainAccuracy = accuracy(trainLabels, pred)
        print "Train accuracy (%%): %f" % trainAccuracy

        # Test on dev set
        pred = clf.predict(devFeatures)
        devAccuracy = accuracy(devLabels, pred)
        print "Dev accuracy (%%): %f" % devAccuracy

        # Test on test set
        # Note: always running on test is poor style. Typically, you should
        # do this only after validation.
        pred = clf.predict(testFeatures)
        testAccuracy = accuracy(testLabels, pred)