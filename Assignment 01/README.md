# Assignment #1

## Requirements

To be able to solve this assignment, we need to install a few packages which can be found in the `requirements.txt` file and a few datasets which can be downloaded by simply running the bash script `get_datasets.sh`. 

---

## q1

According to the first question, there is only one script which is `q1_softmax.py` at which we are going to implement the Softmax activation function that will be used in the remaining questions. In other words, we need to implement the following function:

![Softmax](http://www.mediafire.com/convkey/8c36/2asbi4gjc4iai2hzg.jpg)

---

## q2

According to the second question, there are three scripts:

- `q2_sigmoid.py`: Which contains two functions. The first functions applies the sigmoid function and the second applies its gradient .
- `q2_gradcheck.py`: Where we will perform "Gradient Checking". It's a way to check the right implementation of Gradient Descent.
- `q2_neural.py`: Where we will create a simple neural network and perform both the forward propagation and the back-propagation algorithms.



### q2_sigmoid

In this script, we have to implement just two functions, let's discuss them one by one:

#### 1) sigmoid

This is a super simple function that applies the sigmoid function upon a given numpy.array. We are simple going to perform this equation:

![Sigmoid](http://www.mediafire.com/convkey/6dcb/958jrjh8icaddh4zg.jpg)

#### 2) sigmoid_grad

This is also a super simple function that applies gradient upon the sigmoid function . We are simple going to perform this equation:

![Sigmoid Gradient](http://www.mediafire.com/convkey/8b43/bpxf82cf1wawq8jzg.jpg)

### q2_gradcheck

In this script, we have to implement just one function which is:

#### gradcheck_naive

Gradient Checking is a techniques used to make sure that our implementation of Gradient Descent/Ascent is completely correct. It will be very useful to check [Gradient Checking Video](https://www.youtube.com/watch?v=P6EtCVrvYPU) made by Andrew Ng. which was part of the Deep Learning Specialization at Coursera.

In summary, we have to calculate four parameters:

- `x+h`
- `x-h`
- `f(x+h)`
- `f(x-h)`

Then, calculate;

![gradient_checking](http://www.mediafire.com/convkey/ec8b/9sak4wui2qobdc4zg.jpg)

Where `h` (GIVEN) is a very small number that tends to zero. Then, check the difference between the approximated gradient calculated from the previous formula and the real gradient. The difference should be a very small number.

### q2_neural

In this script, we have to implement just one function which is:

#### forward_backward_prop

In this function we are going to perform the Forward Propagation and Backward Propagation. Then, we are going to test this function over a simple neural network with two layers.

In Forward Propagation, we will use the `sigmoid` function that we built in *q2* as activation function in the first two layers. Then, we are going to use the `Softmax` function, that we build in *q1*, with the last layer.

The forward propagation is roughly straight-forward, but the backward propagation is trickier. So, these functions will come in handy:

![Backprop](http://www.mediafire.com/convkey/0d50/nxag4p60uei1m0azg.jpg)

Where:

- `dz2` is the gradient of the weights of the second layer
- `dz1` is the gradient of the weights of the first layer
- `db2` is the gradient of the bias of the second layer
- `db1` is the gradient of the bias of the first layer
- `a1` is the output of the first layer
- `a2` is the output of the second layer
- `W2` is the weights of the second layer
- `W1` is the weights of the first layer
- `x` is the input.
- `g1'()`is the gradient of the Softmax activation function.

If you couldn't implement this function, no worries... you won't use it later.

---

## q3

According to the third question, there are three scripts:

- `q3_word2vec.py`: Which contains the Word2Vec model with the two flavors (Skipgram and CBOW).
- `q3_sgd.py`: 
- `q3_run.py`:



### q3_word2vec 

This model has different functions in it, let's discuss them one by one:

#### 1) normalizeRows

Since the range of values of raw data varies widely, in some machine learning algorithms,objective functions will not work properly without normalization. We should put in mind that re-scaling a feature doesn’t mean that we are changing it. For example, if someone told you that he got 8/10 in an exam, it’s the same if he said he got 80%  ...right? That’s what we mean by scaling, we are just performing some formula over all the data points to make it more comprehensible as we can see in the following graph:

![Normalization](http://www.mediafire.com/convkey/bb95/q92a7akz88dn8g8zg.jpg)

There are multiple types of normalization, here we are going to use Frobenius norm which follows the following formula:

![Frobenius Norm](http://www.mediafire.com/convkey/8d9e/3fp7k0jpoajjm3ozg.jpg)

#### 2) softmaxCostAndGradient

You can check the function description within the script, but I couldn't understand what this function does especially when there is no test for it and to be able to run it, you have to complete the whole Skipgram/CBOW model.

So, I will explain what this function does here in this part, but first you need
to change some of the variable names just to make things simpler as the variable names are not indicative:
-> 'predicted' is the input of the last layer, so we will call it `V`.
-> 'outputVectors' is the weights of the output layer and we will call it `U`.

To make things a little simple, I have drawn a simple graph of the output layer to explain what I mean:

![Last-layer Cost using Softmax](http://www.mediafire.com/convkey/7b19/9vg7m2gfprr6e5pzg.jpg)

Let's explain some of the things about this graph:

- `H` is the number of neurons in the hidden layer.
- `C` is the number of neurons in the last layer which should be equal to the vocabulary size.
- Then, the `model_prediction` will be calculated from this equation:

![Model Prediction](http://www.mediafire.com/convkey/e3a3/ybtua7dh9y28utazg.jpg)

Now, we need to calculate the "Cross Entropy" cost function:

![Cross Entropy, its gradient](http://www.mediafire.com/convkey/54a2/c0kejpq1te44ozuzg.jpg)

Where $Y$ is the true label, and $\hat{Y}$ is our `model_prediction`. In our case, $Y$ is a one-hot vector (just zeros and ones), then the cost will be the summation of ONLY the values where $Y$ is one. Then, the gradient $\Delta$ will exist at the values where $Y$ is 1. Therefore, our functions will be:

![Binary Cross Entropy](http://www.mediafire.com/convkey/aa4b/arnsv6yrzx4rdj3zg.jpg)

After that we update our parameters, putting in mind that:

- `grad` is the gradient of `outputVectors` which is `U`. So, they must have the same dimensions.
- `gradPred` is the gradient of the `predicted` which is `V`. So, they must have the same dimensions.

#### 3) negSamplingCostAndGradient

Here, we are going to do the same as we have done with the previous function but using Negative Sampling. This function is pretty similar to the previous function except for two things actually:

- This function uses `sigmoid` activation function rather than `softmax`.
- Use $U_{target}$ instead of $U$ like so:

![Model Prediction using Sigmoid](http://www.mediafire.com/convkey/2956/39wqdie7jbsx44ozg.jpg)

- We update our parameters row-by-row for a number of times specified by the variable `K`.

So, if you did the previous function, repeat its steps with these small modification in mind.



#### 4) skipgram

Now, we are ready to build our Skipgram model. The following graph shows the whole Skipgram model:

![Skipgram](http://www.mediafire.com/convkey/df1e/5xxjuslh5quxacczg.jpg)

- `H` is the number of neurons in the hidden layer.
- `C` is the number of neurons in the input layer and the last layer which should be equal to the vocabulary size.
- `inputVectors` is the hidden-layer weights (weights between the **input layer** and **hidden layer**). Its shape should be `C x H`.
- `V` is the values of **hidden layer** neurons. Its shape should be `H x 1`.
- `U` is the output weights (weights between the **hidden layer** and the **output layer**. Its shape should be `C x H`.
- Finally, the `model_prediction` is the output of the Skipgram model. Its shape should be `H x 1`.

Implementing this part should be pretty straight-forward!!



#### 5) cbow (OPTIONAL)

Now to the last function we need to implement in this part. CBOW or Continuous Bag-of-Words is another flavor of the Word2Vec model. The major difference between Skip-Gram and CBOW is the shape of the input layer of the network. So, if we understand the Skip-Gram model then the CBOW model should be quite straight-forward because in many ways they are mirror images of each other. 

To understand that model, let’s consider having the following sentence *“the quick brown fox jumps over the lazy dog.”*. Here, we are going to use a small window size of 2 just for the example. A more reasonable size would be 5 or more. The word highlighted in blue is called *center* in the Skipgram model and *target* in the CBOW model.

![Skipgram Vs CBOW](http://www.mediafire.com/convkey/d994/h5fhg0c03i6hp4izg.jpg)

As we can see, it's similar to the Skipgram model. So, if you could manage to implement the Skipgram model, implementing CBOW should be so easy!! Here, I'm going to enumerate the minor differences between Skipgram and CBOW. Putting these changes in mind, implementing CBOW should be easy:

- `inputVector` will be a **matrix** instead of a *vector* and that's because it contains all the context words. So, its shape should be `W x H` where `W` is the context-window size.




### q3_sgd

In this problem, we will implement a dummy algorithm for Stochastic Gradient Descent or (SGD) for short. The difference between GD and SGD is that the first needs to iterate over the whole training dataset before updating its parameters, unlike SGD which updates its parameters each data point. So, we can consider SGD to be equivalent to gradient descent with just one example. 

The update rule does not change. What changes is that we would be computing gradients on just one training example at a time, rather than on the whole training set. When the training set is large, SGD can be faster than Gradient Descent. So, it's an important algorithm to be put in our toolkit.

Solving this problem is so simple, and it doesn't depend on any of the previous problems. In this part, you will need to write 5 lines of code at most.. Let's discuss some of the variables that you will need to use:

- `f` is the function we need to optimize.. it returns two values (`cost` and `gradient`).
- `expcost` or exponential cost which is simply the summation of the `cost` the we get from every iteration.
- `step` is our learning rate.
- `x` is the weights that will be updated. `x0` is the initial value for that weight.
- `postprocess` is the function that will be used after the weight update. **This function MUST be used!!!**.

Enjoy!!

---

## q4

### q4_sentiment

To the last part of the first assignment... This part is going to be a bit long, but that doesn't mean it's difficult. It is rather important. Before getting directly into the problem, let's first discuss some of the provided files that will be used in this problem. You can skip this part and deal with these files as a black box. But, I would recommend to follow this part to get some intuition.

#### Provided Code:

In the *utils* directory, there are a directory and three files:

- `datasets` directory... If you couldn't find it, then you should download the datasets by running the *get_datasets.sh* file.
- `__init__.py` file: it's empty. The `__init__.py` files are required to make Python treat the directories as containing packages
- `glove.py` file: it's used to load the `glove.6B.50d.txt` pre-trained word vectors.
- `treebank.py` file: it contains just one class (`StanfordSentiment`) that will will talk  about it a bit.

Now, let's get to the last two files in more details:

##### glove.py

This file is located in the *utils* directory and contains just one function which is `loadWordVectors`. This function takes a dictionary whose *keys* are words and *values* are the word indices. So, for example

```python
>>> import utils.glove as glove
>>>
>>> tokens = {'apple': 0, 'banana':1} #0 is the index of 'apple'
>>> wordVectors = glove.loadWordVectors(tokens)
>>> type(wordVectors)
<type 'numpy.ndarray'>
>>> wordVectors.shape
(2L, 50L)
>>> wordVectors
[[ 0.52042  -0.8314    0.49961   1.2893    0.1151    0.057521 -1.3753
  -0.97313   0.18346   0.47672  -0.15112   0.35532   0.25912  -0.77857
   0.52181   0.47695  -1.4251    0.858     0.59821  -1.0903    0.33574
  -0.60891   0.41742   0.21569  -0.07417  -0.5822   -0.4502    0.17253
   0.16448  -0.38413   2.3283   -0.66682  -0.58181   0.74389   0.095015
  -0.47865  -0.84591   0.38704   0.23693  -1.5523    0.64802  -0.16521
  -1.4719   -0.16224   0.79857   0.97391   0.40027  -0.21912  -0.30938
   0.26581 ]
 [-0.25522  -0.75249  -0.86655   1.1197    0.12887   1.0121   -0.57249
  -0.36224   0.44341  -0.12211   0.073524  0.21387   0.96744  -0.068611
   0.51452  -0.053425 -0.21966   0.23012   1.043    -0.77016  -0.16753
  -1.0952    0.24837   0.20019  -0.40866  -0.48037   0.10674   0.5316
   1.111    -0.19322   1.4768   -0.51783  -0.79569   1.7971   -0.33392
  -0.14545  -1.5454    0.0135    0.10684  -0.30722  -0.54572   0.38938
   0.24659  -0.85166   0.54966   0.82679  -0.68081  -0.77864  -0.028242
  -0.82872 ]]

```

As we can see, this function retrieves the word vector for "apple" and "banana". Each word vector is 50 dimensions.

##### treebankpy

This file has only one class (`StanfordSentiment`) which has around 17 methods... We need to know just four of them:

###### 1) tokens

This function returns a dictionary of words and their indexes. Each word has only one index associated with it.

```python
>>>from utils.treebank import StanfordSentiment
>>>
>>> dataset = StanfordSentiment()
>>> tokens = dataset.tokens()
>>> type(tokens)
<type 'dict'>
>>> #sort tokens based on their values (word index)
>>> sortedTokens = sorted(tokens.items(), key=lambda x: x[1])
>>> for k, v in sortedTokens:
...     print(k, v)
the 0
rock 1
is 2
destined 3
to 4
be 5
...
```

This means that the word "the" has an index of 0 no matter its position in the sentence, the word "rock" has an index of 1 and "is" has an index of 2... etc.

###### 2) getTrainSentences

This function returns a list of sentences from the `datasetSentences.txt` file located in the *datasets/stanfordSentimentTreebank* directory which contains around 11,855 sentences. Let's try it out:

```python
>>> from utils.treebank import StanfordSentiment
>>>
>>> dataset = StanfordSentiment()
>>> trainset = dataset.getTrainSentences()
>>> len(trainset)
8544
>>> type(trainset)
<type 'list'>
>>> trainset[:2]
[(['the', 'rock', 'is', 'destined', 'to', 'be', 'the', '21st', 'century', "'s", 'new', '``', 'conan', "''", 'and', 'that', 'he', "'s", 'going', 'to', 'make', 'a', 'splash', 'even', 'greater', 'than', 'arnold', 'schwarzenegger', ',', 'jean-claud', 'van', 'damme', 'or', 'steven', 'segal', '.'], 3), (['the', 'gorgeously', 'elaborate', 'continuation', 'of', '``', 'the', 'lord', 'of', 'the', 'rings', "''", 'trilogy', 'is', 'so', 'huge', 'that', 'a', 'column', 'of', 'words', 'can', 'not', 'adequately', 'describe', 'co-writer\\/director', 'peter', 'jackson', "'s", 'expanded', 'vision', 'of', 'j.r.r.', 'tolkien', "'s", 'middle-earth', '.'], 4)]
```

As we can see, each item in the `trainset` is a tuple where the first value is a list of words exist in the list, and the second value is label for this sentence. As said in the *assignment1.pdf* file is that "The sentiment level of the phrases are represented as real values in the original dataset, here we'll just use five classes:

- 0 very negative
- 1 negative
- 2 neutral
- 3 positive
- 4 very positive

These values are determined by the `sentiment_labels.txt` file which has a score for every sentence. The score is a value between 0 and 1, then these values are mapped to the previous discrete values [0, 1, 2, 3, 4] using `categorify` method.

Now to the more important question, why the number of sentences used in the training is 8544 and not the whole 11855 sentences??? The answer because the whole sentences is split into three sets (Train, Dev and Test). 

###### 2) getDevSentences

It does the same job as `getTrainSenteces`.

```python
>>> from utils.treebank import StanfordSentiment
>>>
>>> dataset = StanfordSentiment()
>>> devset = dataset.getDevSentences()
>>> len(devset)
1101
>>> type(trainset)
<type 'list'>
>>> devset[:2]
[(['it', "'s", 'a', 'lovely', 'film', 'with', 'lovely', 'performances', 'by', 'buy', 'and', 'accorsi', '.'], 3), (['no', 'one', 'goes', 'unindicted', 'here', ',', 'which', 'is', 'probably', 'for', 'the', 'best', '.'], 2)]
```



###### 3) getTestSentences

This function does the same job as `getTrainSentences` and `getDevSentences`.

```python
>>> from utils.treebank import StanfordSentiment
>>>
>>> dataset = StanfordSentiment()
>>> testset = dataset.getTestSentences()
>>> len(testset)
2210
>>> type(testset)
<type 'list'>
>>> testset[:2]
[(['effective', 'but', 'too-tepid', 'biopic'], 2), (['if', 'you', 'sometimes', 'like', 'to', 'go', 'to', 'the', 'movies', 'to', 'have', 'fun', ',', 'wasabi', 'is', 'a', 'good', 'place', 'to', 'start', '.'], 3)]
```

Now, we have know all the methods that we need to use in our task. 

#### Required methods:

In this part, we are going to discuss the methods that we need to implement to be able to do our Sentiment Analysis task. Let's get started...

##### 1) getSentenceFeatures

This function takes three arguments:

- `tokens` which is a dictionary whose keys are words and whose values are the indices of these words. It's the output of `StanfordSentiment().tokens()`.
- `wordVectors` which is the matrix representing the word vectors (word embedding). The row of this matrix represents the words and the column represents the features. It's the output of `glove.loadWordVectors(tokens)`
- `words` is the list of the words in the sentence.

We can consider this function as a representation of the sentence embedding. The simplest way to represent a sentence using word vectors is to -as stated in the lecture- average all the input word vectors.

##### 2) getRegularizationValues

Before talking about his function, let's talk first about the concept itself.. what is Regularization?? Deep Learning models have so much flexibility and capacity that overfitting can be a serious problem, if the training dataset is not big enough. Sure it does well on the training set, but the learned network doesn't generalize to new examples that it has never seen! Regularization is a way to solve this issue by increasing the cost function by a certain amount whenever the weights increases. The following image shows the difference between Cross-Entropy Cost function with and without regularization:

![cost with/without regularization](http://www.mediafire.com/convkey/3cf7/rl32lsss38itb33zg.jpg)

Regularization differs from an algorithm to another but the concept remains the same. All of these algorithms need a regularization parameter $\lambda$ and that's all we have to do here. We have to set some numbers to try as a regularization parameter. A good approach is to use Logarithmic Scale!!

##### 3) chooseBestModel

This function takes a list of dictionaries each dictionary has these keys:

- `reg`: which represents the value of regularization parameter.
- `clf`: which represents the classifier used (Logistic Regression)
- `train`: which represents the accuracy of the classifier over the training dataset.
- `dev`: which represents the accuracy of the classifier over the development dataset.
- `test`: which represents the accuracy of the classifier over the testing dataset.

Now, after implementing these functions, we can run the script by typing `python q4_sentiment.py --pretrained` to train the model using the `glove.6B.50d.txt` pre-trained GloVe vectors.

That is it.... Congrats!!!

# Last Words

While solving the `q4_sentiment.py` problem, it's essential to use good values for Regularization Parameter. It's a good practice to try different values on the logarithmic scale first, then try smaller values and smaller. So, the very first values I used were `[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]`. The performance plot was:

 ![reg_vs_acc](http://www.mediafire.com/convkey/e8c8/ecim4kwcj6acyzvzg.jpg)

As we can see, the dev accuracy were at its peak when $\lambda = 10$, but we can see that the train accuracy was declining.... this means that the range between [0.001 : 10] is a good range. Now, let's try a lot of values in this range and see the accuracy again:

![reg_vs_acc](http://www.mediafire.com/convkey/fe68/n7mn9fv7q91mi5ozg.jpg)