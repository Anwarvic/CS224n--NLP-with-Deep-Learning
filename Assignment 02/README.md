# Assignment #2

## Requirements

In this assignment I have used TensorFlow 1.19.0. In the past, TensorFlow was available for both Python2.7 and Python3. Now, it's only available in Python3. This code was written in Python2. So, I had to re-write it for Python3 and you can download my re-written code from [here](http://www.mediafire.com/file/hvg82wc1t4xaogl/assignment2%28python3%29.rar/file).

Another thing to point out... in the file `parser_utils.py`, I changed the importing line from:

```python
>>> from general_utils import logged_loop, get_minibatches
```

To:

```python
>>> from utils.general_utils import logged_loop, get_minibatches
```

---

## Heads-up

I really found the assignment description very helpful. I think that's all you need to be able to solve this assignment. The following is some pieces of advice that could be helpful (hopefully):

- This assignment doesn't need a full comprehension of TensorFlow, you need just to know the basics. I found the lecture 7 video, which was an introduction for TensorFlow, not that helpful especially I was a beginner. But I find the Stanford Course "[TensorFlow for Deep Learning Research](http://web.stanford.edu/class/cs20si/)" is pretty helpful. You don't have to take the whole course, just the first three lecture notes will be more than enough. BTW, it's the best source for learning TensorFlow, I highly recommend it.

- Note that, you will always want to run your code from this assignment directory, not any other directory, like so:

  `$ python q1_classifier.py`

  This ensures that any files created in the process don't pollute the code directory.



## Output (for reference)

After running the `q2_parser_model.py` in debug mode (the default), we get an output of:

```
================================================================================
INITIALIZING
================================================================================
Loading data...
took 5.20 seconds
Building parser...
took 0.07 seconds
Loading pretrained embeddings... took 6.30 seconds
Vectorizing data...
took 0.12 seconds
Preprocessing training data...
1000/1000 [==============================] - 3s
Building model...
took 0.75 seconds

================================================================================
TRAINING
================================================================================
Epoch 1 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.6767
Evaluating on dev set
- dev UAS: 47.13

Epoch 2 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.3592
Evaluating on dev set
- dev UAS: 57.63

Epoch 3 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.2869
Evaluating on dev set
- dev UAS: 61.07

Epoch 4 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.2483
Evaluating on dev set
- dev UAS: 62.83

Epoch 5 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.2266
Evaluating on dev set
- dev UAS: 64.06

Epoch 6 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.2059
Evaluating on dev set
- dev UAS: 65.86

Epoch 7 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.1871
Evaluating on dev set
- dev UAS: 66.71

Epoch 8 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.1732
Evaluating on dev set
- dev UAS: 67.58

Epoch 9 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.1609
Evaluating on dev set
- dev UAS: 68.62

Epoch 10 out of 10
24/24 [============================>.] - ETA: 0s - train loss: 0.1497
Evaluating on dev set
- dev UAS: 69.40
```

When running with `debug=False`, you should be able to get a loss smaller than 0.07 on the train
set (by the end of the last epoch) and an Unlabeled Attachment Score larger than 88 on the dev set
(with the best-performing model out of all the epochs). I got 85.77 on the Dev set, and 86.10 on the test set. 

```
================================================================================
INITIALIZING
================================================================================
Loading data...
took 6.53 seconds
Building parser...
took 3.49 seconds
Loading pretrained embeddings... took 8.35 seconds
Vectorizing data...
took 4.64 seconds
Preprocessing training data...
39832/39832 [==============================] - 147s
Building model...
took 3.72 seconds

================================================================================
TRAINING
================================================================================
Epoch 1 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.1901
Evaluating on dev set
- dev UAS: 80.80
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 2 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.1171
Evaluating on dev set
- dev UAS: 82.75
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 3 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.1030
Evaluating on dev set
- dev UAS: 84.70
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 4 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.0943
Evaluating on dev set
- dev UAS: 84.81
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 5 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.0881
Evaluating on dev set
- dev UAS: 85.17
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 6 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.0831
Evaluating on dev set
- dev UAS: 85.25
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 7 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.0788
Evaluating on dev set
- dev UAS: 85.45
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 8 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.0749
Evaluating on dev set
- dev UAS: 85.66
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 9 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.0714
Evaluating on dev set
- dev UAS: 86.04
New best dev UAS! Saving model in ./data/weights/parser.weights

Epoch 10 out of 10
924/924 [============================>.] - ETA: 0s - train loss: 0.0683
Evaluating on dev set
- dev UAS: 85.77

================================================================================
TESTING
================================================================================
Restoring the best model weights found on the dev set
Final evaluation on test set
- test UAS: 86.10
Restoring the best model weights found on the dev set
Final evaluation on test set
- test UAS: 86.33
Writing predictions
Done!
```