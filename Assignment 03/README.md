# Assignment #3

## Requirements

To be able to solve this assignment, we need to install a few packages which can be found in the `requirements.txt` file. The dataset is downloaded already and ready to be used. 

---

## q1

According to the first question, we are there is only one script that we need to complete implementing which is `q1_window.py` at which we are going to implement two things actually:

- The function `make_windowed_data` which has a really great description inside the script that is totally description enough.
- Most of the methods of the `WindowModel` class. This class can be implemented easily. If you got stuck at any part, just check the [parser model](https://github.com/Anwarvic/Stanford_CS224n--NLP-with-Deep-Learning/blob/master/Assignment%2002/assignment2/q2_parser_model.py) we have made in Assignment #2 putting in mind that there are minor changes like using `tf.nn.sparse_softmax_cross_entropy_with_logits` instead of `tf.nn.softmax_cross_entropy_with_logits_v2`. And, of course, the shapes of the used parameters are different as well.

One of the things that helped me understand this `WindowModel` is the following graph:

![WindowModel](http://www.mediafire.com/convkey/3627/8oj7mhrkiyfjxx1zg.jpg)

As we can see, we have a two-layer neural network where the input contains the features for our window. The input shape is `D` where `D` equal the number of features for every word in our window. So, assuming that the window size is just one (one on the left and one on the right), then `D` will equal to (*3* * *number of word feature*s * *size of word embedding*). The *number of word features* can be found in the member variable `self.config.n_word_features`, the window size can be found in `self.config.window_size` and the width of word embedding can be found in `self.config.embed_size`.

The number of hidden neurons in our hidden layer `H` is equal to the hidden size determined by the member variable `self.hidden_size`. Finally, the number of neurons in the output layer `C` can be found in the member variable `self.n_classes`.



Here, there are the outputs (for reference) that I got when running `q1_window.py`  with:

- `test1`

  ```powershell
  $ python q1_window.py test1
  ```

- `test2`

  ```powershell
  $ python q1_window.py test2
  ...
  ...
  label   acc     prec    rec     f1
  PER     0.96    0.73    0.88    0.80
  ORG     0.96    0.65    0.23    0.34
  LOC     0.98    0.84    0.82    0.83
  MISC    0.98    0.81    0.42    0.55
  O       0.96    0.96    0.99    0.97
  micro   0.97    0.92    0.92    0.92
  macro   0.97    0.80    0.67    0.70
  not-O   0.97    0.76    0.68    0.72

  INFO:Entity level P/R/F1: 0.67/0.63/0.65

  INFO:Model did not crash!
  INFO:Passed!
  ```

- `train`

  ```powershell
  $ python q1_window.py train
  ...
  ...
  DEBUG:Token-level confusion matrix:
  go\gu           PER             ORG             LOC             MISC            O
  PER             2940.00         51.00           79.00           18.00           61.00
  ORG             126.00          1637.00         118.00          71.00           140.00
  LOC             36.00           168.00          1806.00         39.00           45.00
  MISC            41.00           58.00           42.00           1016.00         111.00
  O               43.00           51.00           20.00           34.00           42611.00

  DEBUG:Token-level scores:
  label   acc     prec    rec     f1
  PER     0.99    0.92    0.93    0.93
  ORG     0.98    0.83    0.78    0.81
  LOC     0.99    0.87    0.86    0.87
  MISC    0.99    0.86    0.80    0.83
  O       0.99    0.99    1.00    0.99
  micro   0.99    0.97    0.97    0.97
  macro   0.99    0.90    0.88    0.89
  not-O   0.99    0.88    0.86    0.87

  INFO:Entity level P/R/F1: 0.80/0.83/0.82
  INFO:New best score! Saving model in results/window/20180816_170849/model.weights
  ```

- `evaluate`

  ```powershell
  $ python .\q1_window.py evaluate -m 'results\window\20180817_102055\' -o 'results\window\20180817_102055\results.txt'
  INFO:Initialized embeddings.
  INFO:Building model...
  INFO:took 6.81 seconds
  INFO:tensorflow:Restoring parameters from results\window\20180817_102055\model.weights
  INFO:Restoring parameters from results\window\20180817_102055\model.weights
  ```

- `shell`

  ```powershell
  $ python .\q1_window.py shell
  INFO:Initialized embeddings.
  INFO:Building model...
  INFO:took 7.62 seconds
  INFO:tensorflow:Restoring parameters from results\window\20180816_170849\model.weights
  INFO:Restoring parameters from results\window\20180816_170849\model.weights
  Welcome!
   can use this shell to explore the behavior of your model.
   Please enter sentences with spaces between tokens, e.g.,
   input> Germany's representative to the European Union's veterinary committee.

  input> x : I visited Turkey with Zeynep last summer
  y*:
  y': O O       LOC    O    MISC   O    O
  input>
  ```

  ​

---

## q2

According to the second question, there are two files that need to be done:

- `q2_rnn_cell.py`
- `q2_rnn.py`




### q2_rnn_cell

This is the first problem in q2, and it's pretty straight-forward, you will have to write about five lines of code. This graph will be very helpful with solving this problem:

![RNN Cell](http://www.mediafire.com/convkey/052b/79gso5gfmd32adyzg.jpg)

Putting in mind that:

- `x_t` is the `inputs`, whose shape is `[None, input_size]` and `None` can be any number.
- `h_{t-1}` is the `state`, whose shape is `[None, state_size]` and `None` is the same number as `inputs`. 
- `y_t` and `h_t` are the same.



### q2_rnn

The script that we have managed to solve in the first question `q1_window.py` will be a huge help to solve this problem putting in mind some major differences like:

- `add_operation_op` at which you have to iterate over time_steps and then adjust the `pred` shape.
- `add_loss_op` at which you have to make the `preds` and `labels` before calculating the loss.

> **HEADS-UP:**
> There is a function called `consolidate_predictions()`; I have commented the assertion line cuz it doesn't make sense to me at all:
>
> ```python
> >>> assert len(labels_) == len(labels)
> ```
>
> As we can see, the line before it is used to filter the predictions where the associated mask is `False`.

After running this script with `python q2_rnn.py test2`, you should get an output similar to this:

```powershell
INFO:Epoch 10 out of 10
23/23 [==============================] - 4s - train loss: 0.1927

INFO:Evaluating on development data
23/23 [==============================] - 19s
DEBUG:Token-level confusion matrix:
go\gu   PER     ORG     LOC     MISC    O
PER     786.00  3.00    6.00    4.00    25.00
ORG     88.00   178.00  23.00   26.00   76.00
LOC     18.00   14.00   513.00  9.00    23.00
MISC    24.00   21.00   14.00   170.00  29.00
O       29.00   13.00   4.00    3.00    7067.00

DEBUG:Token-level scores:
label   acc     prec    rec     f1
PER     0.98    0.83    0.95    0.89
ORG     0.97    0.78    0.46    0.57
LOC     0.99    0.92    0.89    0.90
MISC    0.99    0.80    0.66    0.72
O       0.98    0.98    0.99    0.99
micro   0.98    0.95    0.95    0.95
macro   0.98    0.86    0.79    0.81
not-O   0.98    0.85    0.80    0.82

INFO:Entity level P/R/F1: 0.74/0.76/0.75

INFO:Model did not crash!
INFO:Passed!
```

After running this script with `python q2_rnn.py train`, you should get an output similar to this:

```powershell
INFO:Epoch 10 out of 10
439/439 [==============================] - 407s - train loss: 0.0361

INFO:Evaluating on development data
102/102 [==============================] - 220s
DEBUG:Token-level confusion matrix:
go\gu           PER             ORG             LOC             MISC            O
PER             2977.00         30.00           69.00           9.00            64.00
ORG             139.00          1640.00         117.00          49.00           147.00
LOC             45.00           93.00           1901.00         15.00           40.00
MISC            36.00           49.00           51.00           1012.00         120.00
O               39.00           49.00           20.00           19.00           42632.00

DEBUG:Token-level scores:
label   acc     prec    rec     f1
PER     0.99    0.92    0.95    0.93
ORG     0.99    0.88    0.78    0.83
LOC     0.99    0.88    0.91    0.89
MISC    0.99    0.92    0.80    0.85
O       0.99    0.99    1.00    0.99
micro   0.99    0.98    0.98    0.98
macro   0.99    0.92    0.89    0.90
not-O   0.99    0.90    0.88    0.89

INFO:Entity level P/R/F1: 0.83/0.85/0.84

102/102 [==============================] - 248s
```

Running this script with the `train` tag takes around 2 hours!!

Now, let's run `python q2_rnn.py evaluate`:

```powershell
$ python .\q2_rnn.py evaluate -m 'results\rnn\20180819_230024\' -o 'results\rnn\20180819_230024\results.txt'
INFO:Initialized embeddings.
INFO:Building model...
C:\Users\anwar\Anaconda3\lib\site-packages\tensorflow\python\ops\gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
INFO:took 25.87 seconds
INFO:tensorflow:Restoring parameters from results\rnn\20180819_230024\model.weights
INFO:Restoring parameters from results\rnn\20180819_230024\model.weights
102/102 [==============================] - 240s
```

---

## q3



