# Assignment #3

## Requirements

To be able to solve this assignment, we need to install a few packages which can be found in the `requirements.txt` file. The dataset is downloaded already and ready to be used.  This assignment was written for python2.7 and Tensorflow 0.12. Tensorflow now is supported by python3 only. So, I have re-written the starting code in python3. You can downloaded it from [here](http://www.mediafire.com/file/yvh554zv5n2dbd3/starter_code.rar/file)

---

## q1

According to the first question, there is only one script that we need to complete which is `q1_window.py` at which we are going to implement two things actually:

- The function `make_windowed_data` which has a really great description inside the script. So, I think the description is enough.
- Most of the methods of the `WindowModel` class can be implemented easily. If you got stuck at any part, just check the [parser model](https://github.com/Anwarvic/Stanford_CS224n--NLP-with-Deep-Learning/blob/master/Assignment%2002/assignment2/q2_parser_model.py) we have made in Assignment #2 putting in mind that there are minor changes like using `tf.nn.sparse_softmax_cross_entropy_with_logits` instead of `tf.nn.softmax_cross_entropy_with_logits_v2`. And, of course, the shapes of the used parameters are different as well.

One of the things that helped me understand this `WindowModel` is the following graph:

![WindowModel](http://www.mediafire.com/convkey/3627/8oj7mhrkiyfjxx1zg.jpg)

As we can see, we have a two-layer neural network where the input contains the features for our window. Let's talk about the different neurons used in this vanilla neural network:

- The number of neurons in the input layer is `D` where `D` equals the number of features for every word in our window. So, assuming that the window size is just one (one on the left and one on the right), then `D` will equal to (*3* * *number of word feature*s * *size of word embedding*). The *number of word features* can be found in the member variable `self.config.n_word_features`, the window size can be found in `self.config.window_size` and the width of word embedding can be found in `self.config.embed_size`.
- The number of neurons in the hidden layer is `H` which equals to the hidden size determined by the member variable `self.hidden_size`
- Finally, the number of neurons in the output layer is `c` which equals to the number of classes we are trying to classify. In our model, we are using five different classes (`PER`, `LOC`, `ORG`, `MISC`, and `O`). This value can be found in the member variable `self.config.n_classes`

After knowing the number of neurons in each layer in our neural network, determining the variables should be pretty straight-forward.

### Output

There are the outputs (for reference) that I got after running `q1_window.py` with:

- `test1`

  ```powershell
  $ python q1_window.py test1
  INFO:Testing make_windowed_data
  INFO:Passed!
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
  $ python .\q1_window.py shell -m 'results\window\20180817_102055\'
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


The `None` that we used in defining our placeholders should equal to the batch size that we are using.

If you found this figure not helpful, then forget about it and use the following equation:

![RNN_cell equation](http://www.mediafire.com/convkey/21cf/icj5mx2hatclt7gzg.jpg)


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
$ python q2_rnn.py test2
...
...
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
$ python q2_rnn.py train
...
...
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

Running this script with the `train` flag takes around 2 hours!!

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

Now, let's run `python q2_rnn.py shell` command:

```powershell
$ python .\q2_rnn.py shell -m 'results\rnn\20180819_230024\'
INFO:Initialized embeddings.
INFO:Building model...
C:\Users\anwar\Anaconda3\lib\site-packages\tensorflow\python\ops\gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
INFO:took 30.13 seconds
INFO:tensorflow:Restoring parameters from results\rnn\20180819_230024\model.weights
INFO:Restoring parameters from results\rnn\20180819_230024\model.weights
Welcome!
 can use this shell to explore the behavior of your model.
 Please enter sentences with spaces between tokens, e.g.,
 input> Germany's representative to the European Union's veterinary committee.

input> I visited Turkey with Zeynep last summer
1/1 [==============================] - 2s
x : I visited Turkey with Zeynep last summer
y*:
y': O O       LOC    O    LOC    O    O
input>
```

---

## q3

This question is similar to q2, as it has also two problems that need to be done:

- `q3_gru_cell.py`
- `q3_gru.py`


### q3_gru_cell

This is the first problem in q3, and it's pretty straight-forward, you will have to write about 15 lines of code in which you will have to implement the following equations:

![gru_cell equations](http://www.mediafire.com/convkey/1b44/e6p25vrt222hottzg.jpg)

After doing all of that, let's run `python q3_gru_cell.py test`. You should get an output like that:

```powershell
$ python q3_gru_cell.py
INFO:Testing gru_cell
y_ = [[ 0.32035077  0.55478156]
 [-0.00592546  0.0195577 ]]
ht_ = [[ 0.32035077  0.55478156]
 [-0.00592546  0.0195577 ]]
INFO:Passed!
```

Now, we have implemented the `q3_gru_cell.py` correctly. Let's run the `q2_rnn.py` script using GRU cell. We can do that using the `-c gru` flag like so:

```powershell
$ python q2_rnn.py test2 -c gru
INFO:Epoch 10 out of 10
23/23 [==============================] - 10s - train loss: 0.1103

INFO:Evaluating on development data
23/23 [==============================] - 57s
DEBUG:Token-level confusion matrix:
go\gu   PER     ORG     LOC     MISC    O
PER     797.00  16.00   1.00    3.00    7.00
ORG     16.00   335.00  6.00    5.00    29.00
LOC     2.00    39.00   522.00  4.00    10.00
MISC    13.00   27.00   5.00    197.00  16.00
O       4.00    17.00   1.00    2.00    7092.00

DEBUG:Token-level scores:
label   acc     prec    rec     f1
PER     0.99    0.96    0.97    0.96
ORG     0.98    0.77    0.86    0.81
LOC     0.99    0.98    0.90    0.94
MISC    0.99    0.93    0.76    0.84
O       0.99    0.99    1.00    0.99
micro   0.99    0.98    0.98    0.98
macro   0.99    0.93    0.90    0.91
not-O   0.99    0.92    0.90    0.91

INFO:Entity level P/R/F1: 0.84/0.88/0.86

INFO:Model did not crash!
INFO:Passed!
```

As we can see, the model has much better performance using GRU rather than using basic RNN cells. 

Now, let's train our RNN model using GRU cells:

```powershell
$ python q2_rnn.py train -c gru
...
...
INFO:Epoch 9 out of 10
439/439 [==============================] - 1257s - train loss: 0.0301

INFO:Evaluating on development data
102/102 [==============================] - 500s
DEBUG:Token-level confusion matrix:
go\gu           PER             ORG             LOC             MISC            O
PER             2905.00         49.00           96.00           10.00           89.00
ORG             109.00          1701.00         86.00           59.00           137.00
LOC             28.00           86.00           1920.00         21.00           39.00
MISC            43.00           37.00           47.00           1034.00         107.00
O               37.00           53.00           21.00           25.00           42623.00

DEBUG:Token-level scores:
label   acc     prec    rec     f1
PER     0.99    0.93    0.92    0.93
ORG     0.99    0.88    0.81    0.85
LOC     0.99    0.88    0.92    0.90
MISC    0.99    0.90    0.82    0.86
O       0.99    0.99    1.00    0.99
micro   0.99    0.98    0.98    0.98
macro   0.99    0.92    0.89    0.90
not-O   0.99    0.90    0.88    0.89

INFO:Entity level P/R/F1: 0.84/0.85/0.84
INFO:New best score! Saving model in results/gru/20180824_032828/model.weights
```

Running the previous script took around three hours on my humble laptop. As you can see, it was just 9 epochs because my humble laptop crashes before reaching to the tenth epoch. 

Now, let's evaluate our dev set:

```powershell
$ python .\q2_rnn.py evaluate -c gru -m 'results\gru\20180824_032828\' -o 'results\gru\20180824_032828\results.txt'
INFO:Initialized embeddings.
INFO:Building model...
C:\Users\anwar\Anaconda3\lib\site-packages\tensorflow\python\ops\gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
INFO:took 65.42 seconds
INFO:tensorflow:Restoring parameters from results\gru\20180824_032828\model.weights
INFO:Restoring parameters from results\gru\20180824_032828\model.weights
102/102 [==============================] - 481s
```

### q3_gru

This script is used to generate a learning curve for this task for the RNN and GRU models. Here, we are going to generate six learning curves:

- Running RNN using standard RNN cells with/without clipping using prediction/dynamics mode.
- Running RNN using GRU cells with/without clipping using prediction/dynamics mode.
- Running RNN using LSTM cells with/without clipping using prediction/dynamics mode.

**NOTE:**

I have modified the `output_path` variable to generate the images inside a folder called `figures`.

#### Standard RNN cells

Now, let's our RNN model using standard RNN cells without clipping:

```powershell
$ python q3_gru.py predict
INFO:Building model...
INFO:took 20.21 seconds
INFO:Epoch 1 out of 40
100/100 [==============================] - 1s - train loss: 12.6087
...
...
INFO:Epoch 40 out of 40
100/100 [==============================] - 0s - train loss: 12.4978
```

And the generated learning curve will look like this:

![noclip-rnn](http://www.mediafire.com/convkey/ca84/lpfv611zvmaw4ipzg.jpg)

Now, let's run it using gradient clipping:

```powershell
$ python q3_gru.py predict -g
INFO:Building model...
INFO:took 8.55 seconds
INFO:Epoch 1 out of 40
100/100 [==============================] - 1s - train loss: 12.6087
...
...
INFO:Epoch 40 out of 40
100/100 [==============================] - 0s - train loss: 12.4978
```

Which would have the same learning curve as the gradient clipping is a way to solve *Gradient Explosion* which doesn't happen that often with standard RNN cells.

#### GRU cells

Now, let's our RNN model using standard RNN cells without clipping:

```powershell
$ python q3_gru.py predict -c gru
INFO:Building model...
INFO:took 8.06 seconds
INFO:Epoch 1 out of 40
100/100 [==============================] - 1s - train loss: 12.5133
...
...
INFO:Epoch 40 out of 40
100/100 [==============================] - 1s - train loss: 12.5020
```

And the generated learning curve will look like this:

![noclip-gru](http://www.mediafire.com/convkey/2702/9r882qx55mos2a6zg.jpg)

Now, let's run it using gradient clipping:

```powershell
$ python q3_gru.py predict -c gru -g
INFO:Building model...
INFO:took 8.55 seconds
INFO:Epoch 1 out of 40
100/100 [==============================] - 1s - train loss: 12.5133
...
...
INFO:Epoch 40 out of 40
100/100 [==============================] - 0s - train loss: 12.4978
```

Which would have the same learning curve.

#### LSTM cells

Now, let's our RNN model using standard RNN cells without clipping:

```powershell
$ python q3_gru.py predict -c lstm
INFO:Building model...
INFO:took 0.93 seconds
INFO:Epoch 1 out of 40
100/100 [==============================] - 2s - train loss: 25.0919
...
...
INFO:Epoch 40 out of 40
100/100 [==============================] - 1s - train loss: 24.9955
```

And the generated learning curve will look like this:

![noclip-lstm](http://www.mediafire.com/convkey/8156/gxvsutdxq4u8v93zg.jpg)

Now, let's run it using gradient clipping:

```powershell
$ python q3_gru.py predict -c lstm -g
INFO:Building model...
INFO:took 8.55 seconds
INFO:Epoch 1 out of 40
100/100 [==============================] - 1s - train loss: 12.6087
...
...
INFO:Epoch 40 out of 40
100/100 [==============================] - 1s - train loss: 24.9955
```

Which would have the same learning curve.



#### dynamics mode (OPTIONAL)

The two functions responsible for running this mode are (`make_dynamics_plot` and `compute_cell_dynamics`) and both actually had some bugs in them.  According to the first function which is `make_dynamics_plot` I had to comment the following two lines as they raises the following error `RuntimeError: LaTeX was not able to process the following string: b'lp'`. And these two lines are:

```python
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
```

According to the second function which is `compute_cell_dynamics`, it has a lot of bugs actually. First, it didn't use `RNNCell` anywhere in the function even though the whole function is just a comparison between RNN normal cells and GRU cells. Second, the `RNNCell` weights which are `W_x`, `W_h` and `b` weren't defined. 

So, I solved these bugs by defining the `RNNCell` weights using `Ur`, `Wr` and `br` as initializers like so:

```python
tf.get_variable("W_x", initializer=Ur)
tf.get_variable("W_h", initializer=Wr)
tf.get_variable("b", initializer=br)
```

Then I modified the following line from this:

```python
y_rnn, h_rnn = GRUCell(1,1)(x_placeholder, h_placeholder, scope="cell")
```

To this:

```python
y_rnn, h_rnn = RNNCell(1,1)(x_placeholder, h_placeholder, scope="cell")
```

Now, we can run this `q3_gru.py` script in `dynamics` mode to plot how an RNN or GRU map an input state to an output state. And it produced these two graphs... the first when the input is all zeros and the second when the input is all ones:

![all_zeros_input](http://www.mediafire.com/convkey/3570/6baenabwo6v40dezg.jpg)



![all_ones_input](http://www.mediafire.com/convkey/12a4/k2q6968jooj2rcfzg.jpg)