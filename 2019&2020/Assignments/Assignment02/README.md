# Assignment 2 (Word2Vec)

In this assignment you will implement the word2vec model and train your own word vectors with stochastic gradient
descent (SGD) from scratch using NumPy.

Before you begin, first run the following commands within the assignment directory in order to create the appropriate `conda` virtual environment. This guarantees that you have all the necessary
packages to complete the assignment.

```
conda env create -f env.yml
conda activate a2
```
Once you are done with the assignment you can deactivate this environment by running:
```
conda deactivate
```

## Steps

### Part A
- First, implement the sigmoid function in `word2vec.py` to apply the sigmoid function to an input vector.

- In the same file, fill in the implementation for the softmax and negative sampling loss and gradient functions.

- Then, fill in the implementation of the loss and gradient functions for the skip-gram model.

- When you are done, test your implementation by running python `word2vec.py`.

### Part B

Complete the implementation for your SGD optimizer in `sgd.py`. Test your implementation by running python `sgd.py`.

### Part C
Now we are going to load some real data and train word vectors with everything you just implemented! We are going to use the Stanford Sentiment Treebank (SST) dataset to train word vectors, and later apply them to a simple sentiment analysis task.

You will need to fetch the datasets first. To do this, run the following command which will download the data into `a2/utils/datasets` directory.

```bash
cd a2
bash ./get_datasets.sh
```
There is no additional code to write for this part; just run:
```
python run.py
```
**Note:**

The training process may take a long time depending on the efficiency of your implementation (an efficient implementation takes approximately an hour). Plan accordingly! After 40,000 iterations, the script will finish and a visualization for your word vectors will appear. It will also be saved as word `vectors.png` in your project directory. Include the plot in your homework write up. Briefly explain in at most three sentences what you see in the plot.