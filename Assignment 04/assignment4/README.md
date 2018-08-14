# Programming Assignment 4
Welcome to CS224N Project Assignment 4 Reading Comprehension.
The project has several dependencies that have to be satisfied before running the code. You can install them using your preferred method -- we list here the names of the packages using `pip`.

# Requirements

Essentially, this assignment was made for python2.7 and TensorFlow 0.12. But now, most people use Python3 and recent versions of Tensorflow. So, here, I have re-wrote the code to be compatible with Python3 and TensorFlow 1.9.0. You can download my re-written code for Python3 from [here]().

Second, you also should install all needed dependencies through
`pip install -r requirements.txt`.

Finally, running `get_started.sh` is no good for me as this file downloads a huge amount of data which is no good for me as my bandwidth rate is pretty low. So, I had to download everything one at a time. You follow the same steps I took by following these steps (**CAUTION**: All the following steps assumes that you are in the `code` directory):

1. install NLTK and its "popular" collection. You can do that by opening the terminal and typing the following:

   ```python
   $ python
   >>> import nltk
   >>> nltk.download('popular')
   ```

2. Now, all we need is to run the `squad_preprocess.py` file which is in charge of downloading and formatting the SQUAD data to be consumed later; assuming you are in the `code` directory:

   ```python
   $ python preprocessing/squad_preprocess.py
   Downloading datasets into download\squad
   Preprocessing datasets into data\squad
   File train-v1.1.json successfully loaded
   Preprocessing train: 100%|███████████████████████████████████████████████████████████| 442/442 [02:05<00:00,  3.52it/s]
   Skipped 87599 question/answer pairs in train
   Splitting the dataset into train and validation
   Shuffling...
   Processed 87599 questions and 87599 answers in train
   Downloading dev-v1.1.json
   Downloading file https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json...
   File dev-v1.1.json successfully loaded
   ```
   After finishing running this file, you should find these files in `data/squad`:

   ```python
   train.answer   (~2  MB)
   train.context  (~66 MB)
   train.question (~5  MB)
   train.span     (~1  MB)

   val.answer     (~100KB)
   val.context    (~4  MB)
   val.question   (~0.5MB)
   val.span       (~30 KB)

   ```

   Now, we have to do the same with `dwr.py` which is responsible for Downloading distributed word representations (GloVe); assuming you're still in the `code` directory:

   ```python
   $ python preprocessing/dwr.py
   Storing datasets in download\dwr
   Downloading file http://nlp.stanford.edu/data/glove.6B.zip...
   glove.6B.zip:   100%|███████████████████████████████████████████████████████████| 57.3k/862M [02:24<799:17:04, 300B/s]
   File glove.6B.zip successfully loaded
   ```
   After finishing running this code which takes around 5 minutes, you should find the extracted data in `download\dwr` directory. And they are:

   ```python
   glove.6B.50d.txt      (~167 MB)
   glove.6B.100d.txt     (~340 MB)
   glove.6B.200d.txt     (~677 MB)
   glove.6B.300d.txt     (~1   GB)
   ```

3. Now, we have to run `qa_data.py` which is responsible for Data processing for TensorFlow; assuming you are still in the `code` directory:

   ```python
   $ python qa_data.py
   100%|██████████████████████████████████████████████████████████████████████| 400000/400000.0 [1:31:50<00:00, 72.59it/s]
   93844/115746 of word vocab have corresponding vectors in preprocessing\download\dwr\glove.6B.100d.txt
   saved trimmed glove matrix at: preprocessing\data\squad\glove.trimmed.100
   Tokenizing data in preprocessing\data\squad\train.context
   tokenizing line 5000
   tokenizing line 10000
   tokenizing line 15000
   tokenizing line 20000
   tokenizing line 25000
   tokenizing line 30000
   tokenizing line 35000
   tokenizing line 40000
   tokenizing line 45000
   tokenizing line 50000
   tokenizing line 55000
   tokenizing line 60000
   tokenizing line 65000
   tokenizing line 70000
   tokenizing line 75000
   tokenizing line 80000
   Tokenizing data in preprocessing\data\squad\train.question
   tokenizing line 5000
   tokenizing line 10000
   tokenizing line 15000
   tokenizing line 20000
   tokenizing line 25000
   tokenizing line 30000
   tokenizing line 35000
   tokenizing line 40000
   tokenizing line 45000
   tokenizing line 50000
   tokenizing line 55000
   tokenizing line 60000
   tokenizing line 65000
   tokenizing line 70000
   tokenizing line 75000
   tokenizing line 80000
   Tokenizing data in preprocessing\data\squad\val.context
   Tokenizing data in preprocessing\data\squad\val.question
   ```
   After finishing running this code which takes around 1 hour and 35 minutes on my humble laptop, you should find two files have been added to the `data\squad` directory. And they are:

   ```python
   glove.trimmed.100.npz     (~65 MB)
   vocab.dat                 (~2 MB)
   ```

4. Now, after doing all of that, we can delete `download` directory.



Now, we are ready to start our assignment!

Enjoy!!




