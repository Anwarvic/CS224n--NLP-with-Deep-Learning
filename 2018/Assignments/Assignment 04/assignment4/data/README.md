# Data

Here, in this file, we are going to do basically two things:

- To explain briefly the downloaded data located in `download` directory.
- To explain in details the preprocessed data located in `data` directory.

## Downloaded Data

As you can see, we have downloaded two types of data:

- SQuAD dataset
- Distributed Word Representations

### SQuAD Dataset

After downloading the SQuAD dataset would be placed in the `data/squad folder`. SQuAD downloaded
files include train and dev files in JSON format:

- *train-v1.1.json*: a train dataset with around 87k triplets.
- *dev-v1.1.json*: a dev dataset with around 10k triplets.

Each data point inside these two files would look like this:

``` json
{
	'data': [{
		'title': 'Super_Bowl_50'
		'paragraphs':[{
			'context': 'Super Bowl 50 was an American football game ....',
			'qas': [{
				'answers': [{
					'anwer_start': 117,
					'text': 'Denver Broncos'}},
					...
				],
				'question': 'Which NFL team represented the AFC at Super Bowl 50?',
				'id': 56be4db0acb8001400a502ec},
				...
				]},
			...
			]},
		...
		]
	}],
	'version': 1.1
}
```

### Distributed Word Representations

The Distributed Word Representations that we are going to use here is GloVe word embeddings. Here, we are going to download four files `glove.6B.xxxd.txt` of the name of the file matches the dimensionality d = `50`, `100`, `200`, and `300`. All these files have been pre-trained on Wikipedia 2014 and Gigaword. The word vectors are stored in the `download/dwr` subfolder.

The file `qa_data.py` will trim the GloVe embedding with the given dimension (by default `d = 100`) into a much smaller file. Your model only needs to load in that trimmed file. Feel free to remove the `download/dwr` subfolder after preprocessing is finished.

Each line of these files looks like this:

```json
the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 -0.00066023 -0.6566 0.27843 -0.14767 -0.55677 0.14658 -0.0095095 0.011658 0.10204 -0.12792 -0.8443 -0.12181 -0.016801 -0.33279 -0.1552 -0.23131 -0.19181 -1.8823 -0.76746 0.099051 -0.42125 -0.19526 4.0071 -0.18594 -0.52287 -0.31681 0.00059213 0.0074449 0.17778 -0.15897 0.012041 -0.054223 -0.29871 -0.15749 -0.34758 -0.045637 -0.44251 0.18785 0.0027849 -0.18411 -0.11514 -0.78581
```

As we can see, the first token is the word. In this case, we have `the`. And beside that we have numbers, the count of these numbers varies among different files. The above vector was taken from `glove.6B.50d.txt`, that's why we have 50 numbers.

The embeddings used, by default, have dimensionality `d = 100` and have been trained on 6B word corpora (Wikipedia + Gi-gaword). The vocabulary is uncased (all lowercase). Analyze the effect of selecting different
embeddings for this task, e.g., other families of algorithms, larger size, trained on different
corpora, et cetera.



## Processed Data

As you can see, we have downloaded two types of data:

- SQuAD dataset





