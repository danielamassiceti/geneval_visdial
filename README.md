# A Revised Generative Evaluation of Visual Dialogue

We propose a generative evaluation of the [VisDial](http://www.visualdialog.org) dataset which computes established NLP metrics (CIDEr, METEOR, FastText-L2, FastText-CS, BERT-L2, BERT-CS) between a generated answer and a _set_ of reference answers. We use a simple Canonical Correlation Analysis (CCA) [[Hotelling, 1926](https://academic.oup.com/biomet/article/28/3-4/321/220073); [Kettenring, 1971](https://www.jstor.org/stable/2334380?seq=1#metadata_info_tab_contents)] based approach to construct these answer reference sets at-scale across the whole dataset.

See our [paper](https://arxiv.org/abs/2004.09272) and our [previous analysis](http://arxiv.org/abs/1812.06417) (code [here](https://github.com/danielamassiceti/CCA-visualdialogue)) where we highlight the flaws of the existing rank-based evaluation of VisDial by applying CCA between question and answer embeddings and achieving near state-of-the-art results.

## Install package dependencies

All required packages are contained in `geneval_visdial.yml`. You can install them in an Anaconda environment as:

```bash
conda env create -f geneval_visdial.yml
source activate geneval_visdial
```
This repository uses PyTorch 1.0.1, Python 3.6 and CUDA 8.0. You can change these specifications in `geneval_visdial.yml`.

## How to evaluate your VisDial generations

### 1. Save your generations in the required format

Your generations will need to be saved as a `.json` file in the following format:
```python
[{"image_id": 185565, "round_id": 0, "generations": ["yes", "maybe", "yes, I think so"]}, 
{"image_id": 185565, "round_id": 1, "generations": ["several", "I can see two", "2, I think", "not sure"]}, 
{"image_id": 185565, "round_id": 2, "generations": ["yes", "yes, it is"]}, 
..., 
{"image_id": 185565, "round_id": 9, "generations": ["no", "no, can't see"]}, 
{"image_id": 284024, "round_id": 0, "generations": ["one"]}, 
... ]
```
`image_id` should correspond exactly to the VisDial v1.0 image IDs. `round_id` is 0-indexed and corresponds to the dialogue round (0 to 9). `generations` should contain a list of strings with **no** `<START>` or `<END>` tokens. Note, the number of generations can vary per image/round, but in our experiments we fix it to 1, 5, 10, or 15 generations per entry. In the case of 1 generation, each entry should still be a list (i.e. `"generations" : ["yes"]`).

See [`gens.json`](https://github.com/danielamassiceti/cca_visdial/blob/geneval_visdial/gens.json) as a example. These answers have been generated for the full VisDial _validation_ set using `CCA-AQ-G (k=1)` - see Table 4 (right) in paper. The code to generate these answers can be found in the [CCA-visualdialogue](https://github.com/danielamassiceti/CCA-visualdialogue) repository.

### 2. Download _DenseVisDial_ answer reference sets

Download [`refs_S_val.json`](https://www.robots.ox.ac.uk/~daniela/research/geneval_visdial/static/densevisdial/refs_S_val.json) and save it in `densevisdial` directory. These are the answer reference sets for the entire VisDial _validation_ set, automatically generated using the `S` or \Sigma clustering method. This method yields the best overlap with human-annotated reference sets, and we use it for all generative evaluation metrics reported in the paper.

Answer reference sets generated using other clustering methods (`S`, `M` and `G`) can be downloaded for the VisDial _train_ and _validation_ sets) here:
| C | Train | Val | Description |
| ------------- | ------------- | ------------- | ------------- |
| `S`  | [`refs_S_train.json`](https://www.robots.ox.ac.uk/~daniela/research/geneval_visdial/static/densevisdial/refs_S_train.json) | [`refs_S_val.json`](https://www.robots.ox.ac.uk/~daniela/research/geneval_visdial/static/densevisdial/refs_S_val.json)  | `\Sigma` clustering (based on standard deviation of correlations)
| `M`  | [`refs_M_train.json`](https://www.robots.ox.ac.uk/~daniela/research/geneval_visdial/static/densevisdial/refs_M_train.json) | [`refs_M_val.json`](https://www.robots.ox.ac.uk/~daniela/research/geneval_visdial/static/densevisdial/refs_M_val.json)  | Meanshift clustering
| `G`  | [`refs_G_train.json`](https://www.robots.ox.ac.uk/~daniela/research/geneval_visdial/static/densevisdial/refs_G_train.json) | [`refs_G_val.json`](https://www.robots.ox.ac.uk/~daniela/research/geneval_visdial/static/densevisdial/refs_G_val.json)  | Agglomerative clusting (n=5)

<!--You can use `get_clusters.py` to generate your own answer reference sets using one of the prescribed methods.-->

### 3. Download pre-trained BERT model and start `bert-as-a-service` server

The evaluation script uses the [`bert-as-a-service`](https://github.com/hanxiao/bert-as-service) client/server package. Download the [pre-trained BERT-Base, Uncased model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and save it in `<bert_model_dir>`.

Then start the `bert-as-a-service` server in a separate shell:
```bash
bert-serving-start  -model_dir <bert_model_dir>/uncased_L-12_H-768_A-12 -num_worker 2 \
                    -max_seq_len 16 -pooling_strategy CLS_TOKEN -pooling_layer -1
```
`num_workers` controls the number of GPUs or CPU cores (add `-cpu` flag) to use.

### 4. Download pre-trained FastText word vectors

The evaluation script uses pre-trained [FastText](https://fasttext.cc) word vectors. Download and unzip the English [`bin+text`](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip) FastText model pre-trained on Wikipedia. Save the `wiki.en.bin` file to `<fasttext_model_dir>`.

### 5. Run evaluation

The evaluation will compute CIDEr (n-grams 1 to 4), METEOR, BERT-L2 (L2 distance), BERT-CS (cosine similarity), FastText-L2 and FastText-CS between each generation and its corresponding set of reference answers.

```bash
python evaluate.py --generations gens.json --references densevisdial/refs_S_val.json> \
                   --fast_text_model <fasttext_model_dir>/wiki.en.bin
```

### TODO

* Provide DenseVisDial annotations for only human-annotated train and validation sets
* Add code for generating clusters 

## Citation

```
@article{massiceti2020revised,
  title={A Revised Generation Evaluation of Visual Dialogue},
  author={Massiceti, Daniela and Kulharia, Viveka, and Dokania, Puneet K and Siddharth, N and Torr, Philip HS},
  journal={arXiv preprint arXiv:2004.09272},
  year={2020}
}
```
