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

Download [`refs_S_full_val.json`](https://drive.google.com/uc?export=download&id=1HgmEDIUPZveFs4DfIsnLAosHjmWXWsmM) and save it in `densevisdial` directory. These are the answer reference sets for the entire VisDial _validation_ set, automatically generated using the `S` or \Sigma clustering method. This method yields the best overlap with human-annotated reference sets, and we use it for all generative evaluation metrics reported in the paper.

Answer reference sets generated using other clustering methods (`S`, `M` and `G`) and the human-annotated reference sets (`H`) can be downloaded for the VisDial _train_ and _validation_ sets) here:
| C | Train | Val | Description |
| ------------- | ------------- | ------------- | ------------- |
| `S`  | [`refs_S_full_train.json`](https://drive.google.com/uc?export=download&id=1RWz4x7-dUyFQDh5BqVz82jxnBUP7wJGA), [`refs_S_human_train.json`](https://drive.google.com/uc?export=download&id=15cJsQwLOOSVfCgLg_MzJoXXeneURG68_) | [`refs_S_full_val.json`](https://drive.google.com/uc?export=download&id=1HgmEDIUPZveFs4DfIsnLAosHjmWXWsmM), [`refs_S_human_val.json`](https://drive.google.com/uc?export=download&id=1t6utnkkoUPcljsecwSEwrKvMzMdadIDH)  | `\Sigma` clustering (based on standard deviation of correlations)
| `M`  | [`refs_M_full_train.json`](https://drive.google.com/uc?export=download&id=1KPkV4fGQIyWg0bS0_rqjhlw-eoJWMpez), [`refs_M_human_train.json`](https://drive.google.com/uc?export=download&id=1nqBWOklt-JiM5rJois0XoIue7P73u1yM) | [`refs_M_full_val.json`](https://drive.google.com/uc?export=download&id=1QGX24_8J-yHo6lltzDD4Dh-fKwtMb2Wv), [`refs_M_human_val.json`](https://drive.google.com/uc?export=download&id=1y7zGbcXriNZ_8qvAk-Ukr0jXPJUmtZMd) | Meanshift clustering
| `G`  | [`refs_G_full_train.json`](https://drive.google.com/uc?export=download&id=1S_UidG18z0-73mYPRsRfUbSBXH32o7lm), [`refs_G_human_train.json`](https://drive.google.com/uc?export=download&id=1hPdqwNn7TVYqziCwaNHNQgrngFysgLhO)| [`refs_G_full_val.json`](https://drive.google.com/uc?export=download&id=1ctTsJg4kayV4PluQx0Mu8EDqYGfXpIqD), [`refs_G_human_val.json`](https://drive.google.com/uc?export=download&id=1ujRluBr8qIYbZbMzzhOhXKwSrFu0pheC)  | Agglomerative clustering (n=5)
| `H`  | [`refs_H_human_train.json`](https://drive.google.com/uc?export=download&id=1d-LE3VNXTwTGcpd2ZDBq7GlfgntavW-k) | [`refs_H_human_val.json`](https://drive.google.com/uc?export=download&id=1ZDwmGj7dc4mc3e0UAD8sMH_nty5D92wa) | Human-annotated reference sets (relevance scores > 0)

See [How to generate answer reference sets](https://github.com/danielamassiceti/cca_visdial/blob/geneval_visdial/README.md#how-to-generate-answer-reference-sets) to generate your own answer reference sets using one of the prescribed methods.

### 3. Download pre-trained BERT model and start `bert-as-a-service` server

The evaluation script uses the [`bert-as-a-service`](https://github.com/hanxiao/bert-as-service) client/server package. Download the [pre-trained BERT-Base, Uncased model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and save it in `<bert_model_dir>`.

Then start the `bert-as-a-service` server in a separate shell:
```bash
bert-serving-start  -model_dir <bert_model_dir>/uncased_L-12_H-768_A-12 -num_worker 2 \
                    -max_seq_len 16 -pooling_strategy CLS_TOKEN -pooling_layer -1
```
`num_workers` controls the number of GPUs or CPU cores (add `-cpu` flag) to use.

### 4. Download pre-trained FastText word vectors

The evaluation script uses pre-trained [FastText](https://fasttext.cc) word vectors. Download and unzip the English [`bin+text`](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip) FastText model pre-trained on Wikipedia. Save the `wiki.en.bin` file as `<fasttext_model_dir>/fasttext.wiki.en.bin`.

### 5. Run evaluation

The evaluation will compute CIDEr (n-grams 1 to 4), METEOR, BERT-L2 (L2 distance), BERT-CS (cosine similarity), FastText-L2 and FastText-CS between each generation and its corresponding set of reference answers.

```bash
python evaluate.py --generations gens.json --references densevisdial/refs_S_val.json> \
                   --fast_text_model <fasttext_model_dir>/fasttext.wiki.en.bin
```

## How to generate answer reference sets

You can generate the answer reference sets yourself using clustering methods `S`, `M`, and `G`.

### 1. Prepare VisDial dataset

Download and unzip the dialogue `.json` files from:
* [`visdial_1.0_train.zip`](https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=1)
* [`visdial_1.0_val.zip`](https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=1)

Download the `.json` files with human-annotated scores:
* [`visdial_1.0_val_dense_annotations.json`](https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=1)
* [`visdial_1.0_train_dense_sample.json`](https://www.dropbox.com/s/1ajjfpepzyt3q4m/visdial_1.0_train_dense_sample.json?dl=1) (you will need to rename this `visdial_1.0_train_dense_annotations.json`)

Save all these `.jsons` to `<dataset_root>/1.0/`. 

Download and unzip the FastText English [`bin+text`](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip) model pre-trained on Wikipedia. Save the `wiki.en.bin` file as `<fasttext_model_dir>/fasttext.wiki.en.bin`.

The `compute_clusters.py` script will automatically load and pre-process the data. This may take 10-15 minutes. If you prefer, you can download the pre-processed features to `<dataset_root>/1.0/` directly:
* Train QAs: [`train_processed_S16_D10_woTrue_whsTrue.zip`](https://drive.google.com/uc?export=download&id=19qYK-i9ASSVmyN-n7gNs_7IEqqqUPjsP)
* Validation QAs: [`val_processed_S16_D10_woTrue_whsTrue.zip`](https://drive.google.com/uc?export=download&id=1NVoWCX691yH_bXmMkAGSHuOBNNT5Tn-f)
* Pre-processed vocabulary: [`vocab_visdial_v1.0_train.pt`](https://drive.google.com/uc?export=download&id=1h48U0KP2y72kYbueNPBUyCQXW0F3JV59)
* Pre-processed word vectors: [`fasttext.wiki.en.bin_vocab_vecs.pt`](https://drive.google.com/uc?export=download&id=1vMAPwk7EpjUaBwx3i_2xhW0wcaP5jCO4)

### 2. Generate clusters

You can now generate clusters using:
```bash
source activate geneval_visdial
python clusters/compute_clusters.py --dataset_root <dataset_root> \
                                    --fast_text_model <fasttext_model_dir>/fasttext.wiki.en.bin
                                    --gpu 1 \
                                    --cca QA_human_trainval \
                                    --eval_set full \
                                    --cluster_method S
```

This will compute clusters on the _full_ VisDial dataset (both train and validation sets) using the `S` clustering method and save the clusters in `./results` as `refs_S_full_train.json` and `refs_S_full_val.json`. If you want to compute clusters for only the subset of VisDial with human-annotated reference sets, use `--eval_set human`.

The `--cca` flag specifies the data to train the CCA model:
* `QA_human_train` trains on all answers with human-annotated relevance scores > 0 and their corresponding questions in the VisDial _train_ set.
* `QA_human_trainval` trains on all answers with human-annotated relevance scores > 0 and their corresponding questions in the VisDial _trainval_ set.
* `QA_full_train` trains on all ground-truth answers and their corresponding questions in the VisDial _train_ set.
* `QA_full_trainval` trains on all ground-truth answers and their corresponding questions in the VisDial _trainval_ set.

We use these differently depending on the evaluation set.

For `--eval_set human`:
* Table 4 (left), Table 6: we use `--QA_human_train --cluster_method H` to compute the human-annotated reference sets. We report overlap and embedding metrics for generated answers and these sets on the _validation_ subset, `\mathcal{H}_v`
* Table 1, Table 8 (`(A_gt, \tilde{A})` rows): we use `--QA_human_train` (CCA-QA*) and `--QA_full_train` (CCA-QA) to compute the overlap of the automatic reference sets `--cluster_method {M,S,G}` with the human-annotated reference sets (`H`).

For `--eval_set full`:
* Table 4 (right) and Table 7: we use `--QA_human_trainval --cluster_method S` to compute the automatic reference sets. We report overlap and embedding metrics for generated answers and these sets on the full _validation_ set.

## Citation

```
@article{massiceti2020revised,
  title={A Revised Generation Evaluation of Visual Dialogue},
  author={Massiceti, Daniela and Kulharia, Viveka, and Dokania, Puneet K and Siddharth, N and Torr, Philip HS},
  journal={arXiv preprint arXiv:2004.09272},
  year={2020}
}
```
