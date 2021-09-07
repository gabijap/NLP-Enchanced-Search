# NLP-Search: NLP enhanced search based on Quick Thought algorithm

This is PyTorch implementation of the Quick Thought (QT) algorithm from the paper Lajanugen Logeswaran, Honglak Lee, An
efficient framework for learning sentence representations, ICLR 2018.

## Environment Setup

This is a list of what needs to be installed:

* Python 3 (I used [Conda](https://www.anaconda.com/products/individual) distribution)
* [PyTorch](https://pytorch.org/) - for building and training machine learning model
* [GenSim](https://radimrehurek.com/gensim/) - for word embeddings, text tokenization and reading the Wikipedia dataset
* [NLTK](https://www.nltk.org/) - for sentence tokenization
* [NumPy](http://www.numpy.org/) - for handling arrays
* [SentEval](https://github.com/facebookresearch/SentEval) - for evaluating quality of the sentence embeddings
* [TensorBoard](https://www.tensorflow.org/tensorboard/) - for plotting charts while training in a real time
* [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/deb.html) - for text indexing and search

This is a list of data sets, that were used:

* [Wikipedia](https://dumps.wikimedia.org/enwiki/latest/) - 160 million sentences from English Wikipedia
* [arXiv](https://www.kaggle.com/Cornell-University/arxiv) - 1.7 million abstracts of scientific papers over the past 30 years 
* [BookCorpus](https://yknzhu.wixsite.com/mbweb) - collection of many English books


### Environment setup steps:

1. Install Ubuntu 20.04.
   
2. Install some aditional packages:
```bash
sudo apt-get update -y && apt-get install \
  apt-utils \
  tmux \
  wget \
  curl \
  unzip \
  parallel
```

3. Install conda and required Python libraries:
```bash
sudo curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && chmod +x ~/miniconda.sh \
  && ~/miniconda.sh -b -p /opt/conda \
  && /opt/conda/bin/conda config --append channels conda-forge \
  && /opt/conda/bin/conda install -y \
  numpy \
  jupyter \
  scikit-learn \
  matplotlib \
  gensim \
  nltk \
  flask \
  jsonpatch \
  pyyaml \
  scipy \
  ipython \
  mkl \
  mkl-include \
  ninja \
  cython \
  typing \
  pylatexenc \
  && /opt/conda/bin/conda install -y -c conda-forge pathos opencv tensorboard kaggle elasticsearch \
  && /opt/conda/bin/conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

4. Install Elasticsearch (These steps are from Elasticsearch documentation: https://www.elastic.co/guide/en/elasticsearch/reference/current/deb.html)
```bash
sudo apt-get install apt-transport-https \
  && wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add - \
  && echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-7.x.list \
  && apt-get update \
  && apt-get install elasticsearch
```

5. Prepare datasets:
```bash
cd data
./prepare_datasets.sh
cd ..
````

6. Load data into Elastic:
```bash
cd elastic
./parallel_upload.sh
cd ..
```

## Model Training 

Train the model:
```bash
python train.py
```

## Model Evaluation

1. Run model accuracy evaluation (run SentEval tests):
```bash
python sentEval.py
```

2. Run Web Search application and access http://192.168.0.1:8097
```bash
./run_websearch.sh
```

3. Alternatively, run search queries from a command line:
```bash
python doc_rerank_es.py 
```


## Modules Structure

This is a list of most important modules:

```bash
.
├── README.md                       
├── checkpoint                           -> Generated models
├── data                                 -> Directory containing datasets
│   ├── SentEval                         -> Directory containing SentEval datasets
│   ├── arxiv                            -> Directory containing arXiv dataset 
│   │   └── clean_arxiv.sh               -> Script to convert arXiv dataset to format required for loading into Elasticsearch
│   │   └── download_arxiv_dataset.sh    -> Script to download arXiv dataset
│   ├── prepare_datasets.sh              -> Script to download and prepare the datasets
│   └── wikipedia                        -> Scripts to download and prepare Wikipedia dataset
│       ├── clean_data.py                -> Script to clean
│       ├── download_wikipedia.sh        -> Script to download Wikipedia dataset
│       ├── one_sentence_per_line.py     -> Script to convert Wikipedia dataset from JSON to text file 
│       │                                   one sentence per line
│       ├── parallel_clean.sh            -> Script to clean out of vocabulary words
│       └── split_grouped.sh             -> Script to split Wikipedia files into multiple files to clean one
│                                           multiple CPU cores
├── dataset.py                           -> Defines the data for model training
├── doc_rerank_es.py                     -> Generate and save emneddings for computer science (AI, machine learning 
│                                           and information retrieval) categories of the arXiv, also perform search by semantic similary
├── elastic                              -> Directory containing various Elastic scripts 
│   ├── delete_index_elastic.sh          -> Delete Elastic index
│   ├── increase_win_size.sh             -> Increase number of articles returned by Elasticsearch API
│   ├── list_data_files.sh               -> List Elastic data files
│   ├── list_index_elastic.sh            -> List Elastic indexes
│   ├── parallel_upload.sh               -> Upload arXiv data to Elastic using multiple CPUs in parallel
│   ├── restart_elastic.sh               -> Start/restart Elastic
│   ├── search_elastic.sh                -> Sample script to demonstrate Elastic search query
│   ├── split_arxiv_json.sh              -> Split arxiv JSON into smaller files for parallel upload to Elastic
│   └── upload_to_elastic.sh             -> Upload one small file to Elastic (to be run in parallel)
├── params.py                            -> Configuration parameters
├── quick_thoughts.py                    -> Quick-Thought f() an g() encoders implementation
├── run_websearch.sh                     -> Script to run Flash Web Semantic Search application
├── semantic_search.py                   -> Sample script to demonstrate semantic search
├── sentEval.py                          -> SentEval sentence embeddings evaluation benchmarks
├── sentence_similarity.ipynb            -> Sentence similary heatmaps code for the report illustrations
├── templates                            -> Flask Web search application pages templates
│   └── list.html                        -> Search results display page template
├── train.py                             -> Quick-Thought encoder model training code
├── utils.py                             -> Various helper functions
└── websearch.py                         -> Flask Web Search application
```
