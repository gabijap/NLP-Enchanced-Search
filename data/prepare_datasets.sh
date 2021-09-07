#!/bin/bash

# This script prepares datasets

# Prepare arxiv dataset
mkdir -p arxiv
cd arxiv
chmod u+x *.sh
./download_arxiv_dataset.sh

# Prepare wikipedia dataset
mkdir -p ../wikipedia
cd ../wikipedia
chmod u+x *.sh
./download_wikipedia.sh


# Prepare bookcorpus dataset
mkdir -p ../bookcorpus
cd ../bookcorpus

# Prepare SentEval according to: https://github.com/facebookresearch/SentEval
cd ..
git clone git@github.com:facebookresearch/SentEval.git
cd SentEval/data/downstream
./get_transfer_data.bash
