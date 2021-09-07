#!/bin/bash

# Download the latest English Wikipedia file (17GB):
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Convert from XML format to JSON
python -m gensim.scripts.segment_wiki -i -f enwiki-latest-pages-articles.xml.bz2 -o enwiki-latest.json.gz

# Convert JSON to text
python one_sentence_per_line.py > wikipedia_raw.txt

# Split into multiple files to process on multiple CPU cores
./split_grouped.sh

# Clean (remove out of vocabulary words, put once sentence per line) in parallel using multiple CPU cores
./parallel_clean.sh
