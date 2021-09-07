#!/bin/bash

kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip
python clean_arxiv.py > arxiv-metadata-oai-snapshot_cleaned.json
rm -f arxiv-metadata-oai-snapshot.json
