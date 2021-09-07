#!/bin/bash

# This script is to split Arxiv JSON file into smaller ones for the
# bulk upload to Elastic server. It is recommended to upload less than 1000
# articles, so that Elastic API does not get overloaded.

mkdir -p /tmp/arxiv
cd /tmp/arxiv
# delete old files in the directory
rm -f x*
# new files will be created with names x??
split -l 800 ../data/arxiv/arxiv-metadata-oai-snapshot_cleaned.json
