#!/bin/bash

# Run data cleaning (remove out of vocabulary words and put one sentence per line in parallel on multiple CPU cores

mkdir -p ./cleaned
rm -f ./cleaned/*
cd ./grouped
parallel python ../clean_data.py ::: x??
cd ../cleaned
cat * > ../wikipedia_all_cleaned-voc400000-ln160M.txt
cd ..

# Create variations
cat wikipedia_all_cleaned-voc400000-ln160M.txt | head -149000000 > wikipedia_all_cleaned-voc400000-ln160M-train.txt
cat wikipedia_all_cleaned-voc400000-ln160M.txt | tail -11685048 > wikipedia_all_cleaned-voc400000-ln160M-test.txt
cat wikipedia_all_cleaned-voc400000-ln160M.txt | head -75000000 > wikipedia_all_cleaned-voc400000-ln160M-train-half.txt
cat wikipedia_all_cleaned-voc400000-ln160M.txt | head -37500000 > wikipedia_all_cleaned-voc400000-ln160M-train-quarter.txt

