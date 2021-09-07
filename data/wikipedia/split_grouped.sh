#!/bin/bash

# This script is to split Wikipedia JSON file into smaller ones for the
# bulk upload to Elastic server. It is recommended to upload less than 1000
# articles, so that Elastic API does not get overloaded.

mkdir -p ./grouped
cd ./grouped
# delete old files in the directory
rm -f x*
# new files will be created with names x????
split -l 32425320 ../wikipedia_raw.txt