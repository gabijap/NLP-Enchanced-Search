#!/bin/bash

# This script is to split Wikipedia JSON file into smaller ones for the
# bulk upload to Elastic server. It is recommended to upload less than 1000
# articles, so that Elastic API does not get overloaded.

mkdir -p /tmp/wikipedia
cd /tmp/wikipedia
# delete old files in the directory
rm -f x*
# new files will be created with names x????
split -l 800 ../data/wikipedia/wikipedia_all.json
