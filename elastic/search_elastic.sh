#!/bin/bash

# This is to run a search on Elastic server
curl "localhost:6006/wikijson/_search?q=text:$1"
