#!/bin/bash

# This is to list indexes on Elastic server
curl "localhost:6006/_cat/indices?v=true"
curl -XPUT "http://localhost:6006/arxiv/_settings" -H 'Content-Type: application/json' -d '{ "index" : { "max_result_window" : 500000 } }'
