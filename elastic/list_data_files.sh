#!/bin/bash

# This is to list indexes on Elastic server
curl "localhost:6006/_nodes/stats/fs?pretty"
