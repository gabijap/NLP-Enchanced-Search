#!/bin/bash

# This is to list indexes on Elastic server
if [ -z "$1" ]
  then
    echo "Please specify which index to delete"
else
    curl -XDELETE "localhost:6006/$1"
fi
