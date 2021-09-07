#!/bin/bash

curl --output /dev/null --show-error --fail -H "Content-Type: application/json" -XPOST "localhost:6006/$1/_bulk?pretty&refresh" --data-binary "@$2"
sleep 5
