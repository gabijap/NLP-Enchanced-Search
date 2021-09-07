#!/bin/bash

# This is the script to start Flask server on port :8097
export FLASK_APP=/websearch.py
python -m flask run --host=0.0.0.0 --port=8097
