#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the first Python script
python outlier_word.py

echo "## outlier_word.py is running.. ##"

# Run the second Python script
python example.py

echo "## example.py is running.. ##"