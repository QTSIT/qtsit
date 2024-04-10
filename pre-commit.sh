#!/bin/bash

# Check if file argument was provided
if [ $# -eq 0 ]; then
  echo "Error: no file provided"
  exit 1
fi

# Get filename from first argument
file=$1

# Format file with yapf
echo "Formatting $file with yapf"
python -m yapf -i "$file"

# Type check qtsit package
echo "Type checking deepchem package with mypy"
python -m mypy -p qtsit

# Lint file with flake8 and show count
echo "Linting $file with flake8"
python -m flake8 "$file" --count

# Check if filename contains "test"
if [[ $file != *"test"* ]]; then
  # Test file with doctest
  echo "Testing $file with doctest"
  python -m doctest "$file"
else
  echo "Skipping doctest for test file $file"
  echo "running pytest on $file"
  python -m pytest "$file"
fi
~
