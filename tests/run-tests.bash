#!/usr/bin/env bash

pytest tests/1D_test.py

# $? stores exit value of the last command
if [ $? -ne 0 ]; then
 echo "Tests must pass before commit!"
 exit 1
fi
