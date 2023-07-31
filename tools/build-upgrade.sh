#!/bin/bash

# build
python setup.py bdist_wheel

# upgrade
pip install --upgrade --force-reinstall --no-deps dist/*.whl
