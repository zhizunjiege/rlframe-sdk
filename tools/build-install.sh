#!/bin/bash

# build
python setup.py bdist_wheel

# install
pip install dist/*.whl
