#!/bin/bash
pycodestyle --first *.py
pycodestyle --first utils/*.py
coverage run -m unittest discover -s tests/
coverage report --omit="*/tests*" -m
coverage html --omit="*/tests*"
