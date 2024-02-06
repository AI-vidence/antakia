#!/usr/bin/env sh

poetry run pylint --load-plugins pylint_quotes --reports=no --output-format=text -s n app
