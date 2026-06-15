#!/usr/bin/env sh

poetry run python -m pytest --cov=antakia --cov-report term-missing tests
