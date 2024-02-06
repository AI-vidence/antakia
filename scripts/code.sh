#!/bin/sh
poetry run black app
poetry run isort app
poetry run yapf -i -r app