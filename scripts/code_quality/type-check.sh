#!/usr/bin/env sh

poetry run mypy --non-interactive --install-types --show-column-numbers --no-color-output --no-error-summary --show-error-context app
