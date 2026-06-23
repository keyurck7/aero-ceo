#!/usr/bin/env bash
set -e

python -m app.evaluation.evaluation_suite "$@"
