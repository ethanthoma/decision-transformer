#!/usr/bin/env bash

./scripts/load_modules.sh

virtualenv decision-transformer

source decision-transformer/bin/activate

pip install --upgrade pip

pip install tinygrad ale-py gymnasium[atari] gymnasium[accept-rom-license]

deactivate
