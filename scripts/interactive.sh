#!/usr/bin/env bash

srun --account=st-gerope-1-gpu --partition=interactive_gpu --time=1:0:0 -N 1 -n 1 --cpus-per-task 24 --mem 48G --gres=gpu:v100:1 --pty /bin/bash
