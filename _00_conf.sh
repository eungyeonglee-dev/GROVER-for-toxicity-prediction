#!/bin/bash

CONTAINER_IMAGE_PATH="$HOME/grover-for-toxicity/container/grover.sqsh"
CONTAINER_PATH="/enroot/$UID/data/grover"
CONTAINER_NAME="grover"
RELOAD_CONTAINER=false
# tox21, lc50, lc50_2 which remove outlier point from lc50
dataset="lc50_2" 