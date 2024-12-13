#/bin/bash

# CONTAINER_IMAGE_PATH='$HOME/GROVER-for-toxicity-prediction/container/deepchem.sqsh'
CONTAINER_IMAGE_PATH="$HOME/deepchem.sqsh"
CONTAINER_PATH="/enroot/$UID/data/deepchem"
CONTAINER_NAME="deepchem"
RELOAD_CONTAINER=true

#========== partition configuration
model_name="gcn" # gcn, logreg, rf, svm
chunk_num=3