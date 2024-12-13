#/bin/bash

CONTAINER_IMAGE_PATH='$HOME/grover-for-toxicity/container/deepchem.sqsh'
CONTAINER_PATH="/enroot/$UID/data/deepchem"
CONTAINER_NAME="deepchem"
RELOAD_CONTAINER=false

#========== partition configuration
model_name="gcn" # gcn, logreg, rf, svm
chunk_num=3