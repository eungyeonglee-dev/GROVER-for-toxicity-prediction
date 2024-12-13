#!/bin/bash
#SBATCH --nodes=5
#SBATCH --cpus-per-task=14
#SBATCH --partition=gpu2
#SBATCH -o ./log/%j/sbatch-%N.out
#SBATCH -e ./log/%j/sbatch-%N.err
#SBATCH --gres=gpu:a10:1
#SBATCH --time=00:10:00
##===============================================
GRES="gpu:a10:1"
. _00_conf.sh
mkdir -p $HOME/GROVER-for-toxicity-prediction/log
cd $HOME/GROVER-for-toxicity-prediction/log
mkdir -p $HOME/GROVER-for-toxicity-prediction/log/$SLURM_JOB_ID
echo DATA: $dataset

INIT_CONTAINER_SCRIPT=$(cat <<EOF
    
    if $RELOAD_CONTAINER ; then
        rm -rf $CONTAINER_PATH
    fi

    if [ -d "$CONTAINER_PATH" ] ; then 
        echo "container exist";
    else
        enroot create -n $CONTAINER_NAME $CONTAINER_IMAGE_PATH ;
    fi

EOF
)

ENROOT_SCRIPT="cd /GROVER-for-toxicity-prediction/ && \
               bash _01_run_inter_hpo.sh $SLURM_JOB_ID $dataset"

SRUN_SCRIPT=$(cat <<EOF
    $INIT_CONTAINER_SCRIPT

    enroot start --root \
                --rw \
                -m $HOME/GROVER-for-toxicity-prediction:/GROVER-for-toxicity-prediction \
                grover \
                bash -c "$ENROOT_SCRIPT
EOF
)

srun --partition=$SLURM_JOB_PARTITION \
     --gres=$GRES \
     --cpus-per-task=14 \
     -o $HOME/GROVER-for-toxicity-prediction/log/%j/%N.out \
     -e $HOME/GROVER-for-toxicity-prediction/log/%j/%N.err \
     bash -c "$SRUN_SCRIPT \" "