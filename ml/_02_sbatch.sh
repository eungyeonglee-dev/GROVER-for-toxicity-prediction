#!/bin/bash
#SBATCH --nodes=5
#SBATCH --partition=gpu2
#SBATCH -o ../log/%j/sbatch-%n.out
#SBATCH -e ../log/%j/sbatch-%n.err
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=14
#============================================================
. _00_conf.sh

CPUS_PER_TASK=14
cd $HOME/grover-for-toxicity/log
mkdir -p $HOME/grover-for-toxicity/log/$SLURM_JOB_ID

filename="_01_tox21_hpo.py"

# check old container in calculatation node and remove.
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

ENROOT_SCRIPT="cd /grover-for-toxicity/ml && \
               /opt/conda/envs/deepchem/bin/python $filename $model_name $chunk_num"

SRUN_SCRIPT=$(cat <<EOF
    NODE_LIST=\`scontrol show hostnames \$SLURM_JOB_NODELIST\`
    node_array=(\$NODE_LIST)
    length=\${#node_array[@]}
    hostnode=\`hostname -s\`
    for (( index = 0; index < length ; index++ )); do
        node=\${node_array[\$index]}
        if [ \$node == \$hostnode ]; then
            local_rank=\$index
        fi
    done

    $INIT_CONTAINER_SCRIPT

    enroot start --root \
                --rw \
                -m $HOME/grover-for-toxicity:/grover-for-toxicity \
                $CONTAINER_NAME \
                bash -c "$ENROOT_SCRIPT \$local_rank "
    
EOF
)

srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=$CPUS_PER_TASK \
      -o ../log/%j/%N.out \
      -e ../log/%j/%N.err \
      bash -c "$SRUN_SCRIPT"