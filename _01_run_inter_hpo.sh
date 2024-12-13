#!/bin/bash

jobid=$1
dataset=$2
hostname=`hostname -s`

# model save dir
dir=/grover-for-toxicity/model/finetune/${dataset}/${jobid}/${hostname}

# dataset, features dir
BASELINE='/grover-for-toxicity/exampledata/finetune'
# dataset is the one of tox21, lc50, lc50_2
data_path="${BASELINE}/${dataset}.csv" # only smiles string
features_path="${BASELINE}/${dataset}.npz"
dataset_type="regression" # if you select lc50, it must be regression

echo "jobid: $jobid"
echo "dataset: $dataset"
echo "model save dir: $dir"
mkdir -p $dir

# HPO function
function_parm() {
batch_size=32

arr=(1 2 3 4 5 6 7 8 9 10)
SEED=$(head -1 /dev/urandom | od -N 1 | awk '{ print $2 }')
r=$(($SEED%${#arr[*]}))

max_lr=$( echo "scale=4; 0.0001*${arr[$r]}" | bc )
init_lr=$( echo "scale=5; $max_lr/10" | bc )

arr=(2 3 4 5 6 7 8 9 10)
SEED=$(head -1 /dev/urandom | od -N 1 | awk '{ print $2 }')
r=$(($SEED%${#arr[*]}))
final_lr=$( echo "scale=5; $max_lr/${arr[$r]}"|bc)

arr_dropout=(0 0.05 0.1 0.2)
SEED=$(head -1 /dev/urandom | od -N 1 | awk '{ print $2 }')
r=$(($SEED%${#arr_dropout[*]}))
dropout=${arr_dropout[$r]}

arr_attn_out=(4 8)
SEED=$(head -1 /dev/urandom | od -N 1 | awk '{ print $2 }')
r=$(($SEED%${#arr_attn_out[*]}))
attn_out=${arr_attn_out[$r]}

attn_hidden=128

arr_dist_coff=(0.05 0.1 0.15)
SEED=$(head -1 /dev/urandom | od -N 1 | awk '{ print $2 }')
r=$(($SEED%${#arr_dist_coff[*]}))
dist_coff=${arr_dist_coff[$r]}

arr_bond_drop_rate=(0 0.2 0.4 0.6)
SEED=$(head -1 /dev/urandom | od -N 1 | awk '{ print $2 }')
r=$(($SEED%${#arr_bond_drop_rate[*]}))
bond_drop_rate=${arr_bond_drop_rate[$r]}

arr_ffn_num_layers=(2 3)
SEED=$(head -1 /dev/urandom | od -N 1 | awk '{ print $2 }')
r=$(($SEED%${#arr_ffn_num_layers[*]}))
ffn_num_layers=${arr_ffn_num_layers[$r]}

arr_ffn_hidden_size=(5 7 13)
SEED=$(head -1 /dev/urandom | od -N 1 | awk '{ print $2 }')
r=$(($SEED%${#arr_ffn_hidden_size[*]}))
ffn_hidden_size=$( echo "${arr_ffn_hidden_size[$r]}*100"|bc)
}

s=1
n=30 # the number of finetune model
f=$(($s+$n-1))

for i in `seq $s 1 $f`
do
save_dir="$dir/$i"
echo "save_dir: $save_dir"

if [[ ! -e $save_dir ]]; then
    mkdir $save_dir
    function_parm
    echo "init_lr: $init_lr"
    echo "max_lr: $max_lr"
    echo "final_lr: $final_lr"
    echo "dropout: $dropout"
    echo "ffn_hidden_size: $ffn_hidden_size"
    echo "ffn_num_layers: $ffn_num_layers"
    echo "attn_hidden: $attn_hidden"
    echo "attn_out: $attn_out"
    echo "dist_coff: $dist_coff"
    echo "bond_drop_rate: $bond_drop_rate"

    /opt/conda/envs/horovod/bin/python /grover-for-toxicity/main.py finetune --epochs 100 \
                                   --data_path $data_path \
                                   --features_path $features_path \
                                   --save_dir $save_dir \
                                   --checkpoint_path /grover-for-toxicity/grover_large.pt \
                                   --split_type scaffold_balanced \
                                   --num_folds 3 \
                                   --ensemble_size 1 \
                                   --dataset_type $dataset_type \
                                   --no_features_scaling \
                                   --init_lr $init_lr \
                                   --max_lr $max_lr \
                                   --final_lr $final_lr \
                                   --dropout $dropout \
                                   --ffn_hidden_size $ffn_hidden_size \
                                   --ffn_num_layers $ffn_num_layers \
                                   --attn_hidden $attn_hidden \
                                   --attn_out $attn_out \
                                   --dist_coff $dist_coff \
                                   --bond_drop_rate $bond_drop_rate | tee $save_dir/debugging.log

else
    "$save_dir already exists"
fi
done
