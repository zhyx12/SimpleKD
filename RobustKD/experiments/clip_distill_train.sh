#!/usr/bin/env bash
job_id=$1
config_file=$2

project_home='RobustKD'
###
shell_folder=$(cd "$(dirname "$0")"; pwd)
echo $shell_folder
source $shell_folder/process_data.sh
source $shell_folder/process_home_path.sh
echo $HOME
cd $HOME'/PycharmProjects/'${project_home} || exit
export HOME=$HOME

trainer_class=clip_distill
validator_class=clip_distill
scripts_path=$HOME'/PycharmProjects/'${project_home}'/experiments/get_visible_card_num.py'
port_scripts_path=$HOME'/PycharmProjects/'${project_home}'/experiments/generate_random_port.py'
GPUS=$(python ${scripts_path})
PORT=$(python ${port_scripts_path})

export PYTHONPATH=$HOME'/.local/lib/python3.7/site-packages'
python_file=$HOME'/PycharmProjects/'${project_home}/train.py
# CUDA_LAUNCH_BLOCKING=1  #
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#  ${python_file} --job_id ${job_id} --config ${config_file} \
#  --trainer ${trainer_class} --validator ${validator_class}
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
  ${python_file} --job_id ${job_id} --config ${config_file} \
  --trainer ${trainer_class} --validator ${validator_class}

#python_file=./train.py
## CUDA_LAUNCH_BLOCKING=1
#CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#  ${python_file} --task_type cls --job_id ${job_id} --config ${config_file} \
#  --trainer ${trainer_class} --validator ${validator_class}
