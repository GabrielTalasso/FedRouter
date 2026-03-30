#!/bin/bash

# Training configuration
max_steps=10
num_train_epochs=1
num_rounds=25
eval_round="3,10,25,50,75,100"
batch_size=16
batch_size_eval=128
gradient_accumulation_steps=1
seq_length=1024
num_clients=8
sample_clients=8
lora_r=8
lora_alpha=16  # twice of lora_r
lr=5e-4

# Dataset configuration
dataset_name='multitask'
dataset_sample=400000

# Model configuration
model_name_or_path='meta-llama/Llama-3.2-1B'
seeds="111 222 333 444 555"


# Output configuration
#output_dir="output_multilanguage/experience"
output_dir="output_multitask/experiments_1b"
sim_alias='router_single'

# Federated learning configuration
fed_alg="clustered"
sim_round=105                         # Round at which clustering happens (ignored for fedrouter)
n_clusters=1                          # Number of clusters to create localy
global_n_clusters=4                   #Number of clusters on the federation
split_strategy="multitask_clusters"   # How to split data among clients (cluster for 'single', multi_domain for 'dual' and iid for 'all')
train_split=0.8
evaluation_mode="local"               #local or global (routing type at inference time, only for fedrouter)

gpu='0,1' # Hardware configuration - GPUs IDs

# Advanced configuration
client_resources_cpus=1
client_resources_gpus=1

# Create output directory
mkdir -p $output_dir

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$gpu

# Run the Flower simulation
echo "Starting Flower simulation..."

for seed in $seeds; do
   echo "===== Running with seed: $seed ====="
     python simulation.py \
         --learning_rate $lr \
         --model_name_or_path $model_name_or_path \
         --dataset_name $dataset_name \
         --dataset_sample $dataset_sample \
         --fed_alg $fed_alg \
         --num_clients $num_clients \
         --sample_clients $sample_clients \
         --max_steps $max_steps \
         --num_train_epochs $num_train_epochs \
         --num_rounds $num_rounds \
         --batch_size $batch_size \
         --gradient_accumulation_steps $gradient_accumulation_steps \
         --seq_length $seq_length \
         --peft_lora_r $lora_r \
         --peft_lora_alpha $lora_alpha \
         --use_peft True \
         --load_in_4bit True \
         --output_dir $output_dir \
         --template "alpaca" \
         --sim_round $sim_round \
         --n_clusters $n_clusters \
         --global_n_clusters $global_n_clusters \
         --split_strategy $split_strategy \
         --train_split $train_split \
         --sim_alias ${sim_alias}_seed${seed} \
         --evaluation_rounds $eval_round \
         --eval_batch_size $batch_size_eval \
         --evaluation_mode $evaluation_mode \
         --client_resources_cpus $client_resources_cpus \
         --client_resources_gpus $client_resources_gpus \
         --seed $seed \
         --run_simulation > $output_dir/${sim_alias}_seed${seed}_log.txt
   echo "===== Finished seed: $seed ====="
 done