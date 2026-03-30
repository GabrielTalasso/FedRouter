import copy
import os
import gc
from tqdm import tqdm
import numpy as np
from math import floor

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training 
import torch
from transformers import set_seed

from utils import *
from utils.utils import default_evaluation, save_dataset_test
from federated_learning import *
from federated_learning.router_utils import *
from config import get_config, save_config, get_model_config, get_training_args


def peft_state_to_cpu(state_dict):
    """Move a PEFT state dict to CPU tensors to reduce GPU memory pressure."""
    return {k: v.to("cpu") for k, v in state_dict.items()}

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()

set_seed(script_args.seed)
torch.manual_seed(script_args.seed)
np.random.seed(script_args.seed)

training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset, dataset_test = get_dataset(script_args.dataset_name, script_args.local_data_dir, script_args.train_split)

dataset =      process_sft_dataset(script_args.dataset_name, dataset,      script_args.dataset_sample)
dataset_test = process_sft_dataset(script_args.dataset_name, dataset_test, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset, dataset_len = script_args.max_data_per_client)

if fed_args.evaluation_mode == "local":
    local_datasets_test = split_dataset(fed_args, script_args, dataset_test, test = True, dataset_len = script_args.max_eval_size)

elif fed_args.evaluation_mode == "global":
    aux_split_strategy = fed_args.split_strategy #save the true value
    fed_args.split_strategy = fed_args.split_strategy.split('_')[0] + '_iid' #evaluate with iid data (all domains)
    local_datasets_test = split_dataset(fed_args, script_args, dataset_test, test = True, dataset_len = script_args.max_eval_size)
    fed_args.split_strategy = aux_split_strategy #restore the original value


sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Free up original datasets to save memory =====
del dataset, dataset_test
gc.collect()  # Force garbage collection

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
print(f"Model loaded from {script_args.model_name_or_path}")

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = peft_state_to_cpu(copy.deepcopy(get_peft_model_state_dict(model)))
#local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # following vicuna

if tokenizer.eos_token == tokenizer.unk_token or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    print(f"Pad token is set to {tokenizer.pad_token}.")

print('Special tokens:', tokenizer.special_tokens_map)
model.resize_token_embeddings(len(tokenizer))

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
if response_template:
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    packing = False
else:
    data_collator = None
    packing = True

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
idx = None

client_embeddings_centers = {}
data_cluster_labels = {}

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, script_args, round)
    
    if round  == 0: #return all clients
        clients_this_round = list(range(fed_args.num_clients))

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    global_clusters_this_round = [] ###
    local_dict_list = [ [ ] for _ in range(fed_args.num_clients* fed_args.n_clusters)] ########################

    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue
                
        print(f"Getting Embeddings for client {client}...")
        if client not in client_embeddings_centers and round == 0:
            client_embeddings_centers[client], data_cluster_labels[client] = cluster_embeddings(
                            get_client_embedding(script_args, fed_args, local_datasets[client], quantization_config),
                            num_clusters = fed_args.n_clusters
                            )

            if fed_args.fed_alg == 'router_oracle':
                dict_labels = {task: i for i, task in enumerate(np.unique(local_datasets[client]['task']))}
                oracle_labels = [dict_labels[task] for task in local_datasets[client]['task']]
                local_datasets[client] = local_datasets[client].add_column('cluster_label', oracle_labels)
            else:
                local_datasets[client] = local_datasets[client].add_column('cluster_label', data_cluster_labels[client])
        
        print(f"Client {client} embeddings center shape: {client_embeddings_centers[client].shape}")
        print(f"Client {client} data cluster labels: {np.unique(data_cluster_labels[client])}")

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
        sub_dataset_test = local_datasets_test[client]
        sub_dataset_test = sub_dataset_test.shuffle(seed=round).select(range(script_args.max_eval_size) if script_args.max_eval_size < len(sub_dataset_test) else range(len(sub_dataset_test)))

        if (round+1) in [int(x) for x in fed_args.evaluation_rounds.split(",")]:
            
            test_embeddings = get_client_embedding(script_args, fed_args, sub_dataset_test, quantization_config)
            #save all test embeddings for the client
            test_embeddings_path = os.path.join(script_args.output_dir, f"clients_test_datasets/embeddings/test_embeddings_{client}_round_{round+1}.npy")
            os.makedirs(os.path.dirname(test_embeddings_path),
                        exist_ok=True)
            np.save(test_embeddings_path, test_embeddings)

            if fed_args.evaluation_mode == 'global':
                print("Global evaluation mode: using all clusters adapters")
                infered_cluster_labels = clusterize_dataset(test_embeddings, global_centroids)

                if fed_args.fed_alg == 'router_oracle':
                    #infered_cluster_labels = [dict_labels[task] for task in sub_dataset_test['task']]
                    infered_cluster_labels = [get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][dict_labels[task]]) for task in sub_dataset_test['task']]

            if fed_args.evaluation_mode == 'local':
                test_global_clusters = []
                for c_embed_center in client_embeddings_centers[client]:
                    test_global_clusters.append(get_most_similar_adapter(global_centroids, global_clusters, c_embed_center))

                print(f"Test global clusters: {test_global_clusters}")
                infered_cluster_labels = clusterize_dataset(test_embeddings, global_centroids[test_global_clusters])

                #mapping the labels to the global clusters
                infered_cluster_labels = [test_global_clusters[label] for label in infered_cluster_labels]

                if fed_args.fed_alg == 'router_oracle':
                    infered_cluster_labels = [get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][dict_labels[task]]) for task in sub_dataset_test['task']]

            sub_dataset_test = sub_dataset_test.add_column('cluster_label', infered_cluster_labels)
            print(f"Detected clusters in the test set for client {client}: {np.unique(infered_cluster_labels)}")
            save_dataset_test(sub_dataset_test, script_args, client, round)

            for c in np.unique(infered_cluster_labels):
                # Evaluate for all global cluster (a client can have data from a diverse domain - test time personalization - generability)
                #global_centroid_id = get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][c])
                set_peft_model_state_dict(model, global_dict[c])
                test_dataset_this_cluster = sub_dataset_test.filter(lambda x: x['cluster_label'] == c)

                print(f"Evaluating client {client} on the test set with size {len(test_dataset_this_cluster)} for cluster {c} in round {round}...")
                default_evaluation(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=test_dataset_this_cluster,
                    client_id=client,
                    round=round, #with respect to model from the previous round
                    formatting_prompts_func=formatting_prompts_func,
                    script_args=script_args,
                    cluster_id=c,
                )

            del test_embeddings
            del sub_dataset_test
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()

        #local_dict_list[client] = []
        training_loss_aux = []
        
        print('Length of global_dict: ', len(global_dict))

        #RANDOM CLUSTER SELECTION
        #selected_cluster = np.random.choice(np.arange(fed_args.n_clusters), size=1, replace=True)[0] ###

        #ROUND ROBIN CLUSTER SELECTION
        #selected_clusters = [0,1]#round % fed_args.n_clusters ###
        selected_cluster = list(range(fed_args.n_clusters))

        for c in range(fed_args.n_clusters):
            #cluster_dataset = sub_dataset.filter(lambda x: x['cluster_label'] == c)

            cluster_dataset = local_datasets[client].filter(lambda x: x['cluster_label'] == c).shuffle(seed=round) ###
            cluster_dataset = get_dataset_this_round(cluster_dataset, round, fed_args, script_args) ###

            #starts with global model
            print(f'Setting parameters for client {client}...')
            if round >= 1:
                print(f'Setting global idx {get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][c])} for client {client}...')
                set_peft_model_state_dict(model, global_dict[get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][c])])
            else:
                set_peft_model_state_dict(model, global_dict)

            if not os.path.exists(os.path.join(script_args.output_dir, "clients_adapters")):
                os.makedirs(os.path.join(script_args.output_dir, "clients_adapters"))

            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-5)
            training_args = get_training_args(script_args, new_lr)
            
            #ajusted_max_steps = (len(cluster_dataset) / len(sub_dataset)) * script_args.max_steps
            training_args.max_steps = script_args.max_steps# max(floor(ajusted_max_steps), 1) ###

            print(f"Training for {len(cluster_dataset)} samples in cluster {c} for client {client} with {training_args.max_steps} steps...")

            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=cluster_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                packing=packing
                )

            if round >= 1:
                #for c in selected_clusters: ###
                results = trainer.train()
                training_loss_aux.append(results.training_loss)
                local_dict_list[client].append(peft_state_to_cpu(copy.deepcopy(get_peft_model_state_dict(model))))
                print('Most similar global adapter for client', client, 'in cluster', c, 'is:', get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][c]))
                global_clusters_this_round.append(get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][c]))

            else:
                results = trainer.train()
                training_loss_aux.append(results.training_loss)
                local_dict_list[client].append(peft_state_to_cpu(copy.deepcopy(get_peft_model_state_dict(model))))
                #global_clusters_this_round.append(global_dict[c])
  
            # Clean up cluster dataset
            del cluster_dataset
            gc.collect()  # Force garbage collection
                
        training_loss[client].append(np.mean(training_loss_aux))
        torch.cuda.empty_cache()
    
    if round == 0:
        client_embeddings_centers_list = []
        for client in range(fed_args.num_clients):
            for client_embeddings_center in client_embeddings_centers[client]:
                client_embeddings_centers_list.append(client_embeddings_center)

        global_centroids, global_clusters = cluster_clients_centroids(client_embeddings_centers_list, num_clusters = fed_args.global_n_clusters)
        print(f'Global clusters found: {global_clusters}')
    
        #saving local and global centroids
        print(f"Saving centroids...")

        centroids_path = os.path.join(script_args.output_dir, f"centers/centroids_{round+1}.npy")
        os.makedirs(os.path.dirname(centroids_path), exist_ok=True)
        np.save(centroids_path, global_centroids)

        client_embeddings_path = os.path.join(script_args.output_dir, f"centers/client_embeddings_{round+1}.npy")
        os.makedirs(os.path.dirname(client_embeddings_path), exist_ok=True)
        np.save(client_embeddings_path, client_embeddings_centers)

        clusters_path = os.path.join(script_args.output_dir, f"centers/global_clusters_{round+1}.npy")
        os.makedirs(os.path.dirname(clusters_path), exist_ok=True)
        np.save(clusters_path, global_clusters)

        for c in range(fed_args.num_clients):
            local_datasets[c].save_to_disk(os.path.join(script_args.output_dir, f"clients_train_datasets/client_{c}_round_{round}"))

    #Flattening the local_dict_list
    all_local_dict_list = []
    for client in range(fed_args.num_clients):
        for adapter in local_dict_list[client]:
            all_local_dict_list.append(adapter)

    del local_dict_list
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()
    
    print("Length of all_local_dict_list: ", len(all_local_dict_list))

    #Getting only the global clusters and adapters for clients this round 
    if round == 0:
        global_clusters_this_round = []
        local_dict_list_this_round = []
        for i in range(0, len(global_clusters), fed_args.n_clusters):
            if i // fed_args.n_clusters in clients_this_round:
                global_clusters_this_round += global_clusters[i:i+fed_args.n_clusters].tolist()
                #local_dict_list_this_round += all_local_dict_list[i:i+fed_args.n_clusters]

    # ===== Server aggregates the local models =====
    
    global_dict, global_auxiliary, idx = global_aggregate( ###
            fed_args, script_args, global_dict, all_local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            round = round, idx = global_clusters_this_round
        )

    for cluster in range(fed_args.global_n_clusters):
        set_peft_model_state_dict(model, global_dict[cluster])
        trainer.save_model(os.path.join(script_args.output_dir, f"cluster_{cluster}_checkpoint-{round+1}"))

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
