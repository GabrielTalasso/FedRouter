import random
import json
from datasets import concatenate_datasets

def split_dataset(fed_args, script_args, dataset, n_domains = 2, test = False, dataset_len = 300):


    ## Clustered (Four Domain per Client) = ALL

    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    

    if fed_args.split_strategy == "multitask_iid":
        for i in range(fed_args.num_clients):
            dataset = dataset.shuffle(seed=script_args.seed)
            this_dataset = dataset.shard(fed_args.num_clients, i)
            this_dataset = this_dataset.select(range(min(len(this_dataset), dataset_len)))
            local_datasets.append(this_dataset)


    ## Clustered (One Domain per Client) = SINGLE

    if fed_args.split_strategy == "multitask_clusters":
        tasks = ['qqp', 'gigaword', 'samsum', 'webnlg']
        n_clients_in_cluster = fed_args.num_clients // len(tasks)

        for i in range(fed_args.num_clients):
            task = tasks[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['task'] == task)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            this_dataset = cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster)
            this_dataset = this_dataset.select(range(min(len(this_dataset), dataset_len)))

            local_datasets.append(this_dataset)

    ## Multi-Domain (2 Domains per Client) = DUAL

    if fed_args.split_strategy == "multitask_multi_domain":
        tasks = ['qqp', 'gigaword', 'samsum', 'webnlg']
        n_clients_in_cluster = fed_args.num_clients // len(tasks)
        for i in range(fed_args.num_clients):
            # Each client receives data from two tasks
            task1 = tasks[i // n_clients_in_cluster]
            task2 = tasks[(i // n_clients_in_cluster + 1) % len(tasks)]
            # Filter dataset for either of the two tasks
            client_dataset_task1 = dataset.filter(lambda x, task1=task1: x['task'] == task1)
            client_dataset_task2 = dataset.filter(lambda x, task2=task2: x['task'] == task2)

            #get the fist 50% of the dataset from task1 and the last 50% from task2
            client_dataset_task1 = client_dataset_task1.select(range(len(client_dataset_task1)//2))
            client_dataset_task2 = client_dataset_task2.select(range(len(client_dataset_task2)//2, len(client_dataset_task2)))
            client_dataset = concatenate_datasets([client_dataset_task1, client_dataset_task2])
            #shuffle the dataset
            client_dataset = client_dataset.shuffle(seed=script_args.seed)

            this_dataset = client_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster)
            this_dataset = this_dataset.select(range(min(len(this_dataset), dataset_len)))
            
            local_datasets.append(this_dataset)
    # Save dataset statistics
        save_multi_domain_dataset_stats(local_datasets, script_args.output_dir)


    save_dataset_stats(local_datasets, script_args.output_dir, test = test)
    return local_datasets

def save_multi_domain_dataset_stats(local_datasets, path):
    dataset_stats = {}
    for i, dataset in enumerate(local_datasets):
        domains = set()
        for sample in dataset:
            if 'language' in sample:
                domains.add(sample['language'])
            elif 'label' in sample:
                domains.add(sample['label'])
            elif 'task' in sample:
                domains.add(sample['task'])
        dataset_stats[f'client_{i}'] = list(domains)
    with open(path + '/multi_domain_dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f)

def save_dataset_stats(local_datasets, path, test = False):
    dataset_stats = {}
    for i, dataset in enumerate(local_datasets):
        dataset_stats[f'client_{i}'] = len(dataset)
    if test:
        with open(path + '/dataset_stats_test.json', 'w') as f:
            json.dump(dataset_stats, f)
    else:
        with open(path + '/dataset_stats.json', 'w') as f:
            json.dump(dataset_stats, f)

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round