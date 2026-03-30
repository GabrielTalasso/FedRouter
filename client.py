
import os
import copy
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import gc

from flwr.client import NumPyClient
from flwr.common import Parameters, parameters_to_ndarrays
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from datasets import Dataset
from transformers import AutoTokenizer
from flwr.common.typing import NDArrays, Scalar
from collections import OrderedDict

# Import project utilities
from utils.utils import cosine_learning_rate, default_evaluation, save_dataset_test
from federated_learning.split_dataset import get_dataset_this_round
from federated_learning.fed_local_sft import get_fed_local_sft_trainer
from flower_utils import get_model_flower
from utils.utils import default_evaluation, save_dataset_test
from federated_learning import *
from federated_learning.router_utils import *

class FedRouterClient(NumPyClient):
    
    def __init__(self, 
                 cid: int, 
                 model: torch.nn.Module,
                 tokenizer: AutoTokenizer,
                 peft_config,
                 device_map,
                 quantization_config,
                 torch_dtype,
                 local_dataset: Dataset,
                 local_dataset_test: Dataset,
                 script_args,
                 fed_args,
                 training_args,
                 formatting_prompts_func,
                 data_collator=None,
                 packing: bool = True,
                 device: torch.device = None,
                 output_dir: str = "./output"):
        
        self.cid = cid
        #self.tokenizer = tokenizer
        self.local_dataset = local_dataset
        self.local_dataset_test = local_dataset_test
        self.script_args = script_args
        self.fed_args = fed_args
        self.training_args = training_args
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.packing = packing
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Cluster information - will be set by server
        self.cluster_id = ''
        self.training_losses = []
        self.quantization_config = quantization_config
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.peft_config = peft_config

        self.model, self.tokenizer = get_model_flower(script_args, training_args, peft_config,
                                        device_map, quantization_config, torch_dtype)
        
        print(f"Client {self.cid} initialized with {len(self.local_dataset)} training samples")
    
    def get_parameters(self) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        peft_state_dict_keys = get_peft_model_state_dict(self.model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(self.model, state_dict)

    def fit(self, parameters: Parameters, config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        
        print('Initializing client {} fit...'.format(self.cid))

        current_round = config["current_round"]
        total_rounds = config["total_rounds"]
        cluster_id = config.get("cluster_id")

        print(f"\n=== Client {self.cid} - Round {current_round} ===")


        self.set_parameters(parameters)

        self.model.to(self.device)

        if current_round == 1:
            self.client_embeddings_centers, self.data_cluster_labels = cluster_embeddings(
                            get_client_embedding(self.script_args, self.fed_args, self.local_dataset, self.quantization_config),
                            num_clusters = self.fed_args.n_clusters
                            )

            self.local_dataset = self.local_dataset.add_column('cluster_label', self.data_cluster_labels)

            #save new dataset with cluster labels
            cluster_dataset_path = os.path.join(self.script_args.output_dir, f"client_{self.cid}_clustered_dataset")
            self.local_dataset.save_to_disk(cluster_dataset_path)
            print(f"Client {self.cid} clustered dataset saved to {cluster_dataset_path}")

            #save embeddings centers
            embeddings_path = os.path.join(self.script_args.output_dir, f"client_{self.cid}_embeddings_centers.npy")
            np.save(embeddings_path, self.client_embeddings_centers)
            print(f"Client {self.cid} embeddings centers saved to {embeddings_path}")

        else:
            # Load client embeddings centers from file
            embeddings_path = os.path.join(self.script_args.output_dir, f"client_{self.cid}_embeddings_centers.npy")
            self.client_embeddings_centers = np.load(embeddings_path)
            print(f"Client {self.cid} loaded embeddings centers from {embeddings_path}")

            # Load clustered dataset from disk
            cluster_dataset_path = os.path.join(self.script_args.output_dir, f"client_{self.cid}_clustered_dataset")
            self.local_dataset = Dataset.load_from_disk(cluster_dataset_path)
            print(f"Client {self.cid} loaded clustered dataset from {cluster_dataset_path}")

        print(f"Client {self.cid} embeddings center shape: {self.client_embeddings_centers.shape}")

        # Update learning rate with cosine schedule  
        new_lr = cosine_learning_rate(
            current_round, total_rounds, self.script_args.learning_rate, 1e-5
        )

        # Update training arguments
        updated_training_args = copy.deepcopy(self.training_args)
        updated_training_args.learning_rate = new_lr
        updated_training_args.output_dir = os.path.join(
            self.script_args.output_dir, f"client_{self.cid}_round_{current_round}"
        )
            
        print(f"Client {self.cid} selected cluster for this round: {cluster_id}")

        self.cluster_dataset = self.local_dataset.filter(lambda x: x['cluster_label'] == cluster_id).shuffle(seed=current_round) ###
        self.cluster_dataset = get_dataset_this_round(self.cluster_dataset, current_round, self.fed_args, self.script_args)

        new_lr = cosine_learning_rate(current_round, self.fed_args.num_rounds, self.script_args.learning_rate, 1e-5)

        #ajusted_max_steps = (len(cluster_dataset) / len(sub_dataset)) * script_args.max_steps
        updated_training_args.max_steps = self.script_args.max_steps# max(floor(ajusted_max_steps), 1) ###


        trainer = get_fed_local_sft_trainer(
            script_args=self.script_args,
            fed_args=self.fed_args,
            model=self.model,
            tokenizer=self.tokenizer,
            training_args=updated_training_args,
            local_dataset=self.cluster_dataset,
            formatting_prompts_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
            global_dict=None,  # Not needed for clustering approach
            local_auxiliary=None,  # Not using SCAFFOLD
            global_auxiliary=None,
            packing=self.packing,
        )

        results = trainer.train()

        torch.cuda.empty_cache()

        self.training_losses.append(results.training_loss)

        # Extract updated parameters (LoRA adapter state dict)
        updated_parameters = self.get_parameters()

        dataset_size = len(self.local_dataset)

        state_dict_keys = get_peft_model_state_dict(self.model).keys()

        #client centers to list

        metrics = {
            "training_loss": results.training_loss,
            "dataset_size": dataset_size,
            "cluster_id": cluster_id,
            "learning_rate": new_lr,
            "client_embedding": str(self.client_embeddings_centers.tolist()),
            "state_dict_keys": str(list(state_dict_keys))
        }

        print(f"Client {self.cid} completed training. Loss: {results.training_loss:.4f}")

        return updated_parameters, dataset_size, metrics
    
    def evaluate(self, parameters: Parameters, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:

        print('Initializing client {} evaluation...'.format(self.cid))

        server_round = config["current_round"]
        global_centroids = np.array(eval(config.get("global_centroids")))
        global_clusters = np.array(eval(config.get("global_clusters")))

        if server_round in [int(x) for x in self.fed_args.evaluation_rounds.split(",")]:
            sub_dataset_test = self.local_dataset_test
            sub_dataset_test = sub_dataset_test.shuffle(seed=server_round).select(range(self.script_args.max_eval_size) if self.script_args.max_eval_size < len(sub_dataset_test) else range(len(sub_dataset_test)))

            print('Generating test embeddings for client {}...'.format(self.cid))
            test_embeddings = get_client_embedding(self.script_args, self.fed_args, sub_dataset_test, self.quantization_config)
            print(f"Test embeddings shape: {test_embeddings.shape}")
            #save all test embeddings for the client
            test_embeddings_path = os.path.join(self.script_args.output_dir, f"clients_test_datasets/embeddings/test_embeddings_{self.cid}_round_{server_round}.npy")
            os.makedirs(os.path.dirname(test_embeddings_path),
                        exist_ok=True)
            np.save(test_embeddings_path, test_embeddings)

            self.client_embeddings_centers = np.load(os.path.join(self.script_args.output_dir, f"client_{self.cid}_embeddings_centers.npy"))

            if self.fed_args.evaluation_mode == 'global' or self.fed_args.evaluation_mode == 'data_local_eval_global':
                print("Global evaluation mode: using all clusters adapters")
                infered_cluster_labels = clusterize_dataset(test_embeddings, global_centroids)

            if self.fed_args.evaluation_mode == 'local' or self.fed_args.evaluation_mode == 'data_global_eval_local':
                test_global_clusters = []
                for c_embed_center in self.client_embeddings_centers:
                    test_global_clusters.append(get_most_similar_adapter(global_centroids, global_clusters, c_embed_center))

                print(f"Test global clusters: {test_global_clusters}")
                infered_cluster_labels = clusterize_dataset(test_embeddings, global_centroids[test_global_clusters])

                #mapping the labels to the global clusters
                infered_cluster_labels = [test_global_clusters[label] for label in infered_cluster_labels]

            sub_dataset_test = sub_dataset_test.add_column('cluster_label', infered_cluster_labels)
            print(f"Detected clusters in the test set for client {self.cid}: {np.unique(infered_cluster_labels)}")
            save_dataset_test(sub_dataset_test, self.script_args, self.cid, server_round)

            for c in np.unique(infered_cluster_labels):
                # Evaluate for all global cluster (a client can have data from a diverse domain - test time personalization - generability)
                #global_centroid_id = get_most_similar_adapter(global_centroids, global_clusters, client_embeddings_centers[client][c])
                model_path = self.output_dir + f'/cluster_models/round_{server_round}/cluster_{c}'
        
                # Load the model for this cluster
                model = self._load_model(model_path)

                test_dataset_this_cluster = sub_dataset_test.filter(lambda x: x['cluster_label'] == c)

                print(f"Evaluating client {self.cid} on the test set with size {len(test_dataset_this_cluster)} for cluster {c} in round {server_round}...")
                default_evaluation(
                    model=model,
                    tokenizer=self.tokenizer,
                    dataset=test_dataset_this_cluster,
                    client_id=self.cid,
                    round=server_round, #with respect to model from the previous round
                    formatting_prompts_func=self.formatting_prompts_func,
                    script_args=self.script_args,
                    cluster_id=c,
                )

            del test_embeddings
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()

        loss = 0.0
        metrics = {"eval_loss": loss}
        
        return loss, 100, metrics
    
    def _save_adapter_for_clustering(self, round_idx: int, trainer) -> None:

        output_dir = os.path.join(self.output_dir, "clients_adapters")
        os.makedirs(output_dir, exist_ok=True)
        
        adapter_path = os.path.join(output_dir, f"checkpoint-{round_idx}_client{self.cid}")
        
        # Save the adapter using trainer's save_model method
        trainer.save_model(adapter_path)
        
        print(f"Saved adapter for client {self.cid} at round {round_idx}")

    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the model from the specified path."""

        # Load the model with PEFT configuration
        model = get_model_flower(self.script_args, self.training_args, self.peft_config,
                                 self.device_map, self.quantization_config, self.torch_dtype)[0]
        
        # Load the adapter weights
        model.load_adapter(model_path, adapter_name="default")
        
        model.to(self.device)
        model.eval()
        
        return model

def create_client_fn(experiment_config):
   
    from flwr.common import Context
    
    def client_fn(context: Context) -> FedRouterClient:
        """Create a client instance with the specified configuration."""
        cid = int(context.node_config["partition-id"])
        
        # Create a copy of the model for this client to avoid state sharing
        model_copy = copy.deepcopy(experiment_config['model'])
        
        # Get client's local dataset
        local_dataset = experiment_config['local_datasets'][cid]
        local_dataset_test = experiment_config['local_datasets_test'][cid]
        
        client = FedRouterClient(
            cid=cid,
            model=model_copy,
            tokenizer=experiment_config['tokenizer'],
            peft_config=experiment_config['peft_config'],
            device_map=experiment_config['device_map'],
            quantization_config=experiment_config['quantization_config'],
            torch_dtype=experiment_config['torch_dtype'],
            local_dataset=local_dataset,
            local_dataset_test=local_dataset_test,
            script_args=experiment_config['script_args'],
            fed_args=experiment_config['fed_args'],
            training_args=copy.deepcopy(experiment_config['training_args']),
            formatting_prompts_func=experiment_config['formatting_prompts_func'],
            data_collator=experiment_config['data_collator'],
            packing=experiment_config['packing'],
            device=experiment_config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            output_dir=experiment_config['script_args'].output_dir,
        )
        
        return client
    
    return client_fn
