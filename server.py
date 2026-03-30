import os
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional

from federated_learning.router_utils import get_most_similar_adapter
from flwr.server.strategy import Strategy, FedAvg
from flwr.common import (
    Parameters, 
    FitRes, 
    FitIns, 
    EvaluateRes,
    EvaluateIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

# Import project utilities for clustering
from federated_learning.fed_clustered import calculate_similarity, make_clusters
from peft import get_peft_model_state_dict
from safetensors.torch import save_file

from utils.utils import default_evaluation, save_dataset_test
from federated_learning import *
from federated_learning.router_utils import *


class FedRouterStrategy(FedAvg):
    
    def __init__(self, 
                 global_model,
                 initial_parameters: Parameters,
                 peft_state_dict,
                 num_clients: int,
                 sim_round: int,
                 n_clusters: int,
                 output_dir: str,
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 script_args=None,
                 fed_args=None, 
                 quantization_config=None):

        self.initial_parameters = initial_parameters
        self.num_clients = num_clients
        self.sim_round = sim_round
        self.n_clusters = n_clusters
        self.output_dir = output_dir
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.script_args = script_args
        self.fed_args = fed_args
        
        self.current_round = 0
        self.client_clusters = {}  
        self.cluster_models = {}  
        self.global_model_structure = global_model
        self.global_model = initial_parameters
        self.cluster_assignments = None  
        self.peft_state_dict = peft_state_dict
        self.quantization_config = quantization_config
        
        self.training_metrics = []

        print(f"Initialized FedRouterStrategy with {num_clients} clients, sim_round={sim_round}, n_clusters={n_clusters}")
        
        # Save initial global model
        self._save_initial_global_model()
        
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.initial_parameters
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[Any, FitIns]]:

        self.current_round = server_round
        
        clients = client_manager.sample(num_clients=self.num_clients, min_num_clients=self.min_available_clients)

        if server_round == 1:
            self.global_dict = parameters_to_ndarrays(parameters)
        
        # Prepare configuration for clients
        config = {
            "current_round": server_round,
            "total_rounds": self.fed_args.num_rounds if self.fed_args else 100,
        }
        
        # Determine which parameters to send to each client
        fit_configurations = []

        cluster_id = server_round % self.fed_args.n_clusters
        config["cluster_id"] = cluster_id

        print(f"Round  - {server_round}")
        if server_round == 1:
            for client in clients:
                fit_configurations.append((client, FitIns(parameters, config)))
        else:
            for client in clients:
                global_cluster_idx_to_send = get_most_similar_adapter(self.global_centroids, self.global_clusters, self.client_embeddings_centers[client.cid][cluster_id])
                params_to_client = self.global_dict[global_cluster_idx_to_send]
                fit_configurations.append((client, FitIns(params_to_client, config)))

        print('Initialized fit configurations for clients.')
        print(config)
        return fit_configurations
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[Any, EvaluateIns]]:
        #return server round and global centroids
        clients = client_manager.sample(num_clients=self.num_clients, min_num_clients=self.min_available_clients)
        evaluate_configurations = []

        config = {
            "current_round": server_round,
            "total_rounds": self.fed_args.num_rounds if self.fed_args else 100,
            "global_centroids": str(self.global_centroids.tolist()),
            "global_clusters": str(self.global_clusters.tolist())
        }
        print(config)
        for client in clients:
            evaluate_configurations.append((client, EvaluateIns(self.global_dict[0], config)))
        print('Initialized evaluate configurations for clients.')

        return evaluate_configurations
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, FitRes]], failures: List[Any]) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        
        self.client_embeddings_centers = {}
        for client, fit_res in results:
            self.client_embeddings_centers[client.cid] = eval(fit_res.metrics.get('client_embedding'))


        if server_round == 1:
            client_embeddings_centers_list = []
            for client, fit_res in results:
                for center in eval(fit_res.metrics.get('client_embedding')):
                    client_embeddings_centers_list.append(center)

            self.global_centroids, self.global_clusters = cluster_clients_centroids(client_embeddings_centers_list, num_clusters = self.fed_args.global_n_clusters)
            #save global centroids
            print(f'Initial clustering with {self.fed_args.global_n_clusters} clusters')
            np.save(os.path.join(self.output_dir, "global_centroids.npy"), self.global_centroids)
            print(f"Initial global centroids: {self.global_centroids}")
            np.save(os.path.join(self.output_dir, "global_clusters.npy"), self.global_clusters)
            print(f"Initial global clusters: {self.global_clusters}")


        global_clusters_this_round = {}
        local_dict_list = []
        for client, fit_res in results:
            global_clusters_this_round[client.cid] = get_most_similar_adapter(self.global_centroids, self.global_clusters, eval(fit_res.metrics.get('client_embedding'))[fit_res.metrics.get('cluster_id')])
            local_dict_list.append(parameters_to_ndarrays(fit_res.parameters))

        n_clusters = self.fed_args.global_n_clusters
        idx = global_clusters_this_round
        print('Global cluster assignments: ', self.global_clusters)
        print(f"Client to global cluster assignments: {idx}")
        # Separate models into clusters -------------------------------------
        # Group results by cluster
        cluster_results = {}
        for i, (client, fit_res) in enumerate(results):
            for cluster_id in range(n_clusters):

                if cluster_id not in cluster_results:
                    cluster_results[cluster_id] = []

                if idx[client.cid] == cluster_id:
                    cluster_results[cluster_id].append((client, fit_res))
        
        # Aggregate within each cluster
        updated_cluster_models = {}
        cluster_metrics = {}
        
        for cluster_id in range(n_clusters):
            cluster_res = cluster_results.get(cluster_id, [])
            
            if cluster_res:  # If there are clients in this cluster
                # Perform FedAvg within the cluster
                weights_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in cluster_res]
                sample_sizes = [fit_res.num_examples for _, fit_res in cluster_res]
                
                total_samples = sum(sample_sizes)
                aggregated_weights = []
                
                for i in range(len(weights_list[0])):
                    weighted_sum = sum(weights[i] * size for weights, size in zip(weights_list, sample_sizes))
                    aggregated_weights.append(weighted_sum / total_samples)
                
                updated_cluster_models[cluster_id] = ndarrays_to_parameters(aggregated_weights)
                #updated_cluster_models[cluster_id] = aggregated_weights
                cluster_metrics[cluster_id] = {
                    "num_clients": len(cluster_res),
                    "total_samples": total_samples
                }
                
                print(f"Cluster {cluster_id}: {len(cluster_res)} clients, {total_samples} samples")
            else:
                updated_cluster_models[cluster_id] = self.cluster_models.get(cluster_id, self.global_model)
                cluster_metrics[cluster_id] = {"num_clients": 0, "total_samples": 0}
                print(f"Cluster {cluster_id}: no clients, keeping previous model")
        
        self.global_dict = updated_cluster_models
        self.cluster_models = updated_cluster_models

        self._save_cluster_models(server_round)

        return self.global_dict, cluster_metrics 

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[Any, EvaluateRes]], failures: List[Any]) -> Tuple[Optional[float], Dict[str, Any]]:

        
        print(f"Round {server_round} evaluation - COMPLETED")
        
        return 0, {}
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Any]]]:
        #without global evaluation
        return None
    
    def _save_initial_global_model(self) -> None:
        """Save the initial global model."""
        os.makedirs(os.path.join(self.output_dir, "global_model"), exist_ok=True)
        self._save_global_model(0)
        
    def _save_global_model(self, round_idx: int, peft_state_dict=None) -> None:
        """Save the global model for a specific round."""
        # Create directory for the round
        save_dir = os.path.join(self.output_dir, f"checkpoint-{round_idx}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Get parameters and reconstruct state dict with original keys
        params = parameters_to_ndarrays(self.global_model)
        if peft_state_dict is None:
            peft_state_dict = self.peft_state_dict
            
        # Recreate state dict with previous keys and updated parameters
        updated_state_dict = {}
        for i, (key, _) in enumerate(peft_state_dict.items()):
            updated_state_dict[key] = torch.tensor(params[i])
        
        # Save model using safetensors format
        save_file(updated_state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
        print(f"Saved global model for round {round_idx}")
        
    def _save_cluster_models(self, round_idx: int, cluster_peft_state_dicts=None) -> None:
        """Save all cluster models for a specific round."""
        base_dir = os.path.join(self.output_dir, "cluster_models", f"round_{round_idx}")
        os.makedirs(base_dir, exist_ok=True)
        
        # Save each cluster model
        for cluster_id, cluster_params in self.cluster_models.items():
            cluster_dir = os.path.join(base_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Get parameters for this cluster
            params = parameters_to_ndarrays(cluster_params)
            
            # Recreate state dict with previous keys and updated parameters
            updated_state_dict = {}
            for i, (key, _) in enumerate(self.peft_state_dict.items()):
                updated_state_dict[key] = torch.tensor(params[i])
            
            # Save model using safetensors format
            save_file(updated_state_dict, os.path.join(cluster_dir, "adapter_model.safetensors"))
            print(f"Saved model for cluster {cluster_id}, round {round_idx}")
            
    def _save_parameters_to_path(self, param_arrays: List[np.ndarray], output_path: str) -> None:
        """Save model parameters to a specific path."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Recreate state dict with original keys and provided parameters
        updated_state_dict = {}
        for i, (key, _) in enumerate(self.peft_state_dict.items()):
            if i < len(param_arrays):
                updated_state_dict[key] = torch.tensor(param_arrays[i])
        
        # Save using safetensors format
        save_file(updated_state_dict, output_path)
        print(f"Saved parameters to {output_path}")
       

    def _load_client_adapters(self, round_idx: int) -> Dict[int, List[np.ndarray]]:
        """
        Load client adapters from the output directory.
        
        Args:
            round_idx: Round to load client adapters from
            
        Returns:
            Dictionary mapping client_id to parameter arrays
        """ 
        client_adapters = {}
        adapter_dir = os.path.join(self.output_dir, "client_adapters", f"round_{round_idx}")
        
        if not os.path.exists(adapter_dir):
            print(f"No client adapters found for round {round_idx}")
            return client_adapters
        
        for client_folder in os.listdir(adapter_dir):
            if client_folder.startswith("client_"):
                try:
                    client_id = int(client_folder.split("_")[1])
                    client_path = os.path.join(adapter_dir, client_folder)
                    
                    # Try to load the adapter parameters
                    param_file = os.path.join(client_path, "model_parameters.npz")
                    if os.path.exists(param_file):
                        loaded = np.load(param_file)
                        param_arrays = [loaded[f"arr_{i}"] for i in range(len(loaded.files))]
                        client_adapters[client_id] = param_arrays
                        print(f"Loaded adapter for client {client_id}")
                    else:
                        # Fallback: try to load pytorch model
                        pytorch_file = os.path.join(client_path, "pytorch_model.bin")
                        if os.path.exists(pytorch_file):
                            state_dict = torch.load(pytorch_file, map_location="cpu")
                            param_arrays = [tensor.numpy() for tensor in state_dict.values()]
                            client_adapters[client_id] = param_arrays
                            print(f"Loaded adapter for client {client_id} from pytorch file")
                        else:
                            print(f"No parameter file found for client {client_id}")
                            
                except Exception as e:
                    print(f"Failed to load adapter for {client_folder}: {e}")
        
        return client_adapters


def create_server_strategy(experiment_config: Dict[str, Any]) -> FedRouterStrategy :

    initial_state_dict = get_peft_model_state_dict(experiment_config['model'])
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_state_dict.values()]
    )

    peft_state_dict = get_peft_model_state_dict(experiment_config['model'])
    
    # Create strategy with clustering configuration
    strategy = FedRouterStrategy(
        global_model = experiment_config['model'],
        initial_parameters=initial_parameters,
        peft_state_dict=peft_state_dict,
        num_clients=experiment_config['fed_args'].num_clients,
        sim_round=experiment_config['fed_args'].sim_round,
        n_clusters=experiment_config['fed_args'].n_clusters,
        output_dir=experiment_config['script_args'].output_dir,
        fraction_fit=1.0,  # Use all clients (can be adjusted)
        fraction_evaluate=1.0,
        min_fit_clients=experiment_config['fed_args'].num_clients,
        min_evaluate_clients=experiment_config['fed_args'].num_clients,
        min_available_clients=experiment_config['fed_args'].num_clients,
        script_args=experiment_config['script_args'],
        fed_args=experiment_config['fed_args'],
        quantization_config=experiment_config['quantization_config']
    )
    
    return strategy
