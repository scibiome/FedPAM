from FeatureCloud.app.engine.app import AppState, Role, State, app_state
import time
import functools
import os
import warnings
import traceback
import json
import sys
import itertools
import yaml
import numpy as np
import pandas as pd
import statistics
import networkx as nx
from pgmpy.utils import get_example_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
from collections import Counter

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='pgmpy')
warnings.filterwarnings('ignore', message='.*Replacing existing CPD.*')
warnings.filterwarnings('ignore', message='.*pgmpy.*')

import client
import server
from store import store

INITIAL = 'initial'
FETCH_DATA = 'read config and dataset'
LOCAL_LEARNING = 'local learning'
AGGREGATION = 'aggregation'
AWAIT_AGGREGATION = 'await aggregation'
LOCAL_REFINEMENT = 'local refinement'
FINAL = 'final'
PARAMETER_LEARNING = 'parameter learning'
CATEGORY_LEVELS = 'gather category levels'
AWAIT_CATEGORY_LEVELS = 'await category levels'
PARAMETER_AGGREGATION = 'aggregate parameters'
AWAIT_PARAMETERS_AGGREGATION = 'await parameter aggregation'
EVALUATION = 'evaluation'
VISUALIZE = 'visualize'
TERMINAL = 'terminal'

FINISH_SIGNAL = 'finish'
FINISH_MEMO = 'FEDPAM_FINISH'
EVAL_MEMO = 'FEDPAM_EVAL'


def park_on_error(run_method):
    """Never let a failing state tear the container down.

    FeatureCloud's engine catches any exception escaping a state, marks the run
    as ERROR and finishes immediately -- which kills the container and takes the
    dashboard with it. That is why a crash here looks like "the workflow ended
    right after evaluation and the UI stopped responding".

    Wrapped states instead publish the traceback to the dashboard and divert to
    VISUALIZE, so the results that did compute stay readable and the run still
    ends only when the coordinator says so.
    """
    @functools.wraps(run_method)
    def wrapper(self):
        try:
            return run_method(self)
        except Exception:
            tb = traceback.format_exc()
            self.log(f"[ERROR] state failed, diverting to {VISUALIZE} so the "
                     f"dashboard stays up:\n{tb}")
            store.update(error=tb)
            return VISUALIZE
    return wrapper

@app_state(name = INITIAL, role = Role.BOTH)
class InitialState(AppState):
    """
    InitialState is the class is used for initializing the FedPAM workflow. 
    In this state, each client takes up the role of a participant.
    In the next state, dataset and configuration files are read.
    """
    def register(self):
        self.register_transition(target=FETCH_DATA, role=Role.BOTH)
    
    def run(self):
        self.log("Initializing FedPAM App...")
        # The UI needs the role to decide whether to offer the Finish button.
        store.update(is_coordinator=self.is_coordinator, current_state=INITIAL)
        self.log(f"Role: {'coordinator' if self.is_coordinator else 'participant'}")
        self.log("Initial State to Fetch Data State")
        return FETCH_DATA

@app_state(name = FETCH_DATA, role = Role.BOTH)
class FetchDataState(AppState):
    """
    FetchDataState is the class used for reading private dataset, configuration and expert knowledge files local storage. 
    In this state, each participant sends its dataset size and expert knowledge file to the coordinator.
    In the next state, the coordinator prepares aggregated metadata while the participants wait for the coordinator to finish. 
    """
    def register(self):
        self.register_transition(target=LOCAL_LEARNING, role=Role.BOTH)
    
    def read_config_file(self, input_dir):
        self.log("Reading config file...")
        config_file_path = os.path.join(input_dir, 'config.yml')
        
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found at {config_file_path}.")
        
        with open(config_file_path) as cfp:
            config_file = yaml.safe_load(cfp)

        configs = config_file['fc-fedpam']
        self.store('dataset_location', configs['input']['dataset_location'])
        self.store('has_target', configs['input']['has_target'])
        store.update(has_target=bool(configs['input']['has_target']))
        store.update(testing_enabled=bool(configs['testing']),
                     benchmark=configs.get('benchmark'))
        self.store('target', configs['input']['target'])
        store.update(target=configs['input']['target']
                     if configs['input']['has_target'] else None)
        self.store('split_mode', configs['split']['mode'])
        self.store('split_dir', configs['split']['dir'])
        
        # Load hyperparameters
        self.store('max_iterations', configs['max_iterations'])
        self.store('num_bootstrap_iterations', configs['num_bootstrap_iterations'])
        self.store('alpha', configs['alpha'])
        self.store('gamma', configs['gamma']) 
        self.store('homogeneous', configs['homogeneous']) 
        self.store('testing', configs['testing'])
        self.store('benchmark', configs['benchmark'])
        self.store('threshold', configs['threshold'])
        self.store('num_samples', configs['num_samples'])
        self.store('num_jobs', configs['num_jobs'])
        self.store('n_splits', configs.get('n_splits', 5))

        splits = {}
        if self.load('split_mode') == 'directory':
            split_base_dir = os.path.join(input_dir, self.load('split_dir'))
            if os.path.exists(split_base_dir):
                splits = {f.path: None for f in os.scandir(split_base_dir) if f.is_dir()}
            else:
                splits = {input_dir: None}
        else:
            splits = {input_dir: None}

        roles = {}
        for split_path in splits.keys():
            output_path = split_path.replace('/input/', '/output/')
            os.makedirs(output_path, exist_ok=True)

        self.log("Configuration loaded successfully!")
        
        return splits, roles
    
    def read_dataset(self, splits, roles):
        for split_path in splits.keys():
            roles[split_path] = 'coordinator' if self.is_coordinator else 'client'
            dataset_location = self.load('dataset_location')
            has_target = self.load('has_target')

            dataset_path = os.path.join(split_path, dataset_location)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at location: {dataset_path}.")
            
            self.log("Reading dataset...")
            testing = self.load('testing')
            if testing:
                num_samples = self.load('num_samples')

            dataset = pd.read_csv(dataset_path)
            # randomly shuffle the dataset to prevent sorted target values
            if testing and num_samples:
                dataset = dataset.sample(n = num_samples, random_state=23).reset_index(drop=True)
            else:
                dataset = dataset.sample(frac=1, random_state=23).reset_index(drop=True)

            dataset = dataset.reset_index(drop=True)

            splits[split_path] = dataset
            self.log(
                f"Local dataset from {split_path}: {dataset.shape[0]} observations "
                f"and {dataset.shape[1]} variables (no hold-out split — evaluation uses k-fold CV)."
            )

        client_split_path = None
        client_id = str(self.id).lower()
        for split_path in splits.keys():
            split_dirname = os.path.basename(split_path).lower()
            self.log(f"Comparing client ID '{client_id}' with split directory '{split_dirname}'...")
            if client_id in split_dirname:
                client_split_path = split_path
                break

        if client_split_path is None:
            if len(splits) == 1:
                client_split_path = next(iter(splits.keys()))
                self.log(f"Using split directory: {client_split_path}.")
            else:
                raise RuntimeError(f"No matching split directory for client ID {client_id}.")

        self.store('dataset', splits[client_split_path])
        self.store('client_split_path', client_split_path)

        # Hand the local dataset to the Dash UI.
        store.update(dataset=splits[client_split_path],
                     dataset_path=os.path.join(client_split_path,
                                               self.load('dataset_location')),
                     client_id=self.id)
        self.log(f"[viz] dataset published to UI: {splits[client_split_path].shape}")

        return splits, roles
    
    def run(self):
        iteration = 1
        self.store('iteration', iteration)

        input_dir = "/mnt/input"
        output_dir = "/mnt/output"
        self.store('input_dir', input_dir)
        self.store('output_dir', output_dir)
        
        splits_init, roles_init = self.read_config_file(input_dir)
        splits, roles = self.read_dataset(splits_init, roles_init)
        self.store('splits', splits)
        self.store('roles', roles)

        local_number = np.random.randint(1, 10, 1)
        self.log(f"LOCAL NUMBER: {local_number}")

        self.log("Fetch data state to local learning")
        return LOCAL_LEARNING


@app_state(name = LOCAL_LEARNING, role = Role.BOTH)
class LocalLearningState(AppState):
    def register(self):
        self.register_transition(target = AWAIT_AGGREGATION, role = Role.PARTICIPANT)
        self.register_transition(target = AGGREGATION, role = Role.COORDINATOR)

    def run(self):
        dataset = self.load('dataset')
        dataset_size = len(dataset)
        testing_flag = self.load('testing')
        num_bootstrap_iterations = self.load('num_bootstrap_iterations')
        benchmark = self.load('benchmark')
        num_jobs = self.load('num_jobs')
        has_target = self.load('has_target')

        participant = client.Client()

        if has_target:
            target = self.load('target')
            local_edge_strengths, allowed_edges = participant.create_pam(dataset = dataset, has_target = has_target, target = target, num_iterations = num_bootstrap_iterations, seed = 23, n_jobs=num_jobs)
            local_dag = participant.learn_constrained_local_dag(dataset = dataset, allowed_edges = allowed_edges, has_target = has_target, target = target)
        else:
            local_edge_strengths, allowed_edges = participant.create_pam(dataset = dataset, has_target = False, target = None, num_iterations = num_bootstrap_iterations, seed = 23, n_jobs=num_jobs)
            local_dag = participant.learn_constrained_local_dag(dataset = dataset, allowed_edges = allowed_edges)

        self.store('local_pam', local_edge_strengths)
        self.store('local_dag', local_dag)
        self.store('first_local_dag', local_dag)  

        store.update(local_structure=sorted(local_dag.edges()),
                     first_local_structure=sorted(local_dag.edges()))

        if testing_flag:
            benchmark_network = get_example_model(benchmark)
            true_edges = set(benchmark_network.edges())
            all_nodes = set(benchmark_network.nodes())

            def compute_structure_metrics(network, edge_strengths, true_edges, all_nodes, label):
                """
                Compute structure learning metrics: SHD, Precision, Recall (TPR), 
                Number of edges, AUROC, and AUPR.
                
                Args:
                    network: Predicted DAG (networkx DiGraph)
                    edge_strengths: Dictionary of edge -> strength scores (for AUROC/AUPR)
                    true_edges: Set of true edges from benchmark
                    all_nodes: Set of all nodes in the network
                    label: String identifier for logging
                """
                pred_edges = set(network.edges())
                
                # Basic edge metrics
                tp = len(pred_edges & true_edges)
                fp = len(pred_edges - true_edges)
                fn = len(true_edges - pred_edges)
                
                precision = tp / len(pred_edges) if pred_edges else 0
                recall = tp / len(true_edges) if true_edges else 0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
                num_edges = network.number_of_edges()
                
                # SHD
                shd = participant.compute_shd(benchmark_network, network)
                
                # AUROC and AUPR computation
                # Create binary labels and scores for all possible edges
                edge_scores = []
                edge_labels = []
                
                if edge_strengths is not None and len(edge_strengths) > 0:
                    # Generate all possible edges (excluding self-loops)
                    for u in all_nodes:
                        for v in all_nodes:
                            if u != v:
                                edge = (u, v)
                                # Get score from edge_strengths, default to 0 if not present
                                score = edge_strengths.get(edge, 0.0)
                                edge_scores.append(score)
                                # True label: 1 if edge is in true_edges, 0 otherwise
                                edge_labels.append(1 if edge in true_edges else 0)
                    
                    edge_scores = np.array(edge_scores)
                    edge_labels = np.array(edge_labels)
                    
                    try:
                        auroc = roc_auc_score(edge_labels, edge_scores)
                    except Exception as e:
                        auroc = np.nan
                        self.log(f"Warning: Could not compute AUROC for {label}: {e}")
                    
                    try:
                        aupr = average_precision_score(edge_labels, edge_scores)
                    except Exception as e:
                        aupr = np.nan
                        self.log(f"Warning: Could not compute AUPR for {label}: {e}")
                else:
                    auroc = np.nan
                    aupr = np.nan
                
                # Logging
                self.log(f"[TESTING] --- {label} ---")
                self.log(f"Number of edges: {num_edges}")
                self.log(f"SHD: {shd}")
                self.log(f"TP: {tp}, FP: {fp}, FN: {fn}")
                self.log(f"Precision: {precision:.4f}")
                self.log(f"Recall (TPR): {recall:.4f}")
                self.log(f"F1-score: {f1:.4f}")
                self.log(f"AUROC: {auroc:.4f}" if not np.isnan(auroc) else f"AUROC: NaN")
                self.log(f"AUPR: {aupr:.4f}" if not np.isnan(aupr) else f"AUPR: NaN")
                
                return {
                    'num_edges': num_edges,
                    'shd': shd,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auroc': auroc,
                    'aupr': aupr
                }

            # ---------------- SINGLE ITERATION ----------------
            single_iter_network = participant.learn_local_structure(
                dataset, False, None, True
            )
            single_metrics = compute_structure_metrics(
                single_iter_network, local_edge_strengths, true_edges, all_nodes,
                "SINGLE ITERATION"
            )

            # ---------------- BOOTSTRAP NETWORK ----------------
            bootstrap_metrics = compute_structure_metrics(
                local_dag, local_edge_strengths, true_edges, all_nodes,
                "BOOTSTRAP NETWORK"
            )

            # Store metrics for later use
            self.store('testing_metrics_single', single_metrics)
            self.store('testing_metrics_bootstrap', bootstrap_metrics)

            # ---------------- COMPARISON ----------------
            self.log("[TESTING] --- COMPARISON ---")
            self.log(f"ΔSHD: {bootstrap_metrics['shd'] - single_metrics['shd']:+d}")
            self.log(f"ΔPrecision: {bootstrap_metrics['precision'] - single_metrics['precision']:+.4f}")
            self.log(f"ΔRecall: {bootstrap_metrics['recall'] - single_metrics['recall']:+.4f}")
            self.log(f"ΔF1-score: {bootstrap_metrics['f1'] - single_metrics['f1']:+.4f}")
            self.log(f"ΔAUROC: {bootstrap_metrics['auroc'] - single_metrics['auroc']:+.4f}" if not (np.isnan(bootstrap_metrics['auroc']) or np.isnan(single_metrics['auroc'])) else "ΔAUROC: NaN")
            self.log(f"ΔAUPR: {bootstrap_metrics['aupr'] - single_metrics['aupr']:+.4f}" if not (np.isnan(bootstrap_metrics['aupr']) or np.isnan(single_metrics['aupr'])) else "ΔAUPR: NaN")

        if testing_flag:
            bootstrap_metrics = self.load('testing_metrics_bootstrap')
            local_payload = {
                "client_data_size": dataset_size,
                "client_pam": local_edge_strengths,
                "num_local_edges": bootstrap_metrics['num_edges'],
                "local_shd": bootstrap_metrics['shd'],
                "local_tpr": bootstrap_metrics['recall'],
                "local_precision": bootstrap_metrics['precision'],
                "local_auroc": bootstrap_metrics['auroc'],
                "local_aupr": bootstrap_metrics['aupr']
            }
        else:
            local_payload = {
                "client_data_size": dataset_size,
                "client_pam": local_edge_strengths,
                "num_local_edges": local_dag.number_of_edges()
            }

        self.send_data_to_coordinator(local_payload)

        if self.is_coordinator:
            return AGGREGATION
        else:
            return AWAIT_AGGREGATION


@app_state(name = AGGREGATION, role = Role.COORDINATOR)
class AggregationState(AppState):
    def register(self):
        self.register_transition(target = LOCAL_REFINEMENT, role = Role.COORDINATOR)
        self.register_transition(target = FINAL, role = Role.COORDINATOR)

    def run(self):
        iteration = self.load('iteration')
        self.log(f"ITERATION: {iteration}")
        testing = self.load('testing')
        max_iterations = self.load('max_iterations')
        patience = 5

        client_payloads = self.gather_data()
        client_data_sizes = [cp["client_data_size"] for cp in client_payloads]
        client_weights = [cds / sum(client_data_sizes) for cds in client_data_sizes]
        client_pams = [cp["client_pam"] for cp in client_payloads]

        if testing and iteration == 1:
            client_num_edges = [cp["num_local_edges"] for cp in client_payloads]
            client_shds = [cp["local_shd"] for cp in client_payloads]
            client_tprs = [cp["local_tpr"] for cp in client_payloads]
            client_precisions = [cp.get("local_precision", np.nan) for cp in client_payloads]
            client_aurocs = [cp.get("local_auroc", np.nan) for cp in client_payloads]
            client_auprs = [cp.get("local_aupr", np.nan) for cp in client_payloads]

            mean_client_num_edges = statistics.mean(client_num_edges)
            mean_client_shd = statistics.mean(client_shds)
            mean_client_tpr = statistics.mean(client_tprs)
            mean_client_precision = np.nanmean(client_precisions) if any(~np.isnan(client_precisions)) else np.nan
            mean_client_auroc = np.nanmean(client_aurocs) if any(~np.isnan(client_aurocs)) else np.nan
            mean_client_aupr = np.nanmean(client_auprs) if any(~np.isnan(client_auprs)) else np.nan

            std_client_num_edges = statistics.stdev(client_num_edges) if len(client_num_edges) > 1 else 0
            std_client_shd = statistics.stdev(client_shds) if len(client_shds) > 1 else 0
            std_client_tpr = statistics.stdev(client_tprs) if len(client_tprs) > 1 else 0
            std_client_precision = np.nanstd(client_precisions) if any(~np.isnan(client_precisions)) else 0
            std_client_auroc = np.nanstd(client_aurocs) if any(~np.isnan(client_aurocs)) else 0
            std_client_aupr = np.nanstd(client_auprs) if any(~np.isnan(client_auprs)) else 0

            self.log(f"[TESTING] Client local metrics")
            self.log(f"Number of local edges: {mean_client_num_edges:.2f} ± {std_client_num_edges:.2f}")
            self.log(f"Local SHD: {mean_client_shd:.2f} ± {std_client_shd:.2f}")
            self.log(f"Local Precision: {mean_client_precision:.4f} ± {std_client_precision:.4f}" if not np.isnan(mean_client_precision) else f"Local Precision: NaN")
            self.log(f"Local TPR (Recall): {mean_client_tpr:.4f} ± {std_client_tpr:.4f}")
            self.log(f"Local AUROC: {mean_client_auroc:.4f} ± {std_client_auroc:.4f}" if not np.isnan(mean_client_auroc) else f"Local AUROC: NaN")
            self.log(f"Local AUPR: {mean_client_aupr:.4f} ± {std_client_aupr:.4f}" if not np.isnan(mean_client_aupr) else f"Local AUPR: NaN")

        self.log(f"[COORDINATOR]: Client weights -> {client_weights}")

        coordinator = server.Server()
        global_pam = coordinator.aggregate_pams(client_pams, client_weights)
        self.log(f"[COORDINATOR]: GLOBAL PAM: {global_pam}")

        self.store('global_pam', global_pam)

        prev_global_pam = self.load('prev_global_pam')
        patience_counter = self.load('patience_counter') or 0

        if prev_global_pam is not None and coordinator.pams_equal(global_pam, prev_global_pam):
            patience_counter += 1
        else:
            patience_counter = 0

        self.store('prev_global_pam', global_pam)
        self.store('patience_counter', patience_counter)

        self.log(f"[COORDINATOR]: Patience counter -> {patience_counter}/{patience}")

        stagnated = patience_counter >= patience
        capped = iteration >= max_iterations

        if not stagnated and not capped:
            iteration += 1
            self.store('iteration', iteration)
            message = "continue"
            coordinator_payload = {
                "message": message,
                "global_pam": global_pam
            }

            self.broadcast_data(coordinator_payload)
            return LOCAL_REFINEMENT
        else:
            if stagnated:
                self.log(f"[COORDINATOR]: Stopped — global PAM unchanged for {patience} iterations (round {iteration}).")
            else:
                self.log(f"[COORDINATOR]: Stopped at max_iterations ({iteration}) without stagnation.")
            message = "stop"
            coordinator_payload = {
                "message": message,
                "global_pam": global_pam
            }

            self.broadcast_data(coordinator_payload)
            return FINAL


@app_state(name = AWAIT_AGGREGATION, role = Role.PARTICIPANT)
class AwaitAggregationState(AppState):
    def register(self):
        self.register_transition(target = LOCAL_REFINEMENT, role = Role.PARTICIPANT)
        self.register_transition(target = FINAL, role = Role.PARTICIPANT)

    def run(self):
        coordinator_payload = self.await_data() 
        message = coordinator_payload["message"]
        global_pam = coordinator_payload["global_pam"]
        self.store('global_pam', global_pam)

        if message == "continue":
            return LOCAL_REFINEMENT
        else:
            return FINAL


@app_state(name = LOCAL_REFINEMENT, role = Role.BOTH)
class LocalRefinementState(AppState):
    def register(self):
        self.register_transition(target = AWAIT_AGGREGATION, role = Role.PARTICIPANT)
        self.register_transition(target = AGGREGATION, role = Role.COORDINATOR)
    
    def run(self):
        self.log("DO LOCAL REFINEMENT")
        dataset = self.load('dataset')
        dataset_size = len(dataset)
        local_pam_prev = self.load('local_pam')
        local_dag = self.load('local_dag')
        global_pam = self.load('global_pam')
        alpha = self.load('alpha')
        gamma = self.load('gamma')

        has_target = self.load('has_target')
        target = self.load('target')

        participant = client.Client()
        local_pam, _ = participant.refine_local_pam(local_pam_prev, global_pam, local_dag, alpha, gamma)
        if has_target and target:
            local_pam = {edge: strength for edge, strength in local_pam.items()
                         if edge[0] != target}

        allowed_edges = list(local_pam.keys())
        local_dag = participant.learn_constrained_local_dag(
            dataset=dataset, allowed_edges=allowed_edges,
            has_target=has_target, target=target)
        self.store('local_dag', local_dag)
        self.store('local_pam', local_pam)
        store.update(local_structure=sorted(local_dag.edges()),
                     iteration=self.load('iteration'))
        local_payload = {
            "client_data_size": dataset_size,
            "client_pam": local_pam
        }

        self.send_data_to_coordinator(local_payload)
        if self.is_coordinator:
            return AGGREGATION
        else: 
            return AWAIT_AGGREGATION


@app_state(name = FINAL, role = Role.BOTH)
class FinalState(AppState):
    def register(self):
        self.register_transition(target = VISUALIZE, role = Role.BOTH)
        self.register_transition(target = CATEGORY_LEVELS, role = Role.COORDINATOR)
        self.register_transition(target = AWAIT_CATEGORY_LEVELS, role = Role.PARTICIPANT)

    def run(self):
        dataset = self.load('dataset')
        nodes = sorted(dataset.columns.tolist())
        global_pam = self.load('global_pam')
        testing_flag = self.load('testing')
        benchmark = self.load('benchmark')
        threshold = self.load('threshold')
        has_target = self.load('has_target')
        first_local_dag = self.load('first_local_dag')
        local_dag = self.load('local_dag')

        target = self.load('target')

        # Belt and braces: whatever reached the global PAM, the final DAG must
        # never contain an edge out of the target variable.
        if has_target and target:
            removed = [edge for edge in global_pam if edge[0] == target]
            if removed:
                self.log(f"[FINAL]: dropping {len(removed)} forbidden edges out "
                         f"of '{target}' from the global PAM: {removed}")
                global_pam = {edge: strength for edge, strength in global_pam.items()
                              if edge[0] != target}

        participant = client.Client()
        coordinator = server.Server()
        final_dag, is_valid = coordinator.finalize_dag(global_pam, nodes, threshold=threshold)

        self.log(f"[FINAL]: Final DAG edges -> {list(final_dag.edges())}")
        self.log(f"[FINAL]: Valid DAG -> {is_valid}")
        self.log(f"No. of edges: {final_dag.number_of_edges()}")

        self.store('final_dag', final_dag)
        store.update(global_structure=sorted(final_dag.edges()))
        self.store('final_dag_valid', is_valid)

        if testing_flag: 
            benchmark_network = get_example_model(benchmark)
            true_edges = set(benchmark_network.edges())
            all_nodes = set(benchmark_network.nodes())
            
            # Get the edge strengths for AUROC/AUPR computation
            local_pam = self.load('local_pam')
            global_pam = self.load('global_pam')

            def compute_complete_structure_metrics(network, edge_strengths, true_edges, all_nodes, label):
                """
                Compute complete structure learning metrics for final DAGs.
                Returns all metrics: SHD, Precision, Recall, # Edges, AUROC, AUPR.
                """
                pred_edges = set(network.edges())
                
                # Basic edge metrics
                tp = len(pred_edges & true_edges)
                fp = len(pred_edges - true_edges)
                fn = len(true_edges - pred_edges)
                
                precision = tp / len(pred_edges) if pred_edges else 0
                recall = tp / len(true_edges) if true_edges else 0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
                num_edges = network.number_of_edges()
                
                # SHD
                shd = participant.compute_shd(benchmark_network, network)
                
                # AUROC and AUPR computation
                edge_scores = []
                edge_labels = []
                
                if edge_strengths is not None and len(edge_strengths) > 0:
                    # Generate all possible edges (excluding self-loops)
                    for u in all_nodes:
                        for v in all_nodes:
                            if u != v:
                                edge = (u, v)
                                # Get score from edge_strengths, default to 0 if not present
                                score = edge_strengths.get(edge, 0.0)
                                edge_scores.append(score)
                                # True label: 1 if edge is in true_edges, 0 otherwise
                                edge_labels.append(1 if edge in true_edges else 0)
                    
                    edge_scores = np.array(edge_scores)
                    edge_labels = np.array(edge_labels)
                    
                    try:
                        auroc = roc_auc_score(edge_labels, edge_scores)
                    except Exception as e:
                        auroc = np.nan
                        self.log(f"Warning: Could not compute AUROC for {label}: {e}")
                    
                    try:
                        aupr = average_precision_score(edge_labels, edge_scores)
                    except Exception as e:
                        aupr = np.nan
                        self.log(f"Warning: Could not compute AUPR for {label}: {e}")
                else:
                    auroc = np.nan
                    aupr = np.nan
                
                # Logging
                self.log(f"[TESTING] --- {label} ---")
                self.log(f"Number of edges: {num_edges}")
                self.log(f"SHD: {shd}")
                self.log(f"TP: {tp}, FP: {fp}, FN: {fn}")
                self.log(f"Precision: {precision:.4f}")
                self.log(f"Recall (TPR): {recall:.4f}")
                self.log(f"F1-score: {f1:.4f}")
                self.log(f"AUROC: {auroc:.4f}" if not np.isnan(auroc) else f"AUROC: NaN")
                self.log(f"AUPR: {aupr:.4f}" if not np.isnan(aupr) else f"AUPR: NaN")
                
                return {
                    'num_edges': num_edges,
                    'shd': shd,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auroc': auroc,
                    'aupr': aupr
                }

            # Compute metrics for all three networks
            first_local_metrics = compute_complete_structure_metrics(
                first_local_dag, local_pam, true_edges, all_nodes,
                "FIRST LOCAL DAG"
            )
            
            final_local_metrics = compute_complete_structure_metrics(
                local_dag, local_pam, true_edges, all_nodes,
                "FINAL LOCAL DAG"
            )
            
            final_global_metrics = compute_complete_structure_metrics(
                final_dag, global_pam, true_edges, all_nodes,
                "FINAL GLOBAL DAG"
            )
            
            self.store('final_testing_metrics_first_local', first_local_metrics)
            self.store('final_testing_metrics_local', final_local_metrics)
            self.store('final_testing_metrics_global', final_global_metrics)
            store.update(structure_metrics={
                "Initial local network": first_local_metrics,
                "Final local network": final_local_metrics,
                "Final global network": final_global_metrics,
            })
            self.log("[FINAL] structure-recovery metrics published to the UI.")


        if has_target:
            participant = client.Client()
            local_levels = participant.get_local_category_levels(dataset)
            self.send_data_to_coordinator(local_levels)

            if self.is_coordinator:
                return CATEGORY_LEVELS
            else:
                return AWAIT_CATEGORY_LEVELS
        else:
            self.log("No target column: skipping parameter learning, "
                     "going straight to the waiting state.")
            return VISUALIZE


@app_state(name = CATEGORY_LEVELS, role = Role.COORDINATOR)
class CategoryLevelsAggregationState(AppState):
    def register(self):
        self.register_transition(target = PARAMETER_LEARNING, role = Role.COORDINATOR)

    def run(self):
        client_levels_list = self.gather_data()
        coordinator = server.Server()
        category_levels = coordinator.merge_category_levels(client_levels_list)

        self.store('category_levels', category_levels)
        self.broadcast_data(category_levels)
        self.log(f"Merged global category levels for {len(category_levels)} columns.")
        return PARAMETER_LEARNING


@app_state(name = AWAIT_CATEGORY_LEVELS, role = Role.PARTICIPANT)
class AwaitCategoryLevelsState(AppState):
    def register(self):
        self.register_transition(target = PARAMETER_LEARNING, role = Role.PARTICIPANT)

    def run(self):
        category_levels = self.await_data()
        self.store('category_levels', category_levels)
        self.log(f"Received global category levels for {len(category_levels)} columns.")
        return PARAMETER_LEARNING


@app_state(name = PARAMETER_LEARNING, role = Role.BOTH)
class ParameterLearningState(AppState):
    def register(self):
        self.register_transition(target = PARAMETER_AGGREGATION, role = Role.COORDINATOR)
        self.register_transition(target = AWAIT_PARAMETERS_AGGREGATION, role = Role.PARTICIPANT)

    def run(self):
        train_dataset = self.load('dataset')
        final_dag = self.load('final_dag')
        category_levels = self.load('category_levels')
        if not category_levels:
            raise RuntimeError(
                "category_levels is missing at PARAMETER_LEARNING — this client "
                "is likely running a stale build that predates the CATEGORY_LEVELS "
                "round. Rebuild/redeploy this client's image before retrying."
            )
        self.log(f"Using {len(category_levels)} globally-agreed column category sets.")
        train_dataset = train_dataset.astype('str').astype('category')

        participant = client.Client()
        final_dag_edges = participant.create_network_dict(final_dag.edges(), train_dataset.columns)
        local_params = participant.compute_beta_params_fixed(train_dataset, final_dag_edges, category_levels)

        node_order = participant.get_node_order(final_dag, train_dataset.columns)
        flat_vector, positions = participant.flatten_betas(local_params, node_order)

        self.log(f"Fitted local params for {len(local_params)} nodes.")
        self.store('local_params', local_params)
        self.store('node_order', node_order)
        self.store('local_positions', positions)

        local_param_payload = {
            "local_flat_vector": flat_vector,
            "client_data_size": len(train_dataset),
        }
        self.send_data_to_coordinator(local_param_payload)

        if self.is_coordinator:
            return PARAMETER_AGGREGATION
        else:
            return AWAIT_PARAMETERS_AGGREGATION


@app_state(name = PARAMETER_AGGREGATION, role = Role.COORDINATOR)
class ParameterAggregationState(AppState):
    def register(self):
        self.register_transition(target = EVALUATION, role = Role.COORDINATOR)

    def run(self):
        self.log("Parameter Aggregation")
        clients_payloads = self.gather_data()
        client_flat_vectors = [cp["local_flat_vector"] for cp in clients_payloads]
        client_data_sizes = [cp["client_data_size"] for cp in clients_payloads]
        client_weights = [cds / sum(client_data_sizes) for cds in client_data_sizes]

        coordinator = server.Server()
        global_flat_vector = coordinator.aggregate_betas(client_flat_vectors, client_weights)

        node_order = self.load('node_order')
        positions = self.load('local_positions')

        broadcast_payload = {
            "global_flat_vector": global_flat_vector,
            "node_order": node_order,
            "positions": positions,
        }
        self.broadcast_data(broadcast_payload)

        global_params = coordinator.unflatten_betas(global_flat_vector, node_order, positions)
        self.store('global_params', global_params)
        store.update(global_params=global_params,
                     category_levels=self.load('category_levels'),
                     columns=list(self.load('dataset').columns))
        self.log(f"Aggregated global params for {len(global_params)} nodes.")
        return EVALUATION


@app_state(name = AWAIT_PARAMETERS_AGGREGATION, role = Role.PARTICIPANT)
class AwaitParametersAggregationState(AppState):
    def register(self):
        self.register_transition(target = EVALUATION, role = Role.PARTICIPANT)

    def run(self):
        self.log("Await Parameter Aggregation")
        broadcast_payload = self.await_data()
        global_flat_vector = broadcast_payload["global_flat_vector"]
        node_order = broadcast_payload["node_order"]
        positions = broadcast_payload["positions"]

        participant = client.Client()
        global_params = participant.unflatten_betas(global_flat_vector, node_order, positions)

        self.store('global_params', global_params)
        store.update(global_params=global_params,
                     category_levels=self.load('category_levels'),
                     columns=list(self.load('dataset').columns))
        self.log(f"Received global params for {len(global_params)} nodes.")
        return EVALUATION


@app_state(name = EVALUATION, role = Role.BOTH)
class EvaluationState(AppState):
    def register(self):
        self.register_transition(target = VISUALIZE, role = Role.BOTH)

    @park_on_error
    def run(self):
        self.log("Evaluation")

        target = self.load('target')
        category_levels = self.load('category_levels')
        full_dataset = self.load('dataset').astype('str').astype('category')
        n_splits = self.load('n_splits') if self.load('n_splits') else 5

        participant = client.Client()
        target_classes = category_levels[target]
        y_all = pd.Categorical(full_dataset[target], categories=target_classes).codes

        class_counts = Counter(y_all)
        min_class_count = min(class_counts.values())
        effective_splits = max(2, min(n_splits, min_class_count))
        if effective_splits < n_splits:
            self.log(
                f"[EVALUATION] Rarest class has only {min_class_count} local samples; "
                f"reducing n_splits from {n_splits} to {effective_splits}."
            )

        skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=23)

        def cv_evaluate(dag, label):
            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_dataset, y_all)):
                train_fold = full_dataset.iloc[train_idx]
                test_fold = full_dataset.iloc[test_idx]
                y_true_fold = y_all[test_idx]

                edges = participant.create_network_dict(dag.edges(), train_fold.columns)
                params = participant.compute_beta_params_fixed(train_fold, edges, category_levels)
                y_prob = participant.predict_node_probability_from_beta(test_fold, target, params)
                metrics = participant.evaluate_predictions(y_true_fold, y_prob)
                y_pred = np.argmax(y_prob, axis=1) if y_prob.ndim > 1 else (y_prob > 0.5).astype(int)
                try:
                    weighted_precision = precision_score(
                        y_true_fold, y_pred, average='weighted', zero_division=0)
                except Exception:
                    weighted_precision = np.nan

                metrics['Precision (weighted)'] = weighted_precision
                metrics['num_edges'] = len(dag.edges())
                
                fold_metrics.append(metrics)

            metric_names = fold_metrics[0].keys()
            mean_metrics = {name: float(np.nanmean([m[name] for m in fold_metrics])) for name in metric_names}
            std_metrics = {name: float(np.nanstd([m[name] for m in fold_metrics])) for name in metric_names}

            self.log(f"[EVALUATION] {label} ({effective_splits}-fold CV):")
            for metric_name in metric_names:
                self.log(f"  {metric_name}: {mean_metrics[metric_name]:.4f} +/- {std_metrics[metric_name]:.4f}")

            return {"mean": mean_metrics, "std": std_metrics, "folds": fold_metrics}

        def cv_evaluate_full_evidence(dag, label):
            """
            Same regression params as cv_evaluate, but prediction runs
            through pgmpy VariableElimination with evidence_scope="full" --
            conditioning on every observed non-target column (including the
            target's children), not just its direct parents. Isolates how
            much full-network inference changes results versus the
            parents-only lookup cv_evaluate uses.
            """
            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_dataset, y_all)):
                train_fold = full_dataset.iloc[train_idx]
                test_fold = full_dataset.iloc[test_idx]
                y_true_fold = y_all[test_idx]

                edges = participant.create_network_dict(dag.edges(), train_fold.columns)
                params = participant.compute_beta_params_fixed(train_fold, edges, category_levels)
                model = participant.build_bayesian_network_from_beta(edges, params, category_levels)
                y_prob = participant.predict_target_probability_via_inference(
                    model, test_fold, target, target_classes, evidence_scope="full"
                )
                metrics = participant.evaluate_predictions(y_true_fold, y_prob)
                y_pred = np.argmax(y_prob, axis=1) if y_prob.ndim > 1 else (y_prob > 0.5).astype(int)
                try:
                    weighted_precision = precision_score(
                        y_true_fold, y_pred, average='weighted', zero_division=0)
                except Exception:
                    weighted_precision = np.nan

                metrics['Precision (weighted)'] = weighted_precision
                metrics['num_edges'] = len(dag.edges())
                
                fold_metrics.append(metrics)

            metric_names = fold_metrics[0].keys()
            mean_metrics = {name: float(np.nanmean([m[name] for m in fold_metrics])) for name in metric_names}
            std_metrics = {name: float(np.nanstd([m[name] for m in fold_metrics])) for name in metric_names}

            self.log(f"[EVALUATION] {label} ({effective_splits}-fold CV):")
            for metric_name in metric_names:
                self.log(f"  {metric_name}: {mean_metrics[metric_name]:.4f} +/- {std_metrics[metric_name]:.4f}")

            return {"mean": mean_metrics, "std": std_metrics, "folds": fold_metrics}

        def cv_evaluate_fixed_params(params, label):
            """
            For params that are already fixed going into EvaluationState
            (e.g. the federated-aggregated global_params, computed once by
            Server.aggregate_betas upstream -- NOT locally fit) -- no
            per-fold refitting happens, since there's nothing local left to
            refit. Each fold just scores the SAME fixed params against a
            different held-out slice, so the resulting mean/std still tells
            you about generalization variance across slices of local data,
            just not about local-refit variance.
            """
            fold_metrics = []
            prediction_rows = []
            final_dag = self.load('final_dag')
            num_edges = len(final_dag.edges())
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_dataset, y_all)):
                test_fold = full_dataset.iloc[test_idx]
                y_true_fold = y_all[test_idx]

                y_prob = participant.predict_node_probability_from_beta(test_fold, target, params)
                probs = np.asarray(y_prob)
                matrix = np.column_stack([1 - probs, probs]) if probs.ndim == 1 else probs
                predicted = matrix.argmax(axis=1)
                for position, row_index in enumerate(test_idx):
                    prediction_rows.append({
                        "row": int(row_index),
                        "fold": fold_idx + 1,
                        "true": target_classes[y_true_fold[position]],
                        "predicted": target_classes[predicted[position]],
                        "confidence": float(matrix[position, predicted[position]]),
                        "correct": bool(predicted[position] == y_true_fold[position]),
                    })
                metrics = participant.evaluate_predictions(y_true_fold, y_prob)
                y_pred = np.argmax(y_prob, axis=1) if y_prob.ndim > 1 else (y_prob > 0.5).astype(int)
                try:
                    weighted_precision = precision_score(
                        y_true_fold, y_pred, average='weighted', zero_division=0)
                except Exception:
                    weighted_precision = np.nan

                metrics['Precision (weighted)'] = weighted_precision
                metrics['num_edges'] = num_edges
                
                fold_metrics.append(metrics)

            metric_names = fold_metrics[0].keys()
            mean_metrics = {name: float(np.nanmean([m[name] for m in fold_metrics])) for name in metric_names}
            std_metrics = {name: float(np.nanstd([m[name] for m in fold_metrics])) for name in metric_names}

            predictions = (pd.DataFrame(prediction_rows)
                           .sort_values("row").reset_index(drop=True))
            store.update(predictions=predictions,
                         predictions_model=label,
                         predictions_accuracy=float(predictions["correct"].mean()))

            self.log(f"[EVALUATION] {label} ({effective_splits}-fold CV, fixed params, no per-fold refit):")
            for metric_name in metric_names:
                self.log(f"  {metric_name}: {mean_metrics[metric_name]:.4f} +/- {std_metrics[metric_name]:.4f}")

            return {"mean": mean_metrics, "std": std_metrics, "folds": fold_metrics}

        published = {}

        def evaluate_block(ui_label, log_label, fn, *args):
            self.log(f"[EVALUATION] computing {ui_label}...")
            try:
                block = fn(*args, log_label)
            except Exception:
                self.log(f"[EVALUATION] {ui_label} FAILED:\n{traceback.format_exc()}")
                return None
            published[ui_label] = block
            # Publish after every block so the dashboard fills in progressively.
            store.update(evaluation=dict(published))
            return block

        first_local_dag = self.load('first_local_dag')
        self.log(f"FIRST LOCAL DAG: {first_local_dag.edges()}")
        first_local_metrics = evaluate_block(
            "(1) Initial local network + local params",
            "(1) Initial local network (bootstrap DAG) + local params",
            cv_evaluate, first_local_dag)

        last_local_dag = self.load('local_dag')
        self.log(f"FINAL LOCAL DAG: {last_local_dag.edges()}")
        last_local_metrics = evaluate_block(
            "(2) Final local network + local params",
            "(2) Final local network (refined local DAG) + local params",
            cv_evaluate, last_local_dag)

        final_dag = self.load('final_dag')
        self.log(f"FINAL GLOBAL DAG: {final_dag.edges()}")
        final_global_metrics = evaluate_block(
            "(3) Final global network + local params",
            "(3) Final global network (final DAG) + local params",
            cv_evaluate, final_dag)

        global_params = self.load('global_params')
        final_global_agg_metrics = evaluate_block(
            "(4) Final global network + aggregated global params",
            "(4) Final global network (final DAG) + aggregated global params",
            cv_evaluate_fixed_params, global_params)

        self.store('first_local_metrics', first_local_metrics)
        self.store('last_local_metrics', last_local_metrics)
        self.store('final_global_metrics', final_global_metrics)
        self.store('final_global_agg_metrics', final_global_agg_metrics)

        if not published:
            store.update(error="Every evaluation block failed. See the app log "
                               "for the individual tracebacks.")
        self.log(f"[EVALUATION] done; {len(published)} blocks published to the UI.")

        try:
            self.send_data_to_coordinator(
                {"client": self.id, "evaluation": published}, memo=EVAL_MEMO)
            if self.is_coordinator:
                payloads = self.gather_data(memo=EVAL_MEMO)
                collected = {}
                for payload in payloads:
                    if isinstance(payload, dict) and payload.get("client"):
                        collected[payload["client"]] = payload["evaluation"]
                store.update(all_evaluations=collected)
                self.log(f"[EVALUATION] coordinator collected results from "
                         f"{len(collected)} clients.")
        except Exception:
            self.log(f"[EVALUATION] could not share results across clients:\n"
                     f"{traceback.format_exc()}")

        return VISUALIZE


@app_state(name = VISUALIZE, role = Role.BOTH)
class VisualizeState(AppState):
    """
    Holds the container open so results stay readable in the dashboard.

    FeatureCloud tears the container down as soon as the app reaches 'terminal'
    and only then collects /mnt/output, so everything is written to disk BEFORE
    this state blocks.

    The coordinator alone decides when the run ends: it waits for its Finish
    button, then broadcasts a sentinel. Every participant waits for that
    sentinel. Nothing in here is allowed to raise, because an exception would
    reach the engine, flip the run to ERROR and kill the dashboard.
    """
    def register(self):
        self.register_transition(target = TERMINAL, role = Role.BOTH)

    def write_results(self):
        output_dir = self.load('output_dir') or '/mnt/output'
        os.makedirs(output_dir, exist_ok=True)

        metrics = {
            'first_local': self.load('first_local_metrics'),
            'last_local': self.load('last_local_metrics'),
            'final_global': self.load('final_global_metrics'),
            'final_global_aggregated': self.load('final_global_agg_metrics'),
        }
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as fh:
            json.dump(metrics, fh, indent=2, default=str)

        final_dag = self.load('final_dag')
        if final_dag is not None:
            with open(os.path.join(output_dir, 'global_structure.json'), 'w') as fh:
                json.dump([list(edge) for edge in final_dag.edges()], fh, indent=2)

        local_dag = self.load('local_dag')
        if local_dag is not None:
            with open(os.path.join(output_dir, 'local_structure.json'), 'w') as fh:
                json.dump([list(edge) for edge in local_dag.edges()], fh, indent=2)

        self.log(f"[VISUALIZE] Results written to {output_dir}")

    def wait_as_coordinator(self):
        self.log("[VISUALIZE] Results ready. Holding the workflow open until "
                 "Finish is clicked...")
        waited = 0
        while not store.finish_clicked:
            time.sleep(1)
            waited += 1
            if waited % 60 == 0:
                self.log(f"[VISUALIZE] still waiting for Finish ({waited}s)")

        self.log("[VISUALIZE] Finish clicked. Telling participants to shut down.")
        self.broadcast_data(FINISH_SIGNAL, send_to_self=False, memo=FINISH_MEMO)

    def wait_as_participant(self):
        self.log("[VISUALIZE] Results ready. Holding the workflow open until "
                 "the coordinator ends it...")
        while True:
            try:
                signal = self.await_data(memo=FINISH_MEMO)
            except Exception:
                self.log(f"[VISUALIZE] await_data failed, retrying in 5s:\n"
                         f"{traceback.format_exc()}")
                time.sleep(5)
                continue
            if signal == FINISH_SIGNAL:
                self.log("[VISUALIZE] Coordinator ended the workflow.")
                return
            self.log(f"[VISUALIZE] ignoring unexpected payload while waiting "
                     f"for the finish signal: {signal!r}")

    def run(self):
        role = 'coordinator' if self.is_coordinator else 'participant'
        prefix = os.getenv("PATH_PREFIX")
        store.update(current_state=VISUALIZE)
        self.log(f"[VISUALIZE] entered as {role}; PATH_PREFIX={prefix!r}; "
                 f"evaluation blocks in UI="
                 f"{len(store.evaluation) if store.evaluation else 0}; "
                 f"pending incoming buckets={list(self._app.data_incoming.keys())}")

        try:
            self.write_results()
        except Exception:
            self.log(f"[VISUALIZE] Could not write results:\n{traceback.format_exc()}")

        try:
            if self.is_coordinator:
                self.wait_as_coordinator()
            else:
                self.wait_as_participant()
        except Exception:
            self.log(f"[VISUALIZE] wait failed:\n{traceback.format_exc()}")
            store.update(error="The finish handshake failed. Results are still "
                               "readable; stop the run from the FeatureCloud UI.")
            while True:
                time.sleep(5)

        store.update(finish_signalled=True)
        self.log("[VISUALIZE] -> terminal")
        return TERMINAL