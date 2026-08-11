from FeatureCloud.app.engine.app import AppState, Role, State, app_state
import time
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
from sklearn.model_selection import train_test_split

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='pgmpy')
warnings.filterwarnings('ignore', message='.*Replacing existing CPD.*')
warnings.filterwarnings('ignore', message='.*pgmpy.*')

import client
import server

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
TERMINAL = 'terminal'

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
        self.log("Initializing FedPAM Application...")
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
        self.store('target', configs['input']['target'])
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
        test_splits = {}

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
            if testing:
                dataset = dataset.sample(n = num_samples, random_state=23).reset_index(drop=True)
            else:
                dataset = dataset.sample(frac=1, random_state=23).reset_index(drop=True)

            test_dataset = None
            if has_target:
                target = self.load('target')
                dataset, test_dataset = train_test_split(
                    dataset,
                    test_size=0.2,
                    random_state=23,
                    stratify=dataset[target]
                )
                dataset = dataset.reset_index(drop=True)
                test_dataset = test_dataset.reset_index(drop=True)

            splits[split_path] = dataset
            test_splits[split_path] = test_dataset
            self.log(
                f"Local dataset from {split_path}: {dataset.shape[0]} train observations"
                f"{f', {test_dataset.shape[0]} test observations' if test_dataset is not None else ''} "
                f"and {dataset.shape[1]} variables."
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
        self.store('test_dataset', test_splits[client_split_path])
        self.store('client_split_path', client_split_path)

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
        self.store('first_local_dag', local_dag)  # never overwritten by LOCAL_REFINEMENT, unlike 'local_dag'

        if testing_flag:
            benchmark_network = get_example_model(benchmark)
            true_edges = set(benchmark_network.edges())

            # ---------------- SINGLE ITERATION ----------------
            single_iter_network = participant.learn_local_structure(
                dataset, False, None, True
            )

            single_iter_shd = participant.compute_shd(
                benchmark_network, single_iter_network
            )

            single_pred_edges = set(single_iter_network.edges())

            single_tp = len(single_pred_edges & true_edges)
            single_fp = len(single_pred_edges - true_edges)
            single_fn = len(true_edges - single_pred_edges)

            single_precision = (
                single_tp / len(single_pred_edges)
                if single_pred_edges else 0
            )
            single_recall = (
                single_tp / len(true_edges)
                if true_edges else 0
            )
            single_f1 = (
                2 * single_precision * single_recall /
                (single_precision + single_recall)
                if (single_precision + single_recall) > 0 else 0
            )

            self.log("[TESTING] --- SINGLE ITERATION ---")
            self.log(f"Number of edges: {single_iter_network.number_of_edges()}")
            self.log(f"SHD: {single_iter_shd}")
            self.log(f"TP: {single_tp}, FP: {single_fp}, FN: {single_fn}")
            self.log(f"Precision: {single_precision:.3f}")
            self.log(f"Recall (TPR): {single_recall:.3f}")
            self.log(f"F1-score: {single_f1:.3f}")

            # ---------------- BOOTSTRAP NETWORK ----------------
            bootstrap_network_shd = participant.compute_shd(
                benchmark_network, local_dag
            )

            bootstrap_pred_edges = set(local_dag.edges())

            bootstrap_tp = len(bootstrap_pred_edges & true_edges)
            bootstrap_fp = len(bootstrap_pred_edges - true_edges)
            bootstrap_fn = len(true_edges - bootstrap_pred_edges)

            bootstrap_precision = (
                bootstrap_tp / len(bootstrap_pred_edges)
                if bootstrap_pred_edges else 0
            )
            bootstrap_recall = (
                bootstrap_tp / len(true_edges)
                if true_edges else 0
            )
            bootstrap_f1 = (
                2 * bootstrap_precision * bootstrap_recall /
                (bootstrap_precision + bootstrap_recall)
                if (bootstrap_precision + bootstrap_recall) > 0 else 0
            )

            self.log("[TESTING] --- BOOTSTRAP NETWORK ---")
            self.log(f"Number of edges: {local_dag.number_of_edges()}")
            self.log(f"SHD: {bootstrap_network_shd}")
            self.log(f"TP: {bootstrap_tp}, FP: {bootstrap_fp}, FN: {bootstrap_fn}")
            self.log(f"Precision: {bootstrap_precision:.3f}")
            self.log(f"Recall (TPR): {bootstrap_recall:.3f}")
            self.log(f"F1-score: {bootstrap_f1:.3f}")

            # ---------------- COMPARISON ----------------
            self.log("[TESTING] --- COMPARISON ---")
            self.log(f"ΔSHD: {bootstrap_network_shd - single_iter_shd:+d}")
            self.log(f"ΔPrecision: {bootstrap_precision - single_precision:+.3f}")
            self.log(f"ΔRecall: {bootstrap_recall - single_recall:+.3f}")
            self.log(f"ΔF1-score: {bootstrap_f1 - single_f1:+.3f}")

        if testing_flag:
            local_payload = {
                "client_data_size": dataset_size,
                "client_pam": local_edge_strengths,
                "num_local_edges": local_dag.number_of_edges(),
                "local_shd": bootstrap_network_shd,
                "local_tpr": bootstrap_recall
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

            mean_client_num_edges = statistics.mean(client_num_edges)
            mean_client_shd = statistics.mean(client_shds)
            mean_client_tpr = statistics.mean(client_tprs)

            std_client_num_edges = statistics.stdev(client_num_edges)
            std_client_shd = statistics.stdev(client_shds)
            std_client_tpr = statistics.stdev(client_tprs)

            self.log(f"[TESTING] Client local metrics")
            self.log(f"Local SHD: {mean_client_shd} ± {std_client_shd}")
            self.log(f"Local TPR: {mean_client_tpr} ± {std_client_tpr}")
            self.log(f"Number of local edges: {mean_client_num_edges} ± {std_client_num_edges}")

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

        participant = client.Client()
        local_pam, _ = participant.refine_local_pam(local_pam_prev, global_pam, local_dag, alpha, gamma)
        allowed_edges = global_pam.keys()
        local_dag = participant.learn_constrained_local_dag(dataset, allowed_edges)
        self.store('local_dag', local_dag)
        self.store('local_pam', local_pam)
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
        self.register_transition(target = TERMINAL, role = Role.BOTH)
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

        coordinator = server.Server()
        final_dag, is_valid = coordinator.finalize_dag(global_pam, nodes, threshold=threshold)

        self.log(f"[FINAL]: Final DAG edges -> {list(final_dag.edges())}")
        self.log(f"[FINAL]: Valid DAG -> {is_valid}")
        self.log(f"No. of edges: {final_dag.number_of_edges()}")

        self.store('final_dag', final_dag)
        self.store('final_dag_valid', is_valid)

        if testing_flag: 
            benchmark_network = get_example_model(benchmark)
            true_edges = [edge for edge in final_dag.edges() if edge in benchmark_network.edges()]
            final_network_shd = coordinator.compute_shd(benchmark_network, final_dag)
            self.log(f"[TESTING] FINAL NETWORK")
            self.log(f"Final network SHD: {final_network_shd}")
            self.log(f"TPR: {len(true_edges) / len(benchmark_network.edges())}")

        if has_target:
            participant = client.Client()
            local_levels = participant.get_local_category_levels(dataset)
            self.send_data_to_coordinator(local_levels)

            if self.is_coordinator:
                return CATEGORY_LEVELS
            else:
                return AWAIT_CATEGORY_LEVELS
        else:
            self.log("Final to terminal state")
            return TERMINAL


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
        self.log(f"Received global params for {len(global_params)} nodes.")
        return EVALUATION


@app_state(name = EVALUATION, role = Role.BOTH)
class EvaluationState(AppState):
    def register(self):
        self.register_transition(target = TERMINAL, role = Role.BOTH)

    def run(self):
        self.log("Evaluation")

        target = self.load('target')
        category_levels = self.load('category_levels')
        train_dataset = self.load('dataset').astype('str').astype('category')
        test_dataset = self.load('test_dataset').astype('str').astype('category')

        participant = client.Client()
        target_classes = category_levels[target]
        y_true = pd.Categorical(test_dataset[target], categories=target_classes).codes

        def fit_and_evaluate(dag, label):
            edges = participant.create_network_dict(dag.edges(), train_dataset.columns)
            params = participant.compute_beta_params_fixed(train_dataset, edges, category_levels)
            y_prob = participant.predict_node_probability_from_beta(test_dataset, target, params)
            metrics = participant.evaluate_predictions(y_true, y_prob)

            self.log(f"[EVALUATION] {label}:")
            for metric_name, value in metrics.items():
                self.log(f"  {metric_name}: {value:.4f}")
            return metrics

        first_local_dag = self.load('first_local_dag')
        first_local_metrics = fit_and_evaluate(
            first_local_dag, "First local network (first local DAG + local params) on local test data"
        )

        last_local_dag = self.load('local_dag')
        last_local_metrics = fit_and_evaluate(
            last_local_dag, "Last local network (last local DAG + local params) on local test data"
        )

        global_params = self.load('global_params')
        global_y_prob = participant.predict_node_probability_from_beta(test_dataset, target, global_params)
        global_metrics = participant.evaluate_predictions(y_true, global_y_prob)

        self.log("[EVALUATION] Final global network (final DAG + aggregated global params) on local test data:")
        for metric_name, value in global_metrics.items():
            self.log(f"  {metric_name}: {value:.4f}")

    
        local_params = self.load('local_params')
        local_global_y_prob = participant.predict_node_probability_from_beta(test_dataset, target, local_params)
        local_global_metrics = participant.evaluate_predictions(y_true, local_global_y_prob)

        self.log("[EVALUATION] Final global network (final DAG + local params) on local test data:")
        for metric_name, value in local_global_metrics.items():
            self.log(f"  {metric_name}: {value:.4f}")

        self.store('first_local_metrics', first_local_metrics)
        self.store('last_local_metrics', last_local_metrics)
        self.store('global_metrics', global_metrics)
        self.store('local_global_metrics', local_global_metrics)

        return TERMINAL