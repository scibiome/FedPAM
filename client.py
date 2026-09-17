import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, label_binarize
from scipy.linalg import expm
from scipy.special import logsumexp
import torch
import torch.optim as optim
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from pgmpy.estimators import HillClimbSearch, ExpertKnowledge, BayesianEstimator, BIC, MaximumLikelihoodEstimator, PC, GES, AIC,BDeu
from scipy.stats import chi2_contingency, chi2 as chi2_dist
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
)

import os
import json
from collections import Counter, defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List

import logging
import warnings
from tqdm import tqdm
from functools import partialmethod

logging.getLogger("pgmpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

class Client():
    def disable_pgmpy_logs(self):
        logger = logging.getLogger("pgmpy")
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

    def learn_local_structure(self,
        dataset: pd.DataFrame = None,
        has_target: bool = False,
        target: str = None,
        testing: bool = True,
        num_hc_iter: int = 1000000
    ):
        self.disable_pgmpy_logs()

        if has_target and target:
            forbidden_edges = [(target, var) for var in dataset.columns if var != target]
            expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)
        else:
            expert_knowledge = None

        if testing: 
            dataset = dataset.astype('str').astype('category')

        equivalent_sample_size = 0.01 * len(dataset)
        structure = HillClimbSearch(dataset).estimate(
            scoring_method=BDeu(data = dataset, equivalent_sample_size=equivalent_sample_size),
            expert_knowledge=expert_knowledge,
            show_progress=False,
            max_iter=num_hc_iter
        )

        return structure

    def create_pam(self,
        dataset: pd.DataFrame = None,
        has_target: bool = False,
        target: str = None,
        testing: bool = True,
        num_iterations: int = 100,
        n_jobs: int = 1,
        seed: int = None,
    ):
        self.disable_pgmpy_logs()

        rng = np.random.default_rng(seed)
        random_states = rng.integers(0, 1_000_000, size=num_iterations)

        def build_sample(random_state):
            sample = dataset.sample(n=len(dataset), replace=True, random_state=random_state)
            cols = list(sample.columns)
            np.random.default_rng(random_state).shuffle(cols)
            return sample[cols]

        bootstrap_samples = [build_sample(int(rs)) for rs in random_states]

        with tqdm_joblib(tqdm(desc="Learning DAGs", total=num_iterations)):
            dags = Parallel(n_jobs=n_jobs)(
                delayed(self.learn_local_structure)(
                    dataset=sample,
                    has_target=has_target,
                    target=target,
                    testing=testing
                )
                for sample in bootstrap_samples
            )

        all_edges = []

        for dag in dags:
            all_edges.extend(dag.edges())

        edge_counts = Counter(all_edges)
        edge_strengths = {
            edge: count / len(dags) for edge, count in edge_counts.items()
        }

        allowed_edges = list(edge_strengths.keys())

        return edge_strengths, allowed_edges

    def learn_constrained_local_dag(self,
        dataset: pd.DataFrame = None,
        allowed_edges: List = None,
        has_target: bool = False,
        target: str = None,
        testing: bool = True,
        num_hc_iter=1000000
    ):
        self.disable_pgmpy_logs()

        forbidden_edges = None
        if has_target and target:
            forbidden_edges = [(target, var) for var in dataset.columns if var != target]

        # pgmpy's ExpertKnowledge takes `search_space`, not `allowed_edges`.
        # `allowed_edges` was absorbed by **kwargs and silently discarded, so
        # this "constrained" search was in fact unconstrained.
        if allowed_edges is not None:
            allowed_edges = [tuple(edge) for edge in allowed_edges]
            # An allowed edge out of the target would defeat forbidden_edges,
            # so strip those before they reach the search space.
            if has_target and target:
                allowed_edges = [(u, v) for u, v in allowed_edges if u != target]

        expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges,
                                           search_space=allowed_edges)

        if testing:
            dataset = dataset.astype('str').astype('category')

        equivalent_sample_size = 0.01 * len(dataset)
        structure = HillClimbSearch(dataset).estimate(
            scoring_method=BDeu(data = dataset, equivalent_sample_size=equivalent_sample_size),
            expert_knowledge=expert_knowledge,
            show_progress=False,
            max_iter=num_hc_iter
        )

        return structure

    def get_allowed_edges(self, pam_global, threshold=0.5):
        return [edge for edge, strength in pam_global.items() if strength >= threshold]

    def refine_local_pam(self,
        local_pam_prev: dict,
        pam_global: dict,
        constrained_network,
        alpha: float,
        gamma: float):
        """
        Refines a client's local PAM using the broadcasted global PAM and this
        round's locally learned constrained network. All PAMs are sparse
        {(u, v): strength} dicts; missing entries are treated as 0.0.

        local_pam_new = alpha * local_pam_prev + (1 - alpha) * pam_global
                        + gamma * (+0.1 if edge in constrained_network else -0.1)
        clipped to [0, 1]. Entries that clip to 0.0 are dropped.
        """
        nodes = set()
        for u, v in local_pam_prev.keys():
            nodes.add(u); nodes.add(v)
        for u, v in pam_global.keys():
            nodes.add(u); nodes.add(v)
        for u, v in constrained_network.edges():
            nodes.add(u); nodes.add(v)

        constrained_edges = set(constrained_network.edges())

        edge_strengths = {}
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                edge = (u, v)
                indicator = 1 if edge in constrained_edges else -1
                value = alpha * local_pam_prev.get(edge, 0.0) + (1 - alpha) * pam_global.get(edge, 0.0) + gamma * indicator
                value = min(max(value, 0.0), 1.0)
                value = round(value, 2)
                if value > 0.0:
                    edge_strengths[edge] = value

        allowed_edges = list(edge_strengths.keys())

        return edge_strengths, allowed_edges

    def compute_shd(self, true_model, est_model):
        """
        Computes Structural Hamming Distance (SHD) between two Bayesian Networks.

        Handles isolated nodes correctly (unlike some pgmpy versions).
        """

        true_nodes = set(true_model.nodes())
        est_nodes = set(est_model.nodes())

        if true_nodes != est_nodes:
            raise ValueError(
                f"Node sets differ.\n"
                f"Missing in estimated: {true_nodes - est_nodes}\n"
                f"Extra in estimated: {est_nodes - true_nodes}"
            )

        nodes = sorted(true_nodes)

        # Build graphs INCLUDING isolated nodes
        G_true = nx.DiGraph()
        G_true.add_nodes_from(nodes)
        G_true.add_edges_from(true_model.edges())

        G_est = nx.DiGraph()
        G_est.add_nodes_from(nodes)
        G_est.add_edges_from(est_model.edges())

        A_true = nx.to_numpy_array(G_true, nodelist=nodes, dtype=int)
        A_est = nx.to_numpy_array(G_est, nodelist=nodes, dtype=int)

        shd = 0

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # edge i->j
                t_ij = A_true[i, j]
                e_ij = A_est[i, j]

                # edge j->i
                t_ji = A_true[j, i]
                e_ji = A_est[j, i]

                if (t_ij, t_ji) == (e_ij, e_ji):
                    continue

                # reversed edge counts as one edit
                if t_ij and e_ji:
                    shd += 1
                elif t_ji and e_ij:
                    shd += 1
                else:
                    # additions/deletions
                    shd += abs(t_ij - e_ij) + abs(t_ji - e_ji)

        return shd


    def create_network_dict(self,
        edges, dataset_columns):
        parents = defaultdict(list)

        for parent, child in edges:
            parents[child].append(parent)

        return {
            node: list(parents[node])
            for node in dataset_columns
        }

    def get_local_category_levels(self, dataset: pd.DataFrame) -> dict:
        """
        Per-column sorted unique values this client observes locally, as
        strings. Send this to the coordinator (once, after the final DAG is
        agreed) so it can build the GLOBAL union per column — required
        before compute_beta_params_fixed, otherwise each client's dummy-encoded
        design matrix (and therefore its flattened params vector) ends up a
        different length and aggregate_betas can't sum them.
        """
        return {col: sorted(dataset[col].astype(str).unique().tolist()) for col in dataset.columns}

    def compute_beta_params_fixed(self,
        dataset: pd.DataFrame = None,
        network_edges: dict = None,
        category_levels: dict = None):
        """
        Aggregation-safe parameter fitting: every shape decision comes from
        `category_levels` (the GLOBAL union, agreed by every client before
        this is called — see get_local_category_levels / Server.merge_category_levels),
        not this client's local data. Two things that plain compute_beta_params
        doesn't handle, both needed for aggregate_betas to be meaningful:

        1. PARENTS: dummy columns are built from category_levels[parent],
           so every client's feature_names/coefficient shape matches
           regardless of which parent categories it happens to observe.

        2. THE NODE'S OWN CLASSES (multiclass-safe): multinomial logistic
           regression coefficients aren't uniquely determined — you can
           shift every class's row by a constant and get identical
           predictions, so different clients' raw sklearn fits land in
           different, incomparable "gauges." This fixes that by re-expressing
           every fit relative to one shared reference class (category_levels[node][0]):
           subtract that class's row from every other row, then drop it —
           the reference class becomes the implicit all-zero baseline.
           A client whose local data is missing a given class contributes
           an all-zero row for it (no evidence, defers to the baseline)
           instead of crashing or producing a differently-shaped output.

        Returns
        -------
        dict {node: {
            "has_parents": bool,
            "classes": [...],                     # GLOBAL class list, reference class first
            "intercept": (n_classes-1,) array,
            "coefficients": (n_classes-1, n_features) array,   # only if has_parents
            "parents": [...], "parent_levels": {...}, "feature_names": [...],  # only if has_parents
        }}
        """
        network_params = {}

        for variable in dataset.columns:
            parents = network_edges[variable]
            classes = category_levels[variable]
            n_classes = len(classes)
            ref_class, other_classes = classes[0], classes[1:]
            n_ref_rows = n_classes - 1

            if not parents:
                counts = dataset[variable].astype(str).value_counts()
                total = len(dataset)
                smoothed = {c: (counts.get(c, 0) + 1) / (total + n_classes) for c in classes}
                p_ref = smoothed[ref_class]
                intercept = np.array([np.log(smoothed[c] / p_ref) for c in other_classes])

                network_params[variable] = {
                    "has_parents": False,
                    "classes": classes,
                    "intercept": intercept,
                }
                continue

            feature_names = []
            X_cols = []
            for parent in parents:
                parent_classes = category_levels[parent]
                X_src = pd.Categorical(dataset[parent].astype(str), categories=parent_classes)
                dummies = pd.get_dummies(X_src, drop_first=True, dtype=int)
                dummies.columns = [f"{parent}__{lvl}" for lvl in parent_classes[1:]]
                feature_names.extend(dummies.columns.tolist())
                X_cols.append(dummies)
            X = pd.concat(X_cols, axis=1)
            X_array = X.to_numpy()
            y_raw = dataset[variable].astype(str)
            local_classes_present = sorted(y_raw.unique().tolist())

            coef = np.zeros((n_ref_rows, len(feature_names)))
            intercept = np.zeros(n_ref_rows)

            if len(local_classes_present) < 2:
                pass
            else:
                y_codes = pd.Categorical(y_raw, categories=local_classes_present).codes
                lr = LogisticRegression(solver="lbfgs", max_iter=500)  # fit_intercept=True (default)
                lr.fit(X_array, y_codes)

                if len(local_classes_present) == 2:
                    local_ref, local_other = local_classes_present
                    if local_other in other_classes:
                        row, sign = other_classes.index(local_other), 1.0
                    elif local_other == ref_class and local_ref in other_classes:
                        row, sign = other_classes.index(local_ref), -1.0
                    else:
                        row = None
                    if row is not None:
                        coef[row, :] = sign * lr.coef_[0]
                        intercept[row] = sign * lr.intercept_[0]
                else:
                    if ref_class in local_classes_present:
                        ref_idx = local_classes_present.index(ref_class)
                        ref_coef_row = lr.coef_[ref_idx]
                        ref_intercept_row = lr.intercept_[ref_idx]
                        for i, cls in enumerate(local_classes_present):
                            if cls == ref_class:
                                continue
                            global_row = other_classes.index(cls)
                            coef[global_row, :] = lr.coef_[i] - ref_coef_row
                            intercept[global_row] = lr.intercept_[i] - ref_intercept_row

            network_params[variable] = {
                "has_parents": True,
                "classes": classes,
                "parents": parents,
                "parent_levels": {p: category_levels[p] for p in parents},
                "feature_names": feature_names,
                "coefficients": coef,
                "intercept": intercept,
            }

        return network_params

    def compute_beta_params(self,
        dataset: pd.DataFrame = None,
        network_edges: dict = None,
        category_levels: dict = None):
        """
        `category_levels`, if provided, is the GLOBAL {column: [values]}
        map (from get_local_category_levels + Server.merge_category_levels,
        broadcast identically to every client). Every client then builds
        the same dummy columns / same "keys" for a given node regardless of
        which values it happens to observe locally, so every client's
        flatten_betas output is the same length and aggregate_betas can sum
        them. Without it, falls back to this client's own local uniques
        (fine for single-client use, e.g. the local-model evaluation in
        EvaluationState, which is never aggregated across clients).
        """
        network_params = {}
        for variable in dataset.columns:
            parents = network_edges[variable]
            if not parents:
                if category_levels is not None and variable in category_levels:
                    keys = category_levels[variable]
                else:
                    keys = sorted(dataset[variable].astype(str).unique().tolist())
                local_counts = dataset[variable].astype(str).value_counts()
                total = len(dataset)
                probs = {k: local_counts.get(k, 0) / total for k in keys}
                network_params[variable] = {
                    "has_parents": False,
                    "params": probs
                }
            else:
                if category_levels is not None:
                    parent_levels = {parent: category_levels[parent] for parent in parents}
                else:
                    parent_levels = {
                        parent: sorted(dataset[parent].astype(str).unique().tolist())
                        for parent in parents
                    }

                X_src = dataset[parents].copy()
                for p in parents:
                    X_src[p] = pd.Categorical(X_src[p].astype(str), categories=parent_levels[p])

                X = pd.get_dummies(X_src, drop_first=True, dtype=int)

                # Add intercept column
                X.insert(0, "Intercept", 1)

                feature_names = X.columns.tolist()

                # Convert to numpy to avoid narwhals duplicate column name validation
                X_array = X.to_numpy()

                y = pd.Categorical(dataset[variable]).codes
                lr = LogisticRegression(solver="lbfgs", fit_intercept=False, max_iter=500)
                lr.fit(X_array, y)

                network_params[variable] = {
                    "has_parents": True,
                    "parents": parents,
                    "parent_levels": parent_levels,
                    "feature_names": feature_names,
                    "model": lr,
                    "params": lr.coef_.tolist(),
                }

        return network_params

    def predict_node_probability(self, test_df: pd.DataFrame, node: str, network_params: dict) -> np.ndarray:
        """
        Binary node: returns a 1D array, P(class=1) per row (unchanged
        behavior). Multiclass node (e.g. a 3-class target): returns the
        full (n_rows, n_classes) probability matrix instead — evaluate_predictions
        below detects which shape it got and scores accordingly. Column
        order matches info["model"].classes_ (i.e. the sorted category
        codes the model was fit on).
        """
        info = network_params[node]
        if not info["has_parents"]:
            return np.full(len(test_df), float(info["params"].get(1, info["params"].get("1", 0.0))))
        pars = info["parents"]
        X_src = test_df[pars].copy()
        for p in pars:
            X_src[p] = pd.Categorical(X_src[p].astype(str), categories=info["parent_levels"][p])
        X_test = pd.get_dummies(X_src, drop_first=True, dtype=int)
        X_test.insert(0, "Intercept", 1)
        X_test = X_test.reindex(columns=info["feature_names"], fill_value=0)

        proba = info["model"].predict_proba(X_test.to_numpy())
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba


    def predict_node_probability_from_beta(self, test_df: pd.DataFrame, node: str, network_params: dict) -> np.ndarray:
        """
        Computes probabilities directly from raw intercept/coefficients
        arrays (as produced by compute_beta_params_fixed / unflatten_betas)
        instead of requiring a fitted sklearn "model" object — needed for
        aggregated/global params, which only ever carry numeric arrays
        across the wire. Handles the reference-class encoding: class
        info["classes"][0] is the implicit all-zero baseline; every other
        class's logit is intercept[i] + X @ coefficients[i].

        Returns a 1D array (P(class=1)) when there are exactly 2 classes,
        matching predict_node_probability's binary convention; otherwise
        returns the full (n_rows, n_classes) probability matrix, columns
        in info["classes"] order.
        """
        info = network_params[node]
        classes = info["classes"]
        n_classes = len(classes)

        if not info["has_parents"]:
            logits_other = np.asarray(info["intercept"], dtype=float)
            logits = np.concatenate([[0.0], logits_other])
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            full = np.tile(probs, (len(test_df), 1))
            return full[:, 1] if n_classes == 2 else full

        pars = info["parents"]
        X_cols = []
        for parent in pars:
            parent_classes = info["parent_levels"][parent]
            X_src = pd.Categorical(test_df[parent].astype(str), categories=parent_classes)
            dummies = pd.get_dummies(X_src, drop_first=True, dtype=int)
            dummies.columns = [f"{parent}__{lvl}" for lvl in parent_classes[1:]]
            X_cols.append(dummies)
        X_test = pd.concat(X_cols, axis=1)
        X_test = X_test.reindex(columns=info["feature_names"], fill_value=0)
        X_array = X_test.to_numpy()

        coef = np.asarray(info["coefficients"], dtype=float)      # (n_classes-1, n_features)
        intercept = np.asarray(info["intercept"], dtype=float)    # (n_classes-1,)
        logits_other = X_array @ coef.T + intercept               # (n_rows, n_classes-1)
        logits = np.concatenate([np.zeros((len(test_df), 1)), logits_other], axis=1)  # reference class = 0
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs[:, 1] if n_classes == 2 else probs

    def evaluate_predictions(self, y_true, y_prob) -> dict:
        """
        y_prob can be either:
          - 1D array (n_rows,): binary P(class=1) — original behavior.
          - 2D array (n_rows, n_classes): multiclass probability matrix, as
            returned by predict_node_probability for a >2-class node.
        y_true must be integer class codes 0..n_classes-1 in the SAME order
        as y_prob's columns (i.e. matching info["model"].classes_ — see the
        caller for how that alignment is guaranteed).
        """
        y_prob = np.asarray(y_prob)

        if y_prob.ndim == 1:
            y_pred = (y_prob >= 0.5).astype(int)
            return {
                "Accuracy":  accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall":    recall_score(y_true, y_pred, zero_division=0),
                "F1":        f1_score(y_true, y_pred, zero_division=0),
                "ROC_AUC":   roc_auc_score(y_true, y_prob),
                "PR_AUC":    average_precision_score(y_true, y_prob),
                "LogLoss":   log_loss(y_true, y_prob),
                "Brier":     brier_score_loss(y_true, y_prob),
            }

        n_classes = y_prob.shape[1]
        labels = list(range(n_classes))
        y_pred = np.argmax(y_prob, axis=1)

        metrics = {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
            "Recall":    recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
            "F1":        f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
        }

        try:
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=labels)
        except ValueError:
            metrics["ROC_AUC"] = float('nan')

        y_true_onehot = label_binarize(y_true, classes=labels)
        try:
            metrics["PR_AUC"] = average_precision_score(y_true_onehot, y_prob, average='macro')
        except ValueError:
            metrics["PR_AUC"] = float('nan')

        metrics["LogLoss"] = log_loss(y_true, y_prob, labels=labels)
        metrics["Brier"] = float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))

        return metrics

    def get_node_order(self, final_dag, dataset_columns=None):
        """
        1. Node ordering from the final global network: topological order
        of `final_dag`. Falls back to sorted column names if `final_dag`
        somehow isn't a valid DAG (shouldn't happen post finalize_dag, but
        keeps this usable standalone).
        """
        if nx.is_directed_acyclic_graph(final_dag):
            order = list(nx.topological_sort(final_dag))
        else:
            order = sorted(final_dag.nodes())

        if dataset_columns is not None:
            # include any isolated columns not present as nodes in final_dag
            for col in dataset_columns:
                if col not in order:
                    order.append(col)

        return order

    def flatten_betas(self, network_params: dict, node_order: List):
        """
        Flattens the dict returned by compute_beta_params_fixed into a
        single numpy vector, following `node_order`. Every node contributes
        its "intercept" (always) and "coefficients" (only if has_parents).
        Because every client calls compute_beta_params_fixed with the same
        GLOBAL category_levels and the same final_dag, every client's shapes
        here are identical — that's what makes aggregate_betas's weighted
        sum meaningful. Returns (flat_vector, positions), where `positions`
        records each node's slice + shape + metadata so unflatten_betas can
        reconstruct it.
        """
        flat = []
        positions = {}
        pos = 0

        for node in node_order:
            info = network_params[node]

            intercept = np.asarray(info["intercept"], dtype=float).flatten()
            meta = {
                "has_parents": info["has_parents"],
                "classes": info["classes"],
                "intercept": (pos, pos + len(intercept)),
            }
            flat.extend(intercept.tolist())
            pos += len(intercept)

            if info["has_parents"]:
                coef = np.asarray(info["coefficients"], dtype=float)
                coef_flat = coef.flatten()
                meta["coefficients"] = (pos, pos + len(coef_flat))
                meta["coef_shape"] = coef.shape
                meta["parents"] = info["parents"]
                meta["parent_levels"] = info["parent_levels"]
                meta["feature_names"] = info["feature_names"]
                flat.extend(coef_flat.tolist())
                pos += len(coef_flat)

            positions[node] = meta

        return np.array(flat), positions

    def unflatten_betas(self, flat_vector: np.ndarray, node_order: List, positions: dict):
        """
        Reverses flatten_betas. Returns the same dict shape as
        compute_beta_params_fixed (intercept/coefficients/classes/parents/...),
        ready to pass into predict_node_probability_from_beta.
        """
        network_params = {}

        for node in node_order:
            meta = positions[node]

            s, e = meta["intercept"]
            intercept = np.asarray(flat_vector[s:e])

            node_params = {
                "has_parents": meta["has_parents"],
                "classes": meta["classes"],
                "intercept": intercept,
            }

            if meta["has_parents"]:
                s, e = meta["coefficients"]
                coef = np.asarray(flat_vector[s:e]).reshape(meta["coef_shape"])
                node_params["coefficients"] = coef
                node_params["parents"] = meta["parents"]
                node_params["parent_levels"] = meta["parent_levels"]
                node_params["feature_names"] = meta["feature_names"]

            network_params[node] = node_params

        return network_params

    def evaluate_global_network(self,
        train_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        network_edges: dict = None,
        target: str = None) -> dict:
        """
        Fits beta params for the fixed FINAL GLOBAL structure (`network_edges`,
        learned only on training data across all clients) on `train_df`
        only, then predicts and scores `target` on `test_df`.
        """
        network_params = self.compute_beta_params(
            dataset=train_df,
            network_edges=network_edges,
        )

        y_prob = self.predict_node_probability(test_df, target, network_params)

        target_categories = pd.Categorical(train_df[target]).categories
        y_true = pd.Categorical(test_df[target], categories=target_categories).codes

        return self.evaluate_predictions(y_true, y_prob)