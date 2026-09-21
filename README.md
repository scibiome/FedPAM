# FedPAM 
FedPAM is a federated framework for discrete Bayesian network learning using Probabilistic Adjacency Matrices (PAMs).

### Datasets Used
1. **[Chronic Kidney Disease (CKD) prediction dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease):** 
   * Contains 400 samples with 24 discrete medical, laboratory, and demographic variables for **binary classification** of disease presence.
   * Stored in directory `data/ckd_400` and is split into $K=3$ homogeneous client datasets.

2. **[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success):** 
    * Contains 4,424 samples with
36 variables for <b>multi-class classification (3 classes) </b> of academic outcomes based on enrollment
information and first- and second-semester performance.
    * Stored in `data/clients_datasets` and is split into $K = 3$ to $10$ client datasets for both homogeneous and heterogeneous settings.

### Config File

Modify the hyperparameters in `config.yml` file based on your requirements.

```
fc-fedpam:
  input:
    dataset_location: "client.csv"
    has_target: true
    target: 'class'
    # exp_know_location: "expert_knowledge.json"
  split:
    mode: "file"
    dir: "."  
  alpha: 0.1
  gamma: 0.15
  num_bootstrap_iterations: 100
  max_iterations: 100
  homogeneous: True
  testing: false
  num_hc_iter: 100
  threshold: 0.5
  benchmark: 'child'
  num_samples: null
  num_jobs: 3
```

#### Description of Hyperparameters:

1. `dataset_location`: Location of the csv file containing the discrete dataset. 
During <b>app testing </b>, use the following directory structure:
```
data
└───clients_datasets_directory
│   └──client1
│       │client.csv
│   └──client2
│       │client.csv
│   └──client3
│       │client.csv
```

Check the `data` directory in the fc-fedpam repository before running the app to avoid any errors related to file paths. To test the app on the example datasets present in the `data` directory, change `clients_datasets_directory` to `ckd_400`, `clients_datasets/clients_03` etc.

During actual federated workflow, you will be required to upload a shared `config.yml` file and a `client.csv` file containing the discrete dataset.

2. `has_target`: A boolean variable to inform the model if a target variable is present in the dataset or not.

3. `target`: Set this to the target variable in the dataset if it exists. For CKD-400, use 'class' and for the students success prediction dataset, use 'Target'.

4. `mode`: Controls how the app finds data splits. If set to `mode: 'directory'`, the app looks for subdirectories inside a base folder to use as separate client data splits. Otherwise, it uses the main `/mnt/input` directory as the single split. During testing, you can change client data directories using the FeatureCloud test-bed/workflow interface.

5. `dir`: The base directory (relative to `/mnt/input`) that contains subdirectories for each client's data split. 

6. `alpha`: Hyperparameter $\alpha$ for tuning the dominance of local PAM $P_k$ over global PAM $P_{global}$, to mitigate local drifts due to statistical heterogeneity. Results show that for homogeneous settings, keeping $\alpha=0.5$ gives the best results as both local and global PAM are created from statistically similar data. However, using lower values like 0.1 or 0.2 is preferred for heterogeneous cases.

7. `gamma`: Hyperparameter $\gamma$ for controlling the speed of convergence. Essentially, this hyperparameter acts as a weight for local DAG in each iteration, ensuring that while the algorithm learns from global knowledge, the local evidence is also preserved. Based on experiments, values like 0.1 and 0.15 show faster convergence to low SHD values for a variety of structural complexities.

8. `num_bootstrap_iterations`: Total number of bootstrap iterations. By default, this number $B$ is set to 100 for higher variability and more structural exploration.

9. `max_iterations`: Total number of federated learning rounds.

10. `homogeneous`: Boolean hyperparameter to switch between homogeneous and heterogeneous learning modes. If the existing client data is "known" to be homogeneous, set `homogeneous: true`. Otherwise, set `homogeneous: False`. In fact, in real-world scenarios, keeping the latter is suggested as the client distributions are usually unknown.

11. `testing`: To evaluate the algorithm on BN benchmarks, set this parameter to `true`. Otherwise, `false`.

12. `benchmark`: Name of the BN benchmark used for structure-only evaluation.

13. `num_hc_iter`: Pre-defined number of hill climb search iterations during local learning stage. This decides how much exploration is required by the structure learning algorithm. 

14. `threshold`: Across FL rounds, the local refinement followed by server-side aggregation pushes pushes the PAM probabilities towards either 0 or 1. Therefore, a `threshold` value of 0.5 acts as a suitable measure to cluster the PAM elements into two groups - significant edges and insignificant edges. As a result, the PAM is binzarized and only the significant edges are included in the final global network.

15. `num_samples`: If all clients need to have the same number of samples, use this parameter to control the common sample size. Otherwise, keep it to the default value `null` to ensure variability in client sample size.

16. `num_jobs`: The algorithm supports parallelization of bootstrapping. Use this parameter to allocate $c$ CPU cores for running each Hill Climbing algorithm. During testing, keep in mind that for $K$ clients, $K \times c$ CPU cores will be used in total.

### Steps to run FedPAM application:
1. Install [Docker](https://docs.docker.com/desktop/setup/install/windows-install) and pip package `featurecloud`:

```
pip install featurecloud
```

2. Download the FedPAM image from FeatureCloud Docker repository using

```
featurecloud app download featurecloud.ai/fc-fedpam
```

3. OR build the app locally using:

```
featurecloud app build featurecloud.ai/fc-fedpam
```

## User Interface
The FedPAM app provides an interactive user interface to monitor local and global structures during the workflow, followed by an analyses of both structural and evaluation results. Moreover, it allows the user to modify the learned global DAG interactively and store the results in an expert-knowledge JSON file.
To run the interface, use FeatureCloud's dedicated UI button.

IMPORTANT: The workflow will run and finish as intended but will only be terminated when the coordinator clicks on the `Finish` button (top-right corner on coordinator's app UI page) or the `Stop` button in the Featurecloud workflow UI.


## Testing FedPAM Locally
To test FedPAM on locally stored datasets and simulate the federated learning workflow, you can use the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also use CLI to run the app:

```
featurecloud test start --app-image featurecloud.ai/fc-fedpam --client-dirs './clients_datasets/clients_03/client1,./clients_datasets/
clients_03/client2,./clients_datasets/clients_03/client3' --generic-dir './generic'
```

<b>Important</b>: Keep the shared `config.yml` file in the `generic` directory.

The results of tests will be stored in `fc-fedpam/data/tests`.
