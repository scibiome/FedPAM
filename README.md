# FedPAM 
FedPAM is a federated framework for discrete Bayesian network learning using Probabilistic Adjacency Matrices (PAMs).

### Datasets Used
1. **[Chronic Kidney Disease (CKD) prediction dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease):** 
   * Contains 400 samples with 24 discrete medical, laboratory, and demographic variables for **binary classification** of disease presence.
   * Stored in directory `data/ckd_400` and is split into $K=3$ homogeneous client datasets.




2. [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success): 
    * Contains 4,424 samples with
36 variables for <b>multi-class classification (3 classes) </b> of academic outcomes based on enrollment
information and first- and second-semester performance.
    * Stored in `data/clients_datasets` and is split into $K = 3$ to $10$ client datasets for both homogeneous and heterogeneous settings.

### Config File

Modify the hyperparameters in `config.yml` file based on your requirements.

```
fc-fedpam:
  input:
    dataset_loc: "client.csv"
    target: 'class'
  split:
    mode: "file"
    dir: "."  
  mu: 0.3 
  lam: 0.3
  bootstrap_iterations: 100
  bootstrap_min_iterations: 5 
  bootstrap_patience: 5
  max_iterations: 100
  fl_min_iterations: 5 
  fl_patience: 5     
  homogeneous: True
```

#### Description of Hyperparameters:

1. `dataset_loc`: Location of the csv file containing the discrete dataset. 
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

2. `target`: Set this to the prediction variable in the dataset. For CKD-400, use 'class' and for the students success prediction dataset, use 'Target'.

3. `mode`: Controls how the app finds data splits. If set to `mode: 'directory'`, the app looks for subdirectories inside a base folder to use as separate client data splits. Otherwise, it uses the main `/mnt/input` directory as the single split. During testing, you can change client data directories using the FeatureCloud test-bed/workflow interface.

4. `dir`: The base directory (relative to `/mnt/input`) that contains subdirectories for each client's data split. 

5. `mu`: Hyperparameter $\mu$ for tuning the proximal term $\frac{\mu}{2} \lVert P - P_{\text{global}} \rVert _F^2$ that encourages alignment with the global consensus PAM.

6. `lam`: Hyperparameter $\lambda$ for tuning the inner product between normalized Conditional Mutual Information (CMI) matrix $C$ and the PAM $P$ being optimized. The term $-\lambda \langle C, P\rangle$ promotes edges with high CMI.

7. `bootstrap_iterations`: Number of bootstrapping iterations $B$ for resampling over the dataset to create local PAM.

8. `bootstrap_min_iterations`: Minimum number of bootstrapping iterations $B_{min}$.

9. `max_iterations`: Total number of federated learning iterations.

10. `fl_min_iterations`: Minimum number of federated learning iterations.

11. `fl_patience`: Number of patience iterations for early stopping of the federated learning process if the average BIC score across clients does not improve for these many iterations.

12. `homogeneous`: Boolean hyperparameter to switch between homogeneous and heterogeneous learning modes. If the existing client data is "known" to be homogeneous, set `homogeneous: true`. Otherwise, set `homogeneous: False`. In fact, in real-world scenarios, keeping the latter is suggested as the client distributions are usually unknown.

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

## Testing FedPAM Locally
To test FedPAM on locally stored datasets and simulate the federated learning workflow, you can use the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also use CLI to run the app:

```
featurecloud test start --app-image featurecloud.ai/fc-fedpam --client-dirs './clients_datasets/clients_03/client1,./clients_datasets/
clients_03/client2,./clients_datasets/clients_03/client3' --generic-dir './generic'
```

<b>Important</b>: Keep the shared `config.yml` file in the `generic` directory.

The results of tests will be stored in `fc-fedpam/data/tests`.
