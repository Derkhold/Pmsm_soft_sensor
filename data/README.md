# Data Architecture & Acquisition

This project utilizes the **Electric Motor Temperature (PMSM)** dataset, which contains high-fidelity telemetry from a Permanent Magnet Synchronous Motor collected on a test bench.

**Dataset Source:** [Kaggle: Electric Motor Temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature)

### Version Control Constraints
Due to storage constraints and standard version control best practices for large datasets, the raw telemetry data and the subsequent serialized artifacts are not tracked in this repository. 

To replicate the environment locally, please download the raw data from the link above and ensure your local `data/` directory strictly adheres to the schema outlined below before executing the notebooks or training scripts.

### Expected Directory Schema

```text
data/
├── raw/
│   └── measures_v2.csv                 # Original Kaggle dataset (required for 01_eda & 02_feature_engineering)
├── parquet/
│   ├── XA_full.parquet                 # The complete feature matrix (unrestricted oracle boundaries)
│   ├── XB_soft.parquet                 # The standard Soft-Sensor deployment matrix
│   ├── XB_strict.parquet               # The hardware-constrained ablation matrix
│   └── regimes_kmeans_XB_soft.parquet  # Serialized thermodynamic state assignments (from Notebook 03)
├── processed/
│   └── histgb_best.joblib              # Serialized weights of the champion baseline model
└── README.md
