# Data

This project uses the **Electric Motor Temperature (PMSM)** dataset.

Source:
https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature

The dataset is not included in this repository due to size constraints.

Expected structure:

data/
├── raw/
│ └── measures_v2.csv
├── parquet/
│ ├── XA_full.parquet
│ ├── XB_soft.parquet
│ ├── XB_strict.parquet
│ └── regimes_kmeans_XB_soft.parquet
├── processed/
│ └── histgb_best.joblib
