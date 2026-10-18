# FRHG

Implementation of **FRHG: Fuzzy-Rough Hypergraphs for Rare Failure Discovery in High-Dimensional Manufacturing Data**.

## Installation

Requires Python >=3.11.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1; neural entries use the installed PyTorch 2.7.1+cu118. Checks cover Python syntax, TOML/JSON and CLI options against parser definitions. Training and evaluation were not run.

## Data and preprocessing

Obtain [SECOM](https://www.archive.ics.uci.edu/dataset/179/secom) or use manufacturing records with declared production groups. Export `manufacturing.csv` with `label` (common 0 / rare 1), `batch`, `split` (`train,dev,cal,test`), numeric features and optional actual mechanism columns prefixed `meta_`. A complete batch belongs to one partition. Missing numeric values may be blank; preparation fits imputation/scaling on train only.

```bash
frhg prepare --source manufacturing.csv --output data/manufacturing.npz
```

## Training and inference

```bash
frhg train --data data/manufacturing.npz --output runs/full --epochs 200 --seed 0
```

## Evaluation

```bash
frhg evaluate --data data/manufacturing.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `mlp`, `hgnn`, `no_qualification`, `no_rough`, `data_only`, `random_edges`. Keep partition, reference bank budget and seed fixed.

```bash
frhg train --data data/manufacturing.npz --output runs/no_rough --variant no_rough --epochs 200 --seed 0
frhg evaluate --data data/manufacturing.npz --model runs/no_rough/model.pt --output runs/no_rough/evaluation
```

## Outputs

`model.pt` stores the model, reference bank and qualified hyperedges. `training.json` records objectives and development AUPRC. Evaluation writes `metrics.json`, per-query `decisions.json` and `boundary.png`.
