# FRHG

Implementation of **FRHG: Fuzzy-Rough Hypergraphs for Rare Failure Discovery in High-Dimensional Manufacturing Data**.

FRHG qualifies mechanism/data hyperedges using batch-bootstrap membership stability and positive held-out batch-controlled label information. A residual hypergraph encoder propagates qualified evidence, and class-conditional fuzzy-rough bounds govern selective rare-failure decisions. The propagation affinity retains its original diagonal; only the rough relation receives the paper's exact unit self-anchor. Unsupported nodes are rejected even when their rough interval has zero width.

## Install and data

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11+. Obtain SECOM from the UCI Machine Learning Repository, Bosch Production Line Performance through its Kaggle competition, or Tennessee Eastman process traces from the selected public release. Retain production order and the original batch/line/run metadata; a random row split is not a batch-shift evaluation.

Export `manufacturing.csv` with `label` (common 0 / rare 1), `batch`, `split` (`train`, `dev`, `cal`, `test`), numeric feature columns, and optional categorical mechanism columns whose names begin `meta_`. Missing numeric cells may be blank. A production batch may occur in only one partition. Mechanism columns represent actual stage/chamber/recipe/subsystem groupings when those exist; do not invent them for datasets without those fields.

```bash
frhg prepare --source manufacturing.csv --output manufacturing.npz
frhg train --data manufacturing.npz --output outputs/full --epochs 200 --seed 0
frhg evaluate --data manufacturing.npz --model outputs/full/model.pt --output outputs/full_eval
```

Preparation drops training-all-missing features, imputes training medians, and standardizes with training statistics. Training builds at most 2000 reference vertices using class-stratified training sampling; increase `--references` for a larger bank. Test prevalence is unchanged. Numeric hyperedge proposals cluster correlated feature groups after training batch-mean residualization, then place K-means-centered radius regions in original standardized feature space. The radius caps typical candidate size at `--maximum-fraction`, default 0.2. Oversized mechanism groups are omitted. Mechanism membership predicates are fixed and have structural stability one; data-edge stability is mean bootstrap Jaccard over resampled whole batches.

Qualification subtracts within-validation-batch label-permutation information from observed conditional mutual information. Defaults are 30 bootstrap fits, 30 permutation controls, stability at least 0.5, and positive information above 0.001. Candidate geometry and features use training only; `dev` labels qualify/select the bank and checkpoint, `cal` labels lock the decision layer, and test labels only score outputs.

## Model, controls, and acceptance

```bash
frhg train --data manufacturing.npz --output outputs/mlp --variant mlp
frhg train --data manufacturing.npz --output outputs/hgnn --variant hgnn
frhg train --data manufacturing.npz --output outputs/no_qualification --variant no_qualification
frhg train --data manufacturing.npz --output outputs/no_rough --variant no_rough
frhg train --data manufacturing.npz --output outputs/data_only --variant data_only
frhg train --data manufacturing.npz --output outputs/random --variant random_edges
```

The encoder has two 32-wide residual vertex/edge propagation layers, inverse-batch-frequency normalization, nonnegative learned edge weights, and a binary head. The objective uses balanced classification loss, 0.1 lower-support loss on rare references, 0.1 capped-boundary-weighted loss with cap 0.5, 0.001 edge sparsity, and 0.1 Brier loss. Adam uses 0.001. `mlp` disables edges, `hgnn` admits every candidate and omits rough loss, `no_qualification` admits every candidate while retaining rough loss, `no_rough` removes rough loss and rough-based rejection, and `data_only` omits mechanism candidates. MLP/HGNN/no-rough controls calibrate probability-margin rejection without imposing FRHG's lower/support gate. `random_edges` permutes training incidence rows while preserving edge cardinalities, then assigns each query its nearest reference's randomized incidence.

The finite implementation uses dense incidence/affinity matrices on the bounded reference set. At inference, query messages use reference hyperedges, query/query communication is excluded, and the exact query self-membership anchors both rough bounds. Thus other test labels or prediction batches never determine a query's membership. This differs from a large-scale approximate-neighbor backend but retains the corrected operators and structural support rule.

For qualified variants, current-batch data edges also pass an unlabeled membership-stability check: ten bootstraps of the batch's currently admitted members recompute the local center, and mean membership Jaccard must pass the stored stability threshold. Fewer than two current members cannot support a data edge. Mechanism predicates retain their fixed membership semantics. This batchwise check uses observations and batch IDs without any current labels.

Calibration fits error risk from probability margin and rough boundary width, locks its 90th-percentile validation operating point, and fixes the rare alarm threshold from the 95th percentile of common calibration probabilities. Rare acceptance also requires lower support above the calibration rare-class 10th percentile and qualified nonself support. Discrete calibration-risk ties or support rejection can change realized coverage; the actual coverage and common/rare denominators are reported separately.

## Outputs

`model.pt` contains the learned model, training reference features/incidence, qualified bank, and metadata predicates. It is excluded from Git. `training.json` records dev AUPRC, losses, elapsed training time, and traced Python allocations. `metrics.json` reports rare AUPRC, rare recall at the validation-fixed 5% common false-alarm operating point, actual test common false-alarm rate, Macro-F1, binary ECE, coverage, selective error, bootstrap hyperedge Jaccard, and qualified edge count. Memory is Python allocation memory, excluding native tensor storage.

`decisions.json` stores base probability, lower/upper bounds, structural support, acceptance, supporting edges, and lower-membership changes after removing each supporting edge while freezing probabilities. `boundary.png` displays calibrated risk against boundary width. For prevalence experiments subsample training rare/common rows only, retaining all test rows. For corruption/shift comparisons use paired CSVs with registered batch and mechanism changes. Repeat five seeds and three production partitions. No experiment scores are bundled.
