# ATR-Tool-Compound-Discovery-Engine
Dataset-backed, reproducible active-learning pipeline for prioritizing ATR kinase tool compounds, includes data curation, model training, evaluation, and iterative candidate selection, proof-of-concept for prioritizing **ATR kinase** tool compounds. This repository curates ligand–activity records, trains baseline models, evaluates performance with leakage-resistant splits, and demonstrates an iterative **active-learning** loop to nominate the next compounds for experimental follow-up.

## Why ATR
ATR is a central replication-stress response kinase, cancer cells often rely on ATR to survive chronic DNA replication stress. This makes ATR a high-value, data-rich target for computational compound prioritization and ML benchmarking.

## What this repo contains
- **Notebook-first pipeline** for end-to-end reproducibility.
- **Data curation** utilities (structure cleanup, unit normalization, deduplication).
- **Model training** (baseline fingerprint models, extendable to GNN/SMILES models).
- **Evaluation** with scaffold-aware splitting to reduce overly optimistic metrics.
- **Active learning** loop (train → score pool → pick candidates → repeat).
- **Documentation artifacts** (Model Card, Data Sheet) describing intended use and limitations.

## Repository structure
```text
.
├── data/
│   ├── raw/                # downloaded or exported source files (immutable)
│   ├── interim/            # intermediate cleaned tables
│   └── processed/          # final ML-ready datasets
├── notebooks/
│   └── ATR Tool-Compound Discovery Engine, Dataset-Backed Active Learning Proof-of-Concept.ipynb
├── src/
│   ├── data/               # parsing, cleaning, standardization
│   ├── features/           # featurization (ECFP, physchem, scaffolds)
│   ├── models/             # training, calibration, inference
│   └── active_learning/    # acquisition functions and loop orchestration
├── reports/
│   ├── figures/            # exported plots
│   └── tables/             # exported summary tables
├── modelcard.md
├── datasheet.md
├── LICENSE
└── README.md
```
## Quickstart

### 1) Create an environment
```
Use any modern Python (3.10+ recommended).
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
### 2) Add your raw BindingDB export
Place your exported files into:
```
data/raw/
```
### Recommended columns (names may vary by export):
	•	target name / UniProt / gene symbol (ATR)
	•	SMILES (or InChIKey + SMILES)
	•	activity type (IC50 / Ki / Kd)
	•	activity value + units
	•	assay metadata (optional but helpful)

### 3) Run the notebook

Open and run:
```
notebooks/ATR Tool-Compound Discovery Engine, Dataset-Backed Active Learning Proof-of-Concept.ipynb
```
### The notebook will:
	1.	curate and standardize activity data (nM, optional pIC50),
	2.	deduplicate by InChIKey,
	3.	build scaffold-aware splits,
	4.	train baseline models,
	5.	run an active-learning acquisition cycle,
	6.	export tables/figures to reports/.

## Method overview
### Data curation
	•	Standardize units (e.g., nM), optionally convert to pActivity.
	•	Remove salts/invalid structures where possible.
	•	Deduplicate compounds by InChIKey, aggregate replicate measurements (median).
	•	Keep provenance fields to support auditing.

### Labels (classification mode)
A practical, low-noise default:
	•	Active: IC50 ≤ 1 µM
	•	Inactive: IC50 ≥ 10 µM
	•	Drop ambiguous middle range (or use regression mode instead)

### Splitting and evaluation
	•	Prefer scaffold split (Bemis–Murcko) over random split.
	•	Report AUROC and AUPRC, plus calibration where probabilities are used for triage.
	•	Keep a simple baseline (e.g., logistic regression or XGBoost on ECFP) as a sanity check.

### Active learning

A minimal active-learning cycle:
	1.	Train model on labeled set.
	2.	Score an unlabeled pool.
	3.	Select the next compounds using an acquisition rule (e.g., uncertainty + diversity).
	4.	“Acquire” labels (simulated in this PoC, replace with real assays in practice).
	5.	Repeat.

## Outputs

Typical exports (names may differ depending on configuration):
	•	reports/tables/: curated dataset summaries, split statistics, metrics
	•	reports/figures/: ROC/PR curves, calibration, learning curves, acquisition diagnostics
	•	Optional model artifacts in models/ or reports/ depending on your setup

## Intended use

### This project is intended for:
	•	tool-compound prioritization,
	•	benchmarking ML workflows for kinase inhibitor discovery,
	•	educational demonstrations of active learning on real bioactivity tables.

It is not a substitute for experimental validation, clinical decision-making, or safety/toxicity assessment.

## Reproducibility
	•	The notebook is the authoritative execution path.
	•	Keep data/raw/ immutable and version your processed datasets.
	•	Record dataset version, model version, and random seeds in outputs.

## See modelcard.md and datasheet.md for documentation of limitations and ethical considerations.

## License
Released under the open-source license 

## Citation
Petalcorin, M.I.R (2026). GitHub Repository. https://github.com/mpetalcorin/ATR-Tool-Compound-Discovery-Engine 

# Model Card: ATR Tool-Compound Discovery Engine

## Model details
**Model name:** ATR Tool-Compound Discovery Engine (baseline predictive models)  
**Primary task:** Predict ATR kinase bioactivity from small-molecule structure  
**Model type:** Supervised learning, baseline fingerprint models (e.g., Logistic Regression, Random Forest, XGBoost), optional probability calibration  
**Input:** Small-molecule structures (SMILES), converted to molecular representations (e.g., Morgan/ECFP fingerprints, optional physicochemical descriptors)  
**Output:**  
- Classification mode, probability of being “active” vs “inactive”, optionally calibrated  
- Regression mode (optional), predicted pActivity (e.g., pIC50)  

**Intended target:** ATR (Ataxia telangiectasia and Rad3-related) kinase, human

## Intended use
This repository provides a reproducible, dataset-backed proof-of-concept to:
- Prioritize **tool compounds** for ATR, based on predicted likelihood of activity.
- Demonstrate an **active-learning** workflow, train → score pool → select candidates → iterate.
- Provide a practical template for kinase-focused ML pipelines using curated activity data.

**Primary users:** Computational chemists, ML practitioners in drug discovery, students, and researchers building screening triage workflows.

**Out-of-scope uses:**
- Clinical decision-making, patient stratification, or dosing recommendations.
- Safety/toxicology prediction, ADME selection, or regulatory-grade analysis.
- Claims of efficacy in vivo or in humans without experimental confirmation.

## Model inputs and outputs
### Inputs
- SMILES strings for candidate compounds
- Optional metadata fields if available (assay type, endpoint, units, provenance identifiers)

### Outputs
- **Classification:** `P(active)` where activity is defined using thresholds on IC50 (or another endpoint).  
- **Regression (optional):** predicted pActivity (e.g., pIC50), intended for ranking rather than absolute potency claims.

## Training data
### Data source
Bioactivity records for ATR kinase were curated from a BindingDB-style export (or equivalent public ligand–activity table). Records typically include:
- target identity (preferably UniProt or curated target name)
- SMILES / compound identifiers
- endpoint type (IC50 / Ki / Kd)
- activity value and units
- optional assay metadata and publication identifiers

### Curation steps (typical)
- Unit normalization to a common scale (e.g., nM), optional conversion to pActivity.
- Structure standardization, removal of invalid structures and obvious salts where feasible.
- Deduplication by InChIKey, replicate aggregation (e.g., median pActivity).
- Filtering to **single-protein human ATR** records when possible.

### Labeling (classification mode)
A low-noise default:
- **Active:** IC50 ≤ 1 µM  
- **Inactive:** IC50 ≥ 10 µM  
- **Ambiguous:** 1–10 µM removed (or reserved for regression)

> Note: Thresholds are configurable. The chosen thresholds trade coverage for reduced label noise.

## Evaluation
### Recommended split strategy
This project prioritizes leakage-resistant evaluation:
- **Scaffold split** (Bemis–Murcko) is preferred over random split to reduce structural leakage.
- Train/validation/test splits are recorded in exported tables.

### Metrics
Common metrics for classification include:
- AUROC
- AUPRC (recommended when actives are a minority)
- Calibration diagnostics (when probabilities guide triage)
For regression (if used):
- MAE/RMSE on pActivity
- Rank correlation (Spearman) for prioritization fidelity

### What “good” looks like here
- Performance should remain credible under scaffold splits.
- Calibration should be reasonable if outputs are used as probabilities.
- Error analysis should show understandable failure modes, rather than spurious shortcuts.

## Limitations
- **Assay heterogeneity:** BindingDB-style tables combine multiple assay formats and conditions, which introduces noise and may limit the meaning of absolute potency predictions.
- **Target mapping noise:** Target names can be ambiguous without curated UniProt mapping, mis-mappings can degrade model quality.
- **Chemical space bias:** Training data reflects historically explored chemotypes, predictions may be unreliable for novel scaffolds.
- **No safety/ADME:** The model does not predict toxicity, selectivity, permeability, metabolism, or off-target effects.
- **No clinical interpretation:** Even strong predicted activity does not imply therapeutic utility.

## Bias, risks, and mitigations
### Potential biases
- Publication and reporting bias toward potent, well-studied chemotypes.
- Overrepresentation of certain inhibitor classes and kinase-like scaffolds.
- Missing negatives, many datasets report actives more readily than inactives.

### Risks
- Overconfidence in predictions for novel compounds.
- Misuse as a “hit confirmation” tool rather than a triage aid.
- Dataset leakage inflating apparent performance if random splits are used.

### Mitigations in this repo
- Prefer scaffold splits and deduplication.
- Retain provenance fields for auditability.
- Encourage calibration and conservative ranking-based use.
- Document intended use and limitations clearly.

## Ethical considerations
- This model is designed to support research workflows and does not incorporate patient data.
- Outputs should be used to guide hypothesis generation, not to justify claims of clinical efficacy.
- Users should disclose dataset provenance and any filtering decisions when reporting results.

## Usage guidance
- Use classification probabilities as **ranking signals**, not as certainty.
- Apply applicability domain checks when possible (e.g., similarity to training set).
- Validate top-ranked compounds experimentally.
- Consider multi-objective filters (e.g., novelty, diversity) alongside predicted potency.

## Model maintenance
- Retraining is recommended when new ATR bioactivity data is added, or when the curation logic changes.
- Track versions for:
  - dataset snapshot
  - featurization parameters
  - model hyperparameters and random seeds
- Export artifacts and metrics with timestamps and commit hashes when possible.

## Contact
Maintained by **Mark I.R. Petalcorin**.

# Data Sheet: ATR Tool-Compound Discovery Engine Dataset

## 1. Dataset name
**ATR Tool-Compound Discovery Engine Dataset**

## 2. Dataset summary
A curated, ML-ready dataset of small-molecule bioactivity measurements against **human ATR kinase** assembled from BindingDB-style ligand–activity records. The dataset supports reproducible model training, evaluation with scaffold-aware splits, and an active-learning proof-of-concept for prioritizing ATR tool compounds.

## 3. Motivation
### 3.1 Why this dataset exists
ATR is a central replication-stress response kinase and a high-value oncology target with substantial public bioactivity coverage. This dataset was created to:
- enable reproducible benchmarking of cheminformatics ML models for ATR activity prediction,
- demonstrate a practical active-learning workflow,
- support hypothesis generation for tool-compound prioritization.

### 3.2 Intended audience
Computational chemists, ML practitioners in drug discovery, and students/researchers building dataset-backed triage pipelines.

## 4. Composition
### 4.1 Data instances
Each record corresponds to a **compound–target activity measurement**. Depending on curation settings, records may be:
- **measurement-level** (one row per reported measurement), and/or
- **compound-level aggregated** (one row per unique compound with aggregated potency).

### 4.2 Typical fields (column examples)
**Compound identifiers**
- `smiles`
- `inchikey`
- `compound_name` (if available)
- `source_compound_id` (if available)

**Target identifiers**
- `target_name` (ATR)
- `gene_symbol` (ATR)
- `uniprot_id` (recommended if available)
- `organism` (Homo sapiens, when filtered)

**Activity**
- `endpoint_type` (IC50 / Ki / Kd)
- `activity_value`
- `activity_units`
- `activity_nM` (normalized)
- `pActivity` (optional, e.g., pIC50)

**Assay and provenance (when present)**
- `assay_description`
- `assay_format`
- `temperature`, `pH` (rare, if present)
- `reference` (PMID/DOI or BindingDB ref)
- `source` / `source_url` (if exported)
- `curation_notes` (optional)

**Derived ML fields**
- `label` (classification)
- `split` (train/val/test)
- `scaffold` (Bemis–Murcko scaffold string or ID)
- `replicate_count`, `activity_spread` (optional)

### 4.3 Data source
The dataset is produced from a BindingDB export (or a BindingDB-like table) provided by the user. The repository retains:
- **raw** exports unchanged,
- intermediate cleaned tables,
- final processed ML-ready tables.

## 5. Data collection process
### 5.1 How data was obtained
Bioactivity measurements were collected from public ligand–target activity records. Records are included when they contain:
- a target mapping to ATR (preferably single-protein ATR),
- a structure representation (SMILES, and/or InChIKey),
- a quantitative activity value with units.

### 5.2 Who collected the data
The dataset is curated programmatically from public records by the repository pipeline. No manual experimental measurements are generated by this project.

### 5.3 Over what timeframe was it collected
The dataset reflects the content of the export at the time it was obtained. The exact snapshot is determined by the raw files in `data/raw/` and the repository commit history.

## 6. Preprocessing, cleaning, and labeling
### 6.1 Structure standardization
Typical steps include:
- remove invalid SMILES and malformed entries,
- optionally strip salts and small fragments where feasible,
- generate or validate InChIKey for deduplication.

> Note: Standardization is conservative to avoid introducing artifacts.

### 6.2 Unit normalization
All activity values are converted to a common unit (typically **nM**) when possible:
- `activity_nM` is computed from the reported value and units.
Optionally:
- convert to `pActivity = -log10(activity_M)` for modeling stability.

### 6.3 Deduplication and aggregation
When multiple measurements exist for the same compound:
- deduplicate by `inchikey`,
- aggregate replicate activities (default: **median**),
- record replicate counts and variability when available.

### 6.4 Labeling rules (classification mode)
Default thresholds (configurable):
- **Active (1):** IC50 ≤ 1 µM (≤ 1000 nM)
- **Inactive (0):** IC50 ≥ 10 µM (≥ 10000 nM)
- **Ambiguous:** intermediate measurements dropped or reserved for regression

If multiple endpoint types are included (IC50/Ki/Kd), the pipeline may:
- restrict to a single endpoint type for cleanliness, or
- include all and track `endpoint_type` explicitly.

## 7. Recommended uses
- ML benchmarking for ATR bioactivity prediction.
- Tool-compound prioritization, ranking-based triage.
- Active learning demonstrations, acquisition strategy comparisons.
- Educational use in cheminformatics workflows.

## 8. Uses that are discouraged
- Clinical decision-making, patient selection, or dosing.
- Claims of efficacy without experimental validation.
- Toxicology, safety, or ADME claims, not supported by this dataset.
- Off-target selectivity claims, dataset is target-centric.

## 9. Distribution
### 9.1 Where the dataset lives in the repo
- `data/raw/` contains immutable raw exports.
- `data/interim/` contains intermediate cleaned tables.
- `data/processed/` contains ML-ready datasets used by models.

### 9.2 Licensing
The dataset is derived from public records. Users must comply with:
- the source database terms for the raw export they use,
- the repository’s open-source license for code and derived artifacts.

If a user includes a proprietary export, they should not commit it publicly.

## 10. Dataset splits and evaluation guidance
### 10.1 Split strategy
To reduce data leakage and inflated performance:
- use **scaffold split** (Bemis–Murcko) for train/validation/test partitions.

### 10.2 Why scaffold split matters
Random splits can place close analogs of the same scaffold in both train and test, producing overly optimistic results. Scaffold splits better reflect prospective screening, where new chemotypes appear.

## 11. Known limitations
- **Assay heterogeneity:** conditions vary across measurements, adding noise.
- **Target mapping ambiguity:** ATR naming may be inconsistent without UniProt mapping.
- **Class imbalance:** actives may be overrepresented or underrepresented depending on filtering.
- **Chemical space bias:** dominated by historically explored inhibitor classes.
- **No selectivity data:** activity against ATR does not imply specificity vs other kinases.

## 12. Potential biases
- Publication/reporting bias toward potent compounds.
- Overrepresentation of certain chemical series.
- Missing “true negatives,” many databases under-report inactive results.

## 13. Privacy and sensitive information
This dataset contains **no patient data** and no personal identifiers. It is derived from public biochemical assay records.

## 14. Ethical considerations
- Predictions trained on this dataset should be used as hypothesis-generation tools.
- Experimental validation is required before any claim of biological or therapeutic effect.
- Users should be transparent about curation rules and thresholds when reporting outcomes.

## 15. Maintenance
### 15.1 Versioning
To support auditability, the project recommends recording:
- raw dataset filename(s) and acquisition date,
- preprocessing configuration,
- dataset version tags,
- commit hashes for results.

### 15.2 Updating the dataset
When new activity records are added:
- rerun preprocessing,
- regenerate splits,
- retrain and re-evaluate models,
- document changes in changelog or release notes.

