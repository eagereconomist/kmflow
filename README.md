# kmflow

Modular data segmentation using PCA and K-Means in Python

## Overview

**kmflow** is a lightweight Python package and CLI toolset for exploring, segmenting, and visualizing tabular data via principal-component analysis (PCA) and K-Means clustering. It also ships with a Streamlit dashboard (`app/app.py`) for interactive analysis.

Key features:

- **CLI workflows** for preprocessing, clustering, PCA, evaluation & plotting  
- **Streamlit dashboard** for point-and-click analysis and download of results  
- **Modular utils** for wrangling, scaling, PCA summary, cluster prep, benchmarks  
- **Cookiecutter-inspired layout** that’s been customized for clarity and flexibility  

---

## Quickstart

1. **Clone the repo**

   ```bash
   git clone git@github.com:you/kmflow.git
   cd kmflow
   ```

2. **Create & activate a virtual environment**

   ```bash
   # Create venv
   $ python -m venv .venv

   # Activate
   source .venv/bin/activate    # macOS / Linux
   source .venv/Scripts/activate # Windows
   ```

3. **Install kmflow package** *(editable mode)*
   ```bash
   pip install -e .
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt

   # If pip isn't installed, run this and then install dependencies
   pip install -U pip
   ```

5. **Run the dashboard if you want a more user-friendly experience** *(Optional)*

   ```bash
   cd app
   streamlit run app.py
   ```
   This will launch and host the dashboard on your local machine. Alternatively, you can skip
   hosting it locally and use the hosted version by following this link:
   [K-Means Clustering Dashboard](https://kmeans-dash.streamlit.app/)

   **If you see this message** after clicking the link directly above: 

   ```
   This app has gone to sleep due to inactivity. Would you like to wake it back up?
   ```

   **It's completely fine to click:**

   ```
   Yes, get this app back up!
   ```

   At this time, I'm not paying for a streamlit account and the free version doesn't allow me to 
   keep the app up for very long if no one is using it.

6. **Check out the *Description* below for a detailed breakdown and examples of how to use the CLI
   if you'd like to go beyond the dashboard experience.**

   
## Project Structure

```
├── app/                          
│   └── app.py                   # Streamlit dashboard entrypoint  
├── docs/                        # A default mkdocs project
│   └── mkdocs.md  
├── kmflow/                      # Python package  
│   ├── cli/                     # Command-line interface  
│   ├── utils/                   # Helper modules (wrangle, scale, PCA, plots, eval)  
│   └── config.py                # Global constants & defaults  
├── main/                        
│   └── entrypoint.py            # Console‐script entry point: builds the `kmflow` Typer app
|                                  by importing and registering all subcommands from kmflow/cli 
├── tests/                       
│   └── test_data.py             # Basic unit tests  
├── LICENSE                      
├── Makefile                     # `make data`, `make train`, etc.  
├── pyproject.toml               # Packaging + tool config (black, isort, etc.)  
├── README.md                    # You are here  
└── requirements.txt             # Pinned dependencies  

```
--------

## Description

### `__init__.py`

- Three different files
  - Top-level
  - util
  - CLI

| File                                  | Purpose                                                      |
|---------------------------------------|--------------------------------------------------------------|
| `kmflow/__init__.py`                  | Package initializer—defines `__version__`, exposes top-level API and makes `kmflow` importable. |
| `kmflow/cli/__init__.py`              | CLI entry-point setup—imports and registers all subcommands into a single Typer app.           |
| `kmflow/utils/__init__.py`            | Utility namespace—collects and exposes core helper functions (`wrangle`, `kmeans`, `pca`, etc.) for easy import. |

### `kmflow/cli/wrangle.py` & `kmflow/utils/wrangle_utils.py`

- **Purpose:** basic data-wrangling and IQR-based outlier handling as a prelude to scaling and clustering.  

- **CLI (`wrangle.py`):** create flags to invoke any custom cleaning steps you create and use built-in flags to detect or remove interquartile-range outliers.  
  
- **Utils (`wrangle_utils.py`):** core functions for:
  - simple cleaning helpers (e.g. drop columns, drop rows)  
  - IQR outlier detection & optional removal  
  
- **Extensibility:** these modules form a scaffold—you’re expected to add dataset-specific wrangling functions (e.g. custom parsers, imputation routines) and register them in the CLI so unique “dirty” cases can be handled.  
  
- **When to use:** once your tabular data is in the general shape you need it to be in (e.g. no missing values, etc.), go ahead and run the IQR outlier & removal processes.

#### **Examples on how to use `wrangle.py` in bash:**

**Note:** `export-outliers` and `--remove-outliers` cannot be used in the same command. 

- Export outliers into a CSV file using raw data:

```bash
cat data/raw/raw_data.csv \
    | kmflow wrangle outlier \
    --export-outliers \
    > exported_outliers.csv
```

You can then expect to see a new CSV file inside your project by the same name you assigned. Upon opening the CSV file, you should see something like this (different column names, row_index, and outlier_value depending on your data):

```
row_index,column,outlier_value
2,headsize,125.0
7,balance,-0.375
23,length,27.6
```

- Remove outliers and export reduced data into a predetermined destination for holding results:
  
```bash
cat data/raw/raw_data.csv \
    | kmflow wrangle outlier \
    --remove-outliers \
    > data/interim/removed_outliers.csv
```

### `kmflow/cli/plots.py` & `kmflow/utils/plots_utils.py`

- **Purpose:** A suite of diagnostic and exploratory visualizations for clustering and PCA results, from 2D/3D biplots, to inertia and silhouette charts.
  
- **CLI (`plots.py`):** Exposes commands like (`biplot`, `3d-biplot`, `inertia`, `silhouette`, etc.). Reads input via stdin, accepts plot-specific flags (e.g. `--hue`, `--skip-scores`, etc.) and writes output to disk using a properly named `.png`, or display the plot interactively (when `--no-save` is used).  
  
- **Utils (`plots_utils.py`):** Core plotting functions (using matplotlib, seaborn, and plotly under the hood) that handle data formatting, layout/styling, and figure saving. Includes shared helpers for colors, themes, and axis formatting.  

#### **Examples on how to use `plots.py` in bash:**

- Create and display an interactive histogram (not saved to disk):

```bash
cat data/processed/processed_data.csv \
    | kmflow plots histogram \
    <COLUMN> \
    --bins 50 \
    --no-save
```

- Create and write histogram to disk:
  
```bash
cat data/processed/processed_data.csv \
    | kmflow plots histogram \
    <COLUMN> \
    --bins 25 \
    > reports/figures/histogram.png
```

### `kmflow/cli/process.py` & `kmflow/utils/process_utils.py`

- **Purpose:** Apply configurable scaling and transformations (normalization, standardization, min–max, log1p, Yeo–Johnson) to prepare data for PCA and K-Means.
  
- **CLI (`process.py`):** Accepts flags for one or more transformations (e.g. `--standardize`, `--minmax`, `--log1p`, `--yeo-johnson`), and writes the transformed DataFrame to disk or stdout.
  
- **Utils (`process_utils.py`):** Implements core scaling functions, handles column selection and transformer fitting, and integrates with I/O helpers for seamless DataFrame processing.
   
- **When to use:** Run immediately after wrangling raw data to ensure features are on a comparable scale before PCA or K-Means clustering, ***if*** your data isn't already on a similar scale.

#### **Example on how to use `process.py` in bash:**

- Standardize data and write to disk:
  
  ```bash
  cat data/interim/preprocessed_data.csv \
      | kmflow process \
      --std \
      > processed_data.csv
  ```

### `kmflow/cli/pca.py` & `kmflow/utils/pca_utils.py`

- **Purpose:** Perform principal component analysis to reduce dimensionality and extract component scores, loadings, proportion of variance explained, and proportion of cumulative variance explained.
    
- **CLI (`pca.py`):** Accepts flags like `--n-components <int>`, and writes a folder containing the PCA summary CSV files.
  
- **Utils (`pca_utils.py`):** Implements PCA decomposition (using scikit-learn), computes scores, loadings, proportion of variance, and cumulative variance explained.
  
- **When to use:** After preprocessing (wrangling and scaling) — and optionally clustering — to explore principal components and feed them into downstream analyses.

#### **Example on how to use `pca.py` in bash:**

- Running PCA with a seed, on 3 specific columns, in a processed dataset and writing to disk:  

  ```bash
  cat data/processed/DIR/std.csv \
      | kmflow pca \
      --seed 3429 \
      --numeric-cols "<COLUMN>, <COLUMN2>, <COLUMN3>" \
      --n-components 3 \
      > processed_pca.csv
  ```

### `kmflow/cli/kmeans.py` & `kmflow/utils/kmeans_utils.py`

- **Purpose:** Fit K-Means clustering to your data, assigning cluster labels and optionally running batch fits over multiple `k` values.
- 
  **CLI (`kmeans.py`):** Accepts clustering flags (`--n-clusters`, `--init-method`, `--seed`, etc.), and writes the labeled DataFrame to disk or stdout.

- **Utils (`kmeans_utils.py`):** Core functions that wrap scikit-learn’s KMeans, handle single—or multi-`k` fits, manage random-state reproducibility, and append cluster labels to the DataFrame.
    
- **When to use:** Run after data has been wrangled and scaled (optional) to segment into groups before evaluation or plotting.

#### **Example on how to use `kmeans.py` in bash:**

- Running and writing k-means to disk on a processed dataset, using 8 clusters and a seed, with the default scikit-learn KMeans initialization method:
   
  ```bash
  cat data/processed/std_outliers/std.csv \
      | kmflow kmeans fit-km \
      8 \
      --seed 123 \
      --init k-means++ \
      > clustered_8.csv
  ```

  - Running and writing k-means to disk on a processed dataset, using 8 clusters and a seed, with the algorithm set to `elkan`, initialization method as `random`, and number of initializations (`--n-init`) set to 100 iterations:
  
  ```bash
  cat data/processed/std_outliers/std.csv \
      | kmflow kmeans fit-km \
      8 \
      --seed 9250 \
      --algorithm elkan \
      --init random \
      --n-init 100 \
      > data/processed/std_outliers/algo_elkan_init_random/clustered_8.csv
  ```

### `kmflow/cli/evaluation.py` & `kmflow/utils/evaluation_utils.py`

- **Purpose:** Compute/benchmark clustering evaluation metrics (inertia, silhouette score, Calinski–Harabasz index, Davies–Bouldin score).
  
- **CLI (`evaluation.py`):** Accepts similar flags from `kmeans.py`. 
  
- **Utils (`evaluation_utils.py`):** Core functions that calculate each metric, assemble them into a tidy DataFrame, and hand off to I/O helpers.
  
- **When to use:** Immediately after all clustering is finished to generate a quantitative report (see "Benchmarking Multiple Metrics" section below).

#### **Example on how to use `evaluation.py` in bash:**

- Run and write a silhouette score analysis using scikit-learn KMeans on a model using a seed, `elkan` algorithm, `random` initialization, and number of iterations (`--n-init`) set to 50 iterations: 
  
  ```bash
  cat data/processed/std_outliers/std.csv \
      | kmflow evaluate silhouette \
      --seed 2985 \ 
      --algorithm elkan \
      --init random \
      --n-init 50 \
      > data/processed/std_outliers/algo_elkan_init_random/std_silhouette.csv
  ```

  ---

### Benchmarking Multiple Metrics inside `evaluation.py`
  
  **kmflow** also provides a `benchmark` command inside `evaluation.py` to find Calinski–Harabasz and Davies–Bouldin CSVs, merge them into one table, and write to stdout or disk. It uses this *regex* to discover result folders:

  ```python
  r"^algo_([^_]+)_init_(.+)$"
  ```

#### Directory & file naming conventions
  ```
  data/processed_root/
  └── <variant>/                      
      └── kmeans_<pipeline>/          
          └── algo_<algorithm>_init_<init>/  
              ├── *_calinski.csv  
              └── *_davies.csv
  ```

 - `<variant>` = your processing variant (e.g. `std`, etc.)
  
 - `algo_<algorithm>_init_<init>` *must* match the regex above
  
 -  **Clustering evaluation metric files** *must* end in `_calinski.csv` or `_davies.csv`

#### Merge via the CLI

  ```bash
  kmflow evaluate benchmark <input_dir> [flag]
  ```

- `<input_dir>` = subfolder under `data/` (e.g. `processed`)
  
- `--decimals` = round metric values (default 3 decimal places)

#### **Example on how to use `benchmark` in bash:**

- Evaluate and write to stdout:
  
  ```bash
  kmflow evaluate benchmark processed \
  --decimals 4
  ```

- Write to disk

  ```bash
  kmflow evaluate benchmark processed \
  > benchmark.csv
  ```

### `kmflow/cli/cluster_prep.py` & `kmflow/utils/cluster_prep_utils.py`

- **Purpose:** Merge raw and clustered DataFrames to produce per-cluster summary statistics and label counts.
    
- **CLI (`cluster_prep.py`):**  
  - `cluster-profiles`: reads a raw CSV and a clustered CSV, accepts `--cluster-col` (and optional `--key-col` to join on a key column), and writes a per-cluster profile table.
    
  - `map-clusters`: reads a clustered CSV, prompts interactively to map integer IDs to human labels, counts each label, and writes the counts table.
    
- **Utils (`cluster_prep_utils.py`):**  
  - `merge_cluster_labels()` aligns raw & clustered rows by index (or key column).  
  - `get_cluster_profiles()` computes summary statistics (mean, median, min, max) per cluster.  
  - `clusters_to_labels()` and `count_labels()` map IDs to labels and tally counts.
  
- **When to use:** After K-Means clustering, to turn raw + cluster outputs into interpretable summaries and human-readable labels.

### Why Prep My Clusters?

If your data is not on a similar scale, K-Means and PCA won't perform well, but scaled data loses its original units—making it hard to interpret clusters in context. 
The `cluster_prep.py` tools let you reattach cluster assignments to the *unscaled* raw data so you can:

- Compute summary statistics in the original measurement units  
- Label clusters meaningfully based on real-world values  
- Count and compare cluster sizes accurately  
- Preserve the benefits of scaling for analysis while keeping outputs interpretable

#### **Example on how to use `cluster_prep.py` in bash:**

**Note**: While stdin -> stdout piping is supported, I *highly* recommend using explicit file arguments as shown below.

- Write to a file

  ```bash
  kmflow cluster-prep cluster-profiles \
    data/raw/DIR/example.csv \
    data/processed/DIR/variant/algo_<algorithm>_init_<init>/std_clustered_5.csv \
    cluster_5 \
    > data/external/cluster_profile.csv
  ```

- Write to stdout 

  ```bash
  kmflow cluster-prep cluster-profiles \
    data/raw/DIR/example.csv \
    data/processed/DIR/variant/algo_<algorithm>_init_<init>/std_clustered_5.csv \
    cluster_5
  ```
