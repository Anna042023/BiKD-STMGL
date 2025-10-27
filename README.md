# BiKD-STMGL
This repository contains the official implementation of the paper "Efficient Lane-Level Traffic Flow Prediction via Bidirectional Knowledge Distillation with Spatio-Temporal Memory Graph Learning".
## Method
The Bidirectional Knowledge Distillation with Spatio-Temporal Memory Graph Learning (BiKD-STMGL) framework addresses core challenges in both spatio-temporal and graph data management for lane-level traffic flow prediction. This integrated solution balances computational efficiency with model adaptability through three key components that advance spatio-temporal graph processing capabilities.

- The framework initiates with an Enhanced Spatio-Temporal Embedding Layer, where multi-granularity temporal networks extract transient and trend patterns while Chebyshev-based graph embeddings encode lane topology through graph-based structural encoding. This establishes the foundational representation for subsequent spatio-temporal analysis.

- The core Spatio-Temporal Memory Graph Learning (STMGL) module then employs coordinated graph learning techniques: multi-granularity networks for temporal pattern extraction, memory-augmented spatial learning for dynamic dependency modeling in graph structures, and soft-DTW-enhanced attention for behavior-aware correlation discovery in spatial networks.

- The Bidirectional Knowledge Distillation (BiKD) mechanism completes the framework by employing large language models as traffic specialists to establish iterative refinement cycles. Through sequence consistency evaluation, teacher-generated guidance systematically optimizes student model parameters, enhancing prediction accuracy and generalization across diverse traffic scenarios while maintaining efficient graph data management.

Empirical results confirm that BiKD-STMGL effectively handles complex spatio-temporal dependencies in graph-structured traffic data, achieving superior prediction accuracy with sustained computational efficiency. The framework successfully addresses key challenges in spatio-temporal and graph data management, providing a robust solution for lane-level prediction in complex transportation networks.

## Datasets
Three datasets are used in this work, stored in separate folders: `PeMS`, `HuaNan`, and `PeMSF`.
Each dataset contains the following files:

- `train.npz`: training set

- `val.npz`: validation set

- `test.npz`: testing set

- `adj.csv`: describing the spatial connectivity between lane segments.

## Code Structure
The implementation framework of BiKD-STMGL consists of four core files:

- `embedding.py:` Implements three embedding modules: temporal position embedding, temporal periodic embedding, and spatial Chebyshev graph embedding.

- `STMGL.py:` Defines the main architecture of the STMGL network.

- `teacher_model.py:` Implements the teacher model.

- `model_BiKD_STMGL.py:` The main script that integrates the student and teacher models to perform BiKD training. It outputs evaluation metrics including MAE, RMSE, and MAPE for each prediction horizon.

## Usage
### Requirements
- numpy
- pandas
- torch
- math
- os
- time
- openai
- json
  
## Running Experiments
You can run the experiments by executing `model_BiKD_STMGL.py`.

## Baseline Models
The comparative experiments include two categories of baseline methods:

- Lane-Level Models — Models designed for lane-level traffic flow prediction. These methods are stored in the `Lane_Level_models`.

- Graph-Based Models — Spatio-temporal graph learning approaches that capture spatial dependencies between road segments. These methods are stored in the `ST_Graph_models`.

## Experimental Results
The comparison between BiKD-STMGL and other baseline comparison methods is based on the following metrics:

- MAE (Mean Absolute Error)

- RMSE (Root Mean Square Error)
  
- MAPE (Mean Absolute Percentage Error)
  
- SCI (Symmetric Consistency Index)
