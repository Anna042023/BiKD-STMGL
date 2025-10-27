# BiKD-STMGL
This repository contains the official implementation of the paper "Efficient Lane-Level Traffic Flow Prediction via Bidirectional Knowledge Distillation with Spatio-Temporal Memory Graph Learning".
## Method


## Code Structure
The implementation framework of BiKD-STMGL consists of four core files:

- embedding.py: Implements three embedding modules: temporal position embedding, temporal periodic embedding, and spatial Chebyshev graph embedding.

- STMGL.py: Defines the main architecture of the STMGL network.

- teacher_model.py: Implements the teacher model.

- model_BiKD_STMGL.py: The main script that integrates the student and teacher networks to perform BiKD training. It outputs evaluation metrics including MAE, RMSE, and MAPE for each prediction horizon.

## Experimental Results
The comparison between BiKD-STMGL and other baseline comparison methods is based on the following metrics:

- MAE (Mean Absolute Error)

- RMSE (Root Mean Square Error)
  
- MAPE (Mean Absolute Percentage Error)
  
- SCI (Symmetric Consistency Index)
