# BiKD-STMGL
This repository contains the official implementation of the paper "Efficient Lane-Level Traffic Flow Prediction via Bidirectional Knowledge Distillation with Spatio-Temporal Memory Graph Learning".
# Code Structure
The implementation framework of BiKD-STMGL consists of four core files:

-**embedding.py**
  
Implements three embedding modules: temporal position embedding, temporal periodic embedding, and spatial Chebyshev graph embedding.

**- STMGL.py**
  
Defines the main architecture of the Spatio-Temporal Memory Graph Learning (STMGL) network.

**- teacher_model.py**
  
Implements the teacher model.

**- model_BiKD_STMGL.py**
  
The main script that integrates the student and teacher networks to perform Bidirectional Knowledge Distillation (BiKD) training.
It outputs evaluation metrics including MAE, RMSE, and MAPE for each prediction horizon.
