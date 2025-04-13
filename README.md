# CyanoHost: A Tool for Cyanophage Host Prediction

CyanoHost is a specialized tool designed for predicting cyanophage hosts using graph neural networks.

## Overview

CyanoHost utilizes protein sequence embeddings and graph neural networks to predict the hosts of cyanophages based on their genomic sequences.

## Installation

1. Install Prokka for genome annotation
2. Install ESMFold for protein sequence embedding (refer to [ESMFold GitHub](https://github.com/facebookresearch/esm))
3. Install required Python packages:
   ```bash
   pip install torch torch-geometric
   ```

## Usage

### 1. Genome Annotation
First, annotate the cyanophage genome sequences using Prokka to obtain the ORF amino acid sequences (`.faa` files).

### 2. Protein Sequence Embedding
Use ESMFold to obtain embedding representations for each ORF amino acid sequence. 
```bash
    esm-extract esm2_t36_3B_UR50D faa_file output --toks_per_batch 1024 --include mean
```
For detailed usage of ESMFold, please refer to the [ESMFold documentation](https://github.com/facebookresearch/esm).

### 3. Model Training
1. Generate graph representations for training data:
   ```bash
   python graph_construct_train.py <train_label_file> <train_save_path> <prokka_path> <esm_path>
   ```
   For example:
   ```bash
   python graph_construct_train.py example_data/cyano_host_label.txt train_graphs example_data/prokka/ example_data/esm/
   ```
2. Train the model:
   ```bash
   python train.py <train_label_file> <graph_data>  
   ```
   For example:
   ```bash
   python train.py example_data/cyano_host_label.txt train_graphs
   ```

### 4. Prediction
1. Generate graph representations for prediction data:
   ```bash
   python graph_construct_pred.py
   ```
2. Make predictions using the trained model:
   ```bash
   python predict.py <model_path> <data_path> <output_path>
   ```
   - `model_path`: Path to the trained model
   - `data_path`: Path to the prediction data
   - `output_path`: Path to save prediction results

## File Descriptions

- `graph_construct_train.py`: Generates graph representations for training data
- `train.py`: Contains the model training code
- `graph_construct_pred.py`: Generates graph representations for prediction data
- `predict.py`: Makes predictions using the trained model

## Output Format

The prediction results are saved in a tab-separated format:
```
index    predicted_class
```
