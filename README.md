# Boston University Computational Neuroscience and Vison Lab, Human Systems Neuroscience Lab
## Machine learning approaches to systematically study short- and long-range cortical pathways and identify axon pathology in autism

**EM Pipeline. Optical microscopy pipeline in different repo, will merge later.** 

## Project Introduction
Axon pathology is at the core of disruptions in cortical connectivity in autism spectrum disorder (ASD). However, the extent and distribution of disruption in a) short- and long-range cortical pathways and b) pathways linking association cortices or other cortices and subcortical structures are unknown. Neuroanatomical analysis of high-resolution features of individual axons, such as density, trajectory, branching patterns, and myelin in multiple cortical pathways, are labor-intensive and time-consuming. This limits large-scale studies that otherwise can help identify core pathways that are altered in ASD and likely mechanisms that underly neural communication disruption. To automate and optimize analysis and visualization of patterns of disruption, we customized machine learning techniques to quantify the requisite power of multiscale optical and electron microscopy for accurately classifying neurotypical and ASD postmortem brain histopathology sections.

This repository provides a unified deep learning pipeline for multi-task classification of microscopy images of brain tissue. It supports both **optical** and **electron microscopy (EM)** modalities for classifying:

- **Pathology** (ASD vs Control)
- **Brain Region** (A25, A46, OFC)
- **Tissue Depth** (DWM vs SWM)

The pipeline leverages transfer learning on **EfficientNet-B2**, data augmentation specific to axon imaging, and automatic hyperparameter optimization via **Optuna**.

---

## Project Structure

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/kkuang0/CNVL_BZ_EM_Axon.git
cd CNVL_BZ_EM_Axon
```
### 2. Set up the environment using:
Conda:
```
conda env create -f environment.yml
conda activate axon_EM
```
pip: to be implemented

## Running Experiments

---

### Train with Optuna (Hyperparameter Search)
```
python scripts/train.py
```
This will run Optuna across N trials, track validation loss, and log to TensorBoard.

### Evaluate best model
```
python scripts/evaluate.py
```
This will load the best model checkpoint (checkpoints/best_model.pth) and evaluate it on a holdout test set.

## Model
We use a multi-task EfficientNet-B2 backbone with custom heads:

Input: 1-channel grayscale microscopy image

Backbone: torchvision.models.efficientnet_b2 (pretrained on ImageNet)

Heads:

pathology_head: Binary classification (ASD vs CTR)

region_head: 3-class classification (A25, A46, OFC)

depth_head: Binary classification (DWM vs SWM)

## TensorBoard logging
TensorBoard logs are saved in runs/ (excluded from Git). To visualize:
```
tensorboard --logdir=runs/
```

## Data requirements
Each pipeline requires a metadata.csv file with the following columns:
| filepath | pathology | region | depth | patient_id |
| ----------- | ----------- |
| path/to/image.tif | ASD | A25 | DWM | HEG |
Data from BZ.
