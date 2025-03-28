import torch
import torch.nn as nn

import random
import numpy as np
import optuna
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from model.MTL_efficientnet import MultiTaskEfficientNet
from dataset.axon_dataset import AxonDataset, AxonAugmenter
from trainers.trainer import Trainer, MultiTaskLoss
from optim.optuna_search import objective
from utils.metrics import test_model

def main():
    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    random.seed(42)
    np.random.seed(42)
    
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")  # We are minimizing val_loss
    
    # Run the study
    study.optimize(objective, n_trials=10, timeout=None, show_progress_bar=True)
    
    # Print summary of results
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    best_trial = study.best_trial
    print(f"    Value (best val_loss): {best_trial.value}")
    print("    Params: ")
    for key, value in best_trial.params.items():
        print(f"      {key}: {value}")
    
    # -----------------------------
    # (Optional) Retrain or load the best model
    # -----------------------------
    best_hparams = best_trial.params
    print("\nRe-building the best model to test on hold-out set...")

    base_dir = Path("C:\Kelvin_ASD_EM_v2")
    metadata_df = pd.read_csv('EM_Filtered_metadata.csv')
    train_val_data, test_data = train_test_split(metadata_df['patient_id'].unique(), test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.RandomCrop(1000),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    test_dataset = AxonDataset(test_data, base_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=int(best_hparams['batch_size']), shuffle=False)

    model = MultiTaskEfficientNet().to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing.")
        model = nn.DataParallel(model)

    # Adjust the dropout layers based on best hyperparameters
    dropout_head = best_hparams['dropout_head']
    for block in [model.module.pathology_head if isinstance(model, nn.DataParallel) else model.pathology_head,
                  model.module.region_head    if isinstance(model, nn.DataParallel) else model.region_head,
                  model.module.depth_head     if isinstance(model, nn.DataParallel) else model.depth_head]:
        for i, layer in enumerate(block):
            if isinstance(layer, nn.Dropout):
                block[i] = nn.Dropout(p=dropout_head)

    # Re-create optimizer, etc., if you plan to retrain. Or just load from 'best_model.pth':
    # checkpoint = torch.load('best_model.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on the test set:
    metrics = test_model(
        model=model,
        test_loader=test_loader,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        pathology_num_classes=2,  # CTR vs ASD
        region_num_classes=3,     # A25, A46, OFC
        depth_num_classes=2       # DWM vs SWM
    )
    print("Test Metrics (with best hyperparams):", metrics)