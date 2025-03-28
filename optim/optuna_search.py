import optuna
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset.axon_dataset import AxonDataset
from model.MTL_efficientnet import MultiTaskEfficientNet
from trainers.trainer import Trainer, MultiTaskLoss
from utils.transforms import AxonAugmenter
from utils.metrics import test_model

def objective(trial):
    """
    Objective function for Optuna. It samples hyperparameters, trains the model, 
    and returns a validation metric (here we use validation loss) for Optuna to minimize.
    """
    # -----------------------------
    # 1) Suggest hyperparameters
    # -----------------------------
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay  = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout_head  = trial.suggest_float('dropout_head', 0.3, 0.6)

    max_epochs    = 15
    patience      = 5

    # -----------------------------
    # 2) Set up data
    # -----------------------------
    base_dir = Path("")  # Adjust path to your data
    metadata_df = pd.read_csv('EM_Filtered_metadata.csv')
    
    # Simple train/val split
    unique_subjects = metadata_df['patient_id'].unique()
    train_subjects, val_subjects = train_test_split(
        unique_subjects,
        test_size=0.2,  # 20% for validation
        random_state=trial.number  # Use trial number as seed for reproducibility
    )
    
    train_df = metadata_df[metadata_df['patient_id'].isin(train_subjects)].reset_index(drop=True)
    val_df = metadata_df[metadata_df['patient_id'].isin(val_subjects)].reset_index(drop=True)
    transform = transforms.Compose([
        transforms.RandomCrop(1000),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    axon_augmenter = AxonAugmenter(
        rotation_range=360,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        noise_range=(0.0, 0.05),
        blur_prob=0.3,
        blur_radius=(0.5, 1.5),
        zoom_range=(0.85, 1.15),
        flip_prob=0.5
    )

    train_dataset = AxonDataset(train_df, base_dir, transform=transform, augment=axon_augmenter)
    val_dataset   = AxonDataset(val_df,   base_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------
    # 3) Build model & optimizer
    # -----------------------------
    model = MultiTaskEfficientNet().to(device)
    
    # Example: set new dropout with the trial-suggested dropout value
    # We'll simply locate the dropout layers in each head and adjust them
    for block in [model.pathology_head, model.region_head, model.depth_head]:
        for i, layer in enumerate(block):
            if isinstance(layer, nn.Dropout):
                block[i] = nn.Dropout(p=dropout_head)
    
    criterion = MultiTaskLoss(weights={'pathology': 1.0, 'region': 1.0, 'depth': 1.0})
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    exp_name = f"optuna_trial_{trial.number}_efficientnet"

    # -----------------------------
    # 4) Train 
    # -----------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=max_epochs,
        early_stopping_patience=patience,
        exp_name=exp_name
    )
    history = trainer.train()

    best_val_loss = min(history['val_loss'])
    return best_val_loss
