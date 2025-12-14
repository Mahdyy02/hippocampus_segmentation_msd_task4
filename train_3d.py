"""
Quick Start Training Script for Medical Decathlon Task04 Hippocampus
Simple single configuration training without grid search
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import json
from pathlib import Path

# Import from your files
from unet_inr_3d import UNet3DWithINRBalanced, count_parameters
from train_3d import (
    MedicalDecathlonHippocampusDataset,
    CombinedLoss,
    compute_dice_score_3d,
    compute_nsd_3d,
    pad_to_max_size  # Import the collate function!
)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_dice = []
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images, masks, _ = batch
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        preds = logits.argmax(dim=1)
        for i in range(images.size(0)):
            dice = compute_dice_score_3d(preds[i], masks[i])
            all_dice.append(dice)
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{np.mean(all_dice):.3f}'
        })
    
    return total_loss / len(loader), np.mean(all_dice, axis=0)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_dice = []
    all_nsd = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for batch in pbar:
            images, masks, _ = batch
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            
            preds = logits.argmax(dim=1)
            for i in range(images.size(0)):
                dice = compute_dice_score_3d(preds[i], masks[i])
                nsd = compute_nsd_3d(preds[i], masks[i])
                all_dice.append(dice)
                all_nsd.append(nsd)
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{np.mean(all_dice):.3f}'
            })
    
    return total_loss / len(loader), np.mean(all_dice, axis=0), np.mean(all_nsd, axis=0)


def main():
    # =============================================================================
    # CONFIGURATION - EDIT THIS!
    # =============================================================================
    config = {
        'data_root': './Task04_Hippocampus',  # Path to your Decathlon data
        'save_dir': './results_hippocampus',
        'batch_size': 2,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'train_ratio': 0.8,
        'num_workers': 0,  # Set to 0 for Windows! Use 4 on Linux/Mac
        'save_every': 10,  # Save checkpoint every N epochs
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"3D UNET + INR TRAINING - MEDICAL DECATHLON TASK04")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Data: {config['data_root']}")
    print(f"Save: {config['save_dir']}")
    
    # =============================================================================
    # DATASET
    # =============================================================================
    print(f"\n{'='*80}")
    print(f"LOADING DATASET")
    print(f"{'='*80}")
    
    train_dataset = MedicalDecathlonHippocampusDataset(
        data_root=config['data_root'],
        split='train',
        train_ratio=config['train_ratio'],
        augment=True,
        normalize=True
    )
    
    val_dataset = MedicalDecathlonHippocampusDataset(
        data_root=config['data_root'],
        split='val',
        train_ratio=config['train_ratio'],
        augment=False,
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=pad_to_max_size  # Handle variable sizes!
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=pad_to_max_size  # Handle variable sizes!
    )
    
    print(f"\nTraining volumes: {len(train_dataset)}")
    print(f"Validation volumes: {len(val_dataset)}")
    
    # Check sample
    sample_img, sample_mask, sample_name = train_dataset[0]
    print(f"\nSample: {sample_name}")
    print(f"  Image shape: {sample_img.shape}")
    print(f"  Mask shape: {sample_mask.shape}")
    print(f"  Unique labels: {torch.unique(sample_mask).tolist()}")
    
    # =============================================================================
    # MODEL
    # =============================================================================
    print(f"\n{'='*80}")
    print(f"CREATING MODEL")
    print(f"{'='*80}")
    
    model = UNet3DWithINRBalanced(in_channels=1, num_classes=3).to(device)
    
    print(f"Model: UNet3D + INR")
    print(f"Parameters: {count_parameters(model):,}")
    
    # =============================================================================
    # TRAINING SETUP
    # =============================================================================
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # =============================================================================
    # TRAINING LOOP
    # =============================================================================
    print(f"\n{'='*80}")
    print(f"TRAINING")
    print(f"{'='*80}\n")
    
    best_dice = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'val_nsd': []
    }
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice, val_nsd = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Save history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_dice'].append(train_dice.tolist())
        history['val_dice'].append(val_dice.tolist())
        history['val_nsd'].append(val_nsd.tolist())
        
        # Print summary
        mean_val_dice = np.mean(val_dice)
        print(f"\n[Epoch {epoch+1}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Dice:   Anterior={val_dice[0]:.3f}, Posterior={val_dice[1]:.3f}, Mean={mean_val_dice:.3f}")
        print(f"  Val NSD:    Anterior={val_nsd[0]:.3f}, Posterior={val_nsd[1]:.3f}")
        
        # Save best model
        if mean_val_dice > best_dice:
            best_dice = mean_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"  ✓ New best model saved! (Dice: {best_dice:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': config
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save history
        with open(os.path.join(config['save_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Results saved to: {config['save_dir']}")
    print(f"\nFiles:")
    print(f"  - best_model.pth")
    print(f"  - history.json")
    print(f"  - config.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()