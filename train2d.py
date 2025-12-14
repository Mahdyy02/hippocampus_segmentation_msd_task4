import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import your model
from model2d import UNetnnUNetLightweight

# =============================================================================
# DATASET
# =============================================================================

class HippocampusDataset(Dataset):
    def __init__(self, img_dir, mask_dir, split='train', train_ratio=0.8):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # Get all files
        all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        
        # Split train/val
        n_train = int(len(all_imgs) * train_ratio)
        if split == 'train':
            self.img_files = all_imgs[:n_train]
        else:
            self.img_files = all_imgs[n_train:]
        
        print(f"{split.upper()} set: {len(self.img_files)} samples")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        mask_name = img_name.replace('img_', 'mask_')
        
        img = np.load(os.path.join(self.img_dir, img_name))
        mask = np.load(os.path.join(self.mask_dir, mask_name))
        
        # Add channel dimension
        img = torch.from_numpy(img).unsqueeze(0).float()  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)
        
        return img, mask


# =============================================================================
# LOSS FUNCTIONS - STATE OF THE ART FOR MEDICAL SEGMENTATION
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss - Gold standard for segmentation
    Reference: Milletari et al. "V-Net" (2016)
    """
    def __init__(self, smooth=1.0, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
    
    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute Dice per class
        start_idx = 0 if self.include_background else 1
        dice_scores = []
        
        for c in range(start_idx, logits.shape[1]):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)
        return 1.0 - dice_scores.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - Handles class imbalance
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Dice + Focal Loss - Best of both worlds
    Used in nnU-Net and many SOTA medical segmentation papers
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, include_background=False):
        super().__init__()
        self.dice_loss = DiceLoss(include_background=include_background)
        self.focal_loss = FocalLoss(gamma=2.0)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal


# =============================================================================
# METRICS - MSD CHALLENGE STANDARD
# =============================================================================

def compute_dice_score(pred, target, num_classes=3, include_background=False):
    """
    Compute Dice Score per class (MSD Challenge metric)
    """
    dice_scores = []
    start_idx = 0 if include_background else 1
    
    for c in range(start_idx, num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float()
        
        if union == 0:
            dice_scores.append(1.0 if intersection == 0 else 0.0)
        else:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())
    
    return dice_scores


def compute_nsd(pred, target, num_classes=3, tolerance=2.0, include_background=False):
    """
    Normalized Surface Distance (NSD) - MSD Challenge metric
    Simplified version: measures boundary agreement within tolerance
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion
    
    nsd_scores = []
    start_idx = 0 if include_background else 1
    
    for c in range(start_idx, num_classes):
        pred_c = (pred == c).cpu().numpy()
        target_c = (target == c).cpu().numpy()
        
        if pred_c.sum() == 0 and target_c.sum() == 0:
            nsd_scores.append(1.0)
            continue
        elif pred_c.sum() == 0 or target_c.sum() == 0:
            nsd_scores.append(0.0)
            continue
        
        # Compute boundaries using XOR with erosion
        pred_border = pred_c ^ binary_erosion(pred_c)
        target_border = target_c ^ binary_erosion(target_c)
        
        if not pred_border.any() or not target_border.any():
            nsd_scores.append(0.0)
            continue
        
        # Distance from prediction border to target
        dist_pred_to_target = distance_transform_edt(~target_border)
        pred_border_dists = dist_pred_to_target[pred_border]
        
        # Distance from target border to prediction
        dist_target_to_pred = distance_transform_edt(~pred_border)
        target_border_dists = dist_target_to_pred[target_border]
        
        # NSD: percentage of border points within tolerance
        nsd = (
            (pred_border_dists <= tolerance).sum() + 
            (target_border_dists <= tolerance).sum()
        ) / (len(pred_border_dists) + len(target_border_dists))
        
        nsd_scores.append(float(nsd))
    
    return nsd_scores


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    all_dice = []
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        
        # Metrics
        preds = logits.argmax(dim=1)
        for i in range(images.size(0)):
            dice = compute_dice_score(preds[i], masks[i])
            all_dice.append(dice)
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(all_dice, axis=0)
    
    return avg_loss, avg_dice


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_dice = []
    all_nsd = []
    
    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            
            preds = logits.argmax(dim=1)
            
            for i in range(images.size(0)):
                dice = compute_dice_score(preds[i], masks[i])
                nsd = compute_nsd(preds[i], masks[i])
                all_dice.append(dice)
                all_nsd.append(nsd)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(all_dice, axis=0)
    avg_nsd = np.mean(all_nsd, axis=0)
    
    return avg_loss, avg_dice, avg_nsd


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_curves(history, save_path='training_curves.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice scores
    train_dice = np.array(history['train_dice'])
    val_dice = np.array(history['val_dice'])
    
    axes[0, 1].plot(train_dice[:, 0], label='Train Dice (Anterior)', linewidth=2)
    axes[0, 1].plot(val_dice[:, 0], label='Val Dice (Anterior)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Score - Anterior Hippocampus')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].plot(train_dice[:, 1], label='Train Dice (Posterior)', linewidth=2)
    axes[1, 0].plot(val_dice[:, 1], label='Val Dice (Posterior)', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].set_title('Dice Score - Posterior Hippocampus')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # NSD scores
    val_nsd = np.array(history['val_nsd'])
    axes[1, 1].plot(val_nsd[:, 0], label='NSD (Anterior)', linewidth=2, marker='o')
    axes[1, 1].plot(val_nsd[:, 1], label='NSD (Posterior)', linewidth=2, marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('NSD Score')
    axes[1, 1].set_title('Normalized Surface Distance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def visualize_predictions(model, loader, device, num_samples=4, save_path='predictions.png'):
    model.eval()
    
    # Get samples
    images, masks = next(iter(loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]
    
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
    
    images = images.cpu()
    
    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Input image
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(masks[i], cmap='jet', vmin=0, vmax=2)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(preds[i], cmap='jet', vmin=0, vmax=2)
        dice = compute_dice_score(preds[i], masks[i])
        axes[i, 2].set_title(f'Prediction\nDice: {np.mean(dice):.3f}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions saved to {save_path}")
    plt.close()


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    # Configuration
    config = {
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'num_classes': 3,
        'use_amp': True,  # Automatic Mixed Precision
        'save_dir': 'checkpoints',
        'dataset_dir': 'dataset_2d'
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset & DataLoader
    train_dataset = HippocampusDataset(
        img_dir=os.path.join(config['dataset_dir'], 'images'),
        mask_dir=os.path.join(config['dataset_dir'], 'masks'),
        split='train',
        train_ratio=0.8
    )
    
    val_dataset = HippocampusDataset(
        img_dir=os.path.join(config['dataset_dir'], 'images'),
        mask_dir=os.path.join(config['dataset_dir'], 'masks'),
        split='val',
        train_ratio=0.8
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = UNetnnUNetLightweight()
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss & Optimizer (SOTA for medical segmentation)
    criterion = CombinedLoss(
        dice_weight=0.5,
        focal_weight=0.5,
        include_background=False
    )
    
    # AdamW optimizer - better than Adam for generalization
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if config['use_amp'] else None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'val_nsd': []
    }
    
    best_dice = 0.0
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_dice, val_nsd = evaluate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['val_nsd'].append(val_nsd)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Train Dice: Anterior={train_dice[0]:.4f}, Posterior={train_dice[1]:.4f}")
        print(f"Val Dice:   Anterior={val_dice[0]:.4f}, Posterior={val_dice[1]:.4f}")
        print(f"Val NSD:    Anterior={val_nsd[0]:.4f}, Posterior={val_nsd[1]:.4f}")
        
        # Save best model
        mean_dice = np.mean(val_dice)
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"💾 Best model saved! Mean Dice: {best_dice:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': config
    }, os.path.join(config['save_dir'], 'final_model.pth'))
    
    # Save history as JSON
    history_serializable = {
        k: [item.tolist() if isinstance(item, np.ndarray) else item 
            for item in v]
        for k, v in history.items()
    }
    with open(os.path.join(config['save_dir'], 'history.json'), 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    # Plot results
    plot_training_curves(history, save_path='training_curves.png')
    visualize_predictions(model, val_loader, device, save_path='predictions.png')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Final Anterior Dice: {val_dice[0]:.4f}")
    print(f"Final Posterior Dice: {val_dice[1]:.4f}")
    print(f"Final Anterior NSD: {val_nsd[0]:.4f}")
    print(f"Final Posterior NSD: {val_nsd[1]:.4f}")


if __name__ == "__main__":
    main() 