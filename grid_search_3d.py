import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import json
import itertools
from datetime import datetime
import nibabel as nib
from pathlib import Path

# Import the 3D model
from unet_inr_3d import (
    UNet3DWithINR, 
    UNet3DWithINRLightweight, 
    UNet3DWithINRBalanced,
    count_parameters
)


# =============================================================================
# COLLATE FUNCTION FOR VARIABLE-SIZED VOLUMES
# =============================================================================

def pad_to_max_size(batch):
    """
    Custom collate function to handle variable-sized 3D volumes
    Pads all volumes to the maximum size in the batch
    """
    images, masks, filenames = [], [], []
    
    for item in batch:
        images.append(item[0])
        masks.append(item[1])
        filenames.append(item[2])
    
    # Find max dimensions
    max_d = max([img.shape[1] for img in images])
    max_h = max([img.shape[2] for img in images])
    max_w = max([img.shape[3] for img in images])
    
    # Pad all volumes to max size
    padded_images = []
    padded_masks = []
    
    for img, mask in zip(images, masks):
        _, d, h, w = img.shape
        
        # Calculate padding
        pad_d = max_d - d
        pad_h = max_h - h
        pad_w = max_w - w
        
        # Pad image (C, D, H, W)
        img_padded = F.pad(img, (
            0, pad_w,  # width
            0, pad_h,  # height
            0, pad_d   # depth
        ), mode='constant', value=0)
        
        # Pad mask (D, H, W)
        mask_padded = F.pad(mask, (
            0, pad_w,
            0, pad_h,
            0, pad_d
        ), mode='constant', value=0)
        
        padded_images.append(img_padded)
        padded_masks.append(mask_padded)
    
    # Stack into batch
    images_batch = torch.stack(padded_images, 0)
    masks_batch = torch.stack(padded_masks, 0)
    
    return images_batch, masks_batch, filenames

# =============================================================================
# MEDICAL SEGMENTATION DECATHLON DATASET (NIfTI FORMAT)
# =============================================================================

class MedicalDecathlonHippocampusDataset(Dataset):
    """
    Medical Segmentation Decathlon - Task04 Hippocampus
    
    Directory structure:
    Task04_Hippocampus/
        imagesTr/
            hippocampus_001.nii.gz
            hippocampus_002.nii.gz
            ...
        labelsTr/
            hippocampus_001.nii.gz
            hippocampus_002.nii.gz
            ...
        imagesTs/
            hippocampus_XXX.nii.gz
            ...
        dataset.json
    """
    def __init__(
        self, 
        data_root, 
        split='train', 
        train_ratio=0.8,
        augment=False,
        normalize=True,
        target_spacing=None  # Optional: resample to target spacing
    ):
        """
        Args:
            data_root: Path to Task04_Hippocampus directory
            split: 'train', 'val', or 'test'
            train_ratio: Ratio for train/val split
            augment: Apply data augmentation (only for training)
            normalize: Normalize intensity values
            target_spacing: Tuple (z, y, x) spacing in mm, or None to keep original
        """
        self.data_root = Path(data_root)
        self.split = split
        self.augment = augment and (split == 'train')
        self.normalize = normalize
        self.target_spacing = target_spacing
        
        # Load dataset.json for metadata
        with open(self.data_root / 'dataset.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        print(f"\nDataset: {self.dataset_info['name']}")
        print(f"Description: {self.dataset_info['description']}")
        print(f"Labels: {self.dataset_info['labels']}")
        
        if split == 'test':
            # Test set
            self.img_dir = self.data_root / 'imagesTs'
            self.label_dir = None
            # Filter out macOS metadata files (._*)
            self.img_files = sorted([
                f for f in self.img_dir.glob('*.nii.gz') 
                if not f.name.startswith('._')
            ])
            print(f"\n{split.upper()} set: {len(self.img_files)} volumes (no labels)")
        else:
            # Training set - split into train/val
            self.img_dir = self.data_root / 'imagesTr'
            self.label_dir = self.data_root / 'labelsTr'
            
            # Filter out macOS metadata files (._*)
            all_files = sorted([
                f for f in self.img_dir.glob('*.nii.gz') 
                if not f.name.startswith('._')
            ])
            n_train = int(len(all_files) * train_ratio)
            
            if split == 'train':
                self.img_files = all_files[:n_train]
            else:  # val
                self.img_files = all_files[n_train:]
            
            print(f"\n{split.upper()} set: {len(self.img_files)} volumes")
    
    def __len__(self):
        return len(self.img_files)
    
    def load_nifti(self, filepath):
        """Load NIfTI file and return numpy array"""
        nii = nib.load(str(filepath))
        data = nii.get_fdata()
        return data.astype(np.float32), nii.affine, nii.header
    
    def normalize_intensity(self, img):
        """Normalize intensity to [0, 1] range"""
        # Clip outliers
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    def augment_volume(self, img, mask):
        """Simple 3D augmentation"""
        # Random flip along axes
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        
        # Random intensity shift (±10%)
        if np.random.rand() > 0.5:
            shift = np.random.uniform(0.9, 1.1)
            img = np.clip(img * shift, 0, 1)
        
        return img, mask
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        # Load image
        img, img_affine, img_header = self.load_nifti(img_path)
        
        # Load label if available
        if self.label_dir is not None:
            label_path = self.label_dir / img_path.name
            mask, _, _ = self.load_nifti(label_path)
            mask = mask.astype(np.int64)
        else:
            # Test set - no labels
            mask = np.zeros_like(img, dtype=np.int64)
        
        # Normalize intensity
        if self.normalize:
            img = self.normalize_intensity(img)
        
        # Data augmentation
        if self.augment:
            img, mask = self.augment_volume(img, mask)
        
        # Convert to torch tensors
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        
        return img, mask, str(img_path.name)


# =============================================================================
# ALTERNATIVE: NUMPY ARRAY DATASET (if you preprocess to .npy)
# =============================================================================

class Hippocampus3DDataset(Dataset):
    """
    For preprocessed numpy arrays
    Expects .npy files in images/ and labels/ directories
    """
    def __init__(self, img_dir, mask_dir, split='train', train_ratio=0.8, augment=False):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment and (split == 'train')
        
        # Get all volume files
        all_vols = sorted([f for f in self.img_dir.glob('*.npy')])
        
        # Split train/val
        n_train = int(len(all_vols) * train_ratio)
        if split == 'train':
            self.vol_files = all_vols[:n_train]
        else:
            self.vol_files = all_vols[n_train:]
        
        print(f"{split.upper()} set: {len(self.vol_files)} 3D volumes")
    
    def __len__(self):
        return len(self.vol_files)
    
    def __getitem__(self, idx):
        vol_name = self.vol_files[idx]
        
        # Load 3D volume and mask
        img = np.load(vol_name)
        mask_name = self.mask_dir / vol_name.name
        mask = np.load(mask_name)
        
        # Basic augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()
        
        # Add channel dimension
        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        
        return img, mask, vol_name.name


# =============================================================================
# LOSS FUNCTIONS (Same as 2D)
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # 3D!
        
        start_idx = 0 if self.include_background else 1
        dice_scores = []
        
        for c in range(start_idx, logits.shape[1]):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1, 2, 3))  # 3D!
            union = pred_c.sum(dim=(1, 2, 3)) + target_c.sum(dim=(1, 2, 3))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)
        return 1.0 - dice_scores.mean()


class FocalLoss(nn.Module):
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
# METRICS (Adapted for 3D)
# =============================================================================

def compute_dice_score_3d(pred, target, num_classes=3, include_background=False):
    """Compute 3D Dice scores"""
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


def compute_nsd_3d(pred, target, num_classes=3, tolerance=2.0, include_background=False):
    """Compute 3D NSD (Surface Distance) scores"""
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
        
        # Get surface voxels (3D erosion)
        pred_border = pred_c ^ binary_erosion(pred_c)
        target_border = target_c ^ binary_erosion(target_c)
        
        if not pred_border.any() or not target_border.any():
            nsd_scores.append(0.0)
            continue
        
        # Compute distances (works in 3D automatically)
        dist_pred_to_target = distance_transform_edt(~target_border)
        pred_border_dists = dist_pred_to_target[pred_border]
        
        dist_target_to_pred = distance_transform_edt(~pred_border)
        target_border_dists = dist_target_to_pred[target_border]
        
        nsd = (
            (pred_border_dists <= tolerance).sum() + 
            (target_border_dists <= tolerance).sum()
        ) / (len(pred_border_dists) + len(target_border_dists))
        
        nsd_scores.append(float(nsd))
    
    return nsd_scores


# =============================================================================
# TRAINING & EVALUATION (Adapted for 3D)
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, show_progress=True):
    model.train()
    total_loss = 0
    all_dice = []
    
    pbar = loader
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(loader, desc='Training', leave=False)
    
    for batch in pbar:
        if len(batch) == 3:
            images, masks, _ = batch  # Unpack filename
        else:
            images, masks = batch
            
        images = images.to(device)  # (B, 1, D, H, W)
        masks = masks.to(device)    # (B, D, H, W)
        
        optimizer.zero_grad()
        
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
        
        preds = logits.argmax(dim=1)  # (B, D, H, W)
        for i in range(images.size(0)):
            dice = compute_dice_score_3d(preds[i], masks[i])
            all_dice.append(dice)
        
        total_loss += loss.item()
        
        if show_progress and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(all_dice, axis=0)
    
    return avg_loss, avg_dice


def evaluate(model, loader, criterion, device, show_progress=True):
    model.eval()
    total_loss = 0
    all_dice = []
    all_nsd = []
    
    pbar = loader
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(loader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            if len(batch) == 3:
                images, masks, _ = batch  # Unpack filename
            else:
                images, masks = batch
                
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
            
            if show_progress and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(all_dice, axis=0)
    avg_nsd = np.mean(all_nsd, axis=0)
    
    return avg_loss, avg_dice, avg_nsd


# =============================================================================
# SINGLE TRAINING RUN
# =============================================================================

def train_single_config(config, train_loader, val_loader, device):
    """Train a single 3D model configuration"""
    
    # Create 3D model
    model = UNet3DWithINR(
        in_channels=1,
        base_channels=config['base_channels'],
        unet_depth=config['unet_depth'],
        inr_hidden_dim=config['inr_hidden_dim'],
        inr_layers=config['inr_layers'],
        projection_dim=config['projection_dim'],
        num_classes=3,
        use_feature_projection=True
    ).to(device)
    
    # Loss & optimizer
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if config['use_amp'] else None
    
    best_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'val_nsd': []}
    
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Starting 3D training for {config['num_epochs']} epochs...")
    print()
    
    # Training loop
    from tqdm import tqdm
    for epoch in tqdm(range(config['num_epochs']), desc='  Epochs', leave=False):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, scaler, show_progress=True)
        val_loss, val_dice, val_nsd = evaluate(model, val_loader, criterion, device, show_progress=True)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice.tolist())
        history['val_dice'].append(val_dice.tolist())
        history['val_nsd'].append(val_nsd.tolist())
        
        mean_dice = np.mean(val_dice)
        if mean_dice > best_dice:
            best_dice = mean_dice
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) in [1, 5]:
            print(f"  Epoch {epoch+1:3d}/{config['num_epochs']}: "
                  f"Loss={val_loss:.4f}, "
                  f"Dice=[A:{val_dice[0]:.3f}, P:{val_dice[1]:.3f}], "
                  f"Best={best_dice:.3f}")
    
    final_metrics = {
        'best_dice': best_dice,
        'final_dice_anterior': val_dice[0],
        'final_dice_posterior': val_dice[1],
        'final_nsd_anterior': val_nsd[0],
        'final_nsd_posterior': val_nsd[1],
        'mean_dice': np.mean(val_dice),
        'mean_nsd': np.mean(val_nsd),
        'num_params': count_parameters(model)
    }
    
    return final_metrics, history, model


# =============================================================================
# 3D GRID SEARCH
# =============================================================================

def grid_search_3d(train_loader, val_loader, device, save_dir='grid_search_3d_results'):
    """
    Grid search for 3D UNet+INR
    Note: Using smaller configs due to 3D memory requirements
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 3D-OPTIMIZED PARAMETER GRID (smaller than 2D!)
    param_grid = {
        'base_channels': [16, 24, 32],      # Smaller for 3D
        'unet_depth': [3, 4],                # Shallower for memory
        'inr_hidden_dim': [32, 48, 64],     # Smaller
        'inr_layers': [2, 3],
        'projection_dim': [4, 8, 12],       # Smaller
        'learning_rate': [1e-4],
        'weight_decay': [1e-5],
        'num_epochs': [5],                  # Fewer epochs for testing
        'use_amp': [True]
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    # Filter combinations
    filtered_combinations = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        
        # Memory-aware filtering for 3D
        if config['base_channels'] == 16 and config['inr_hidden_dim'] > 48:
            continue
        if config['base_channels'] == 32 and config['inr_hidden_dim'] < 48:
            continue
        if config['projection_dim'] > config['base_channels'] // 2:
            continue
        
        filtered_combinations.append(combo)
    
    print("="*80)
    print(f"3D GRID SEARCH")
    print(f"Total combinations: {len(combinations)}")
    print(f"After filtering: {len(filtered_combinations)}")
    print("="*80)
    
    results = []
    
    for idx, combo in enumerate(filtered_combinations):
        config = dict(zip(keys, combo))
        config_str = (
            f"3D_BC{config['base_channels']}_"
            f"UD{config['unet_depth']}_"
            f"IH{config['inr_hidden_dim']}_"
            f"IL{config['inr_layers']}_"
            f"PD{config['projection_dim']}"
        )
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(filtered_combinations)}] TESTING 3D CONFIGURATION")
        print(f"{'='*80}")
        print(f"Config: {config_str}")
        
        try:
            metrics, history, model = train_single_config(config, train_loader, val_loader, device)
            
            result = {
                'config': config,
                'config_str': config_str,
                'metrics': metrics,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            results.append(result)
            
            print(f"\n✓ 3D TRAINING COMPLETE")
            print(f"Best Dice: {metrics['best_dice']:.4f}")
            print(f"Parameters: {metrics['num_params']:,}")
            
            # Save result
            result_file = os.path.join(save_dir, f'result_{config_str}.json')
            with open(result_file, 'w') as f:
                result_copy = {
                    'config': config,
                    'config_str': config_str,
                    'metrics': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                               for k, v in metrics.items()},
                    'timestamp': result['timestamp']
                }
                json.dump(result_copy, f, indent=2)
            
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save all results
    all_results_file = os.path.join(save_dir, 'all_results_3d.json')
    with open(all_results_file, 'w') as f:
        results_copy = []
        for r in results:
            results_copy.append({
                'config': r['config'],
                'config_str': r['config_str'],
                'metrics': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                           for k, v in r['metrics'].items()},
                'timestamp': r['timestamp']
            })
        json.dump(results_copy, f, indent=2)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main training script for Medical Segmentation Decathlon Task04 Hippocampus
    """
    config = {
        'batch_size': 2,  # Small for 3D volumes
        'data_root': './Task04_Hippocampus',  # UPDATE THIS to your path
        'save_dir': 'results_3d_hippocampus',
        'train_ratio': 0.8,
        'num_workers': 0  # Set to 0 for Windows! Use 4 on Linux/Mac
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"MEDICAL SEGMENTATION DECATHLON - TASK04 HIPPOCAMPUS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Data root: {config['data_root']}")
    print(f"Batch size: {config['batch_size']}")
    
    # Dataset & DataLoader - Medical Decathlon format
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
    
    print(f"\n{'='*80}")
    print(f"DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Check a sample
    sample_img, sample_mask, sample_name = train_dataset[0]
    print(f"\nSample volume: {sample_name}")
    print(f"  Image shape: {sample_img.shape}")  # Should be (1, D, H, W)
    print(f"  Mask shape: {sample_mask.shape}")   # Should be (D, H, W)
    print(f"  Mask classes: {torch.unique(sample_mask).tolist()}")
    
    print("\n" + "="*80)
    print("STARTING 3D GRID SEARCH")
    print("="*80)
    
    results = grid_search_3d(train_loader, val_loader, device, config['save_dir'])
    
    print("\n" + "="*80)
    print("3D GRID SEARCH COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()