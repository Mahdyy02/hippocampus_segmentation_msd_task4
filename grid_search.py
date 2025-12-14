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

from model2d import UNetnnUNetWithINR, UNetnnUNetLightweight, UNetnnUNetBalanced

# =============================================================================
# JUPYTER DETECTION
# =============================================================================

def is_jupyter():
    """Detect if running in Jupyter notebook"""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        pass
    return False

def get_tqdm():
    """Get appropriate tqdm for environment"""
    if is_jupyter():
        from tqdm.notebook import tqdm
        return tqdm
    else:
        from tqdm import tqdm
        return tqdm

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
        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        
        return img, mask


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
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
# METRICS
# =============================================================================

def compute_dice_score(pred, target, num_classes=3, include_background=False):
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
        
        pred_border = pred_c ^ binary_erosion(pred_c)
        target_border = target_c ^ binary_erosion(target_c)
        
        if not pred_border.any() or not target_border.any():
            nsd_scores.append(0.0)
            continue
        
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
# TRAINING & EVALUATION
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, show_progress=True):
    model.train()
    total_loss = 0
    all_dice = []
    
    pbar = loader
    if show_progress:
        tqdm = get_tqdm()
        pbar = tqdm(loader, desc='Training', leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
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
        
        preds = logits.argmax(dim=1)
        for i in range(images.size(0)):
            dice = compute_dice_score(preds[i], masks[i])
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
        tqdm = get_tqdm()
        pbar = tqdm(loader, desc='Validation', leave=False)
    
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
    """Train a single model configuration"""
    
    # Create nnU-Net model with optional INR
    model = UNetnnUNetWithINR(
        in_channels=1,
        base_channels=config['base_channels'],
        unet_depth=config['unet_depth'],
        num_classes=3,
        deep_supervision=False,
        use_inr=config['use_inr'],  # Enable/disable INR
        inr_hidden_dim=config.get('inr_hidden_dim', 64)  # INR size if enabled
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
    
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training {config['num_epochs']} epochs...\n")
    
    # Training loop
    tqdm = get_tqdm()
    epoch_pbar = tqdm(range(config['num_epochs']), desc='  Epochs', leave=False)
    
    for epoch in epoch_pbar:
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, scaler, show_progress=False)
        val_loss, val_dice, val_nsd = evaluate(model, val_loader, criterion, device, show_progress=False)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice.tolist())
        history['val_dice'].append(val_dice.tolist())
        history['val_nsd'].append(val_nsd.tolist())
        
        mean_dice = np.mean(val_dice)
        if mean_dice > best_dice:
            best_dice = mean_dice
        
        # Update progress bar with current metrics
        epoch_pbar.set_postfix({
            'Loss': f'{val_loss:.3f}',
            'Dice': f'{mean_dice:.3f}',
            'Best': f'{best_dice:.3f}'
        })
        
        # Print milestone updates (much less frequent for Jupyter)
        if (epoch + 1) in [1, 5, 10, 20, 30, 40, 50, 75, 100] or (epoch + 1) == config['num_epochs']:
            print(f"  Epoch {epoch+1:3d}: Loss={val_loss:.4f}, Dice=[A:{val_dice[0]:.3f}, P:{val_dice[1]:.3f}], Best={best_dice:.3f}")
    
    # Calculate final metrics
    final_metrics = {
        'best_dice': best_dice,
        'final_dice_anterior': val_dice[0],
        'final_dice_posterior': val_dice[1],
        'final_nsd_anterior': val_nsd[0],
        'final_nsd_posterior': val_nsd[1],
        'mean_dice': np.mean(val_dice),
        'mean_nsd': np.mean(val_nsd),
        'num_params': sum(p.numel() for p in model.parameters())
    }
    
    return final_metrics, history, model


# =============================================================================
# REFINED GRID SEARCH
# =============================================================================

def refined_grid_search(train_loader, val_loader, device, save_dir='grid_search_results'):
    """
    Refined grid search with more sensible parameter combinations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # STANDARD nnU-NET + INR FINETUNING
    # Fixed standard nnU-Net configs for 2D segmentation
    # Only finetune INR parameters
    param_grid = {
        'base_channels': [32],  # Standard for 2D
        'unet_depth': [4],      # Standard for 2D
        'learning_rate': [1e-4],  # Standard
        'weight_decay': [1e-5],   # Standard
        'num_epochs': [100],
        'use_amp': [True],
        
        # INR finetuning - only these vary
        'use_inr': [False, True],  # Compare with/without INR
        'inr_hidden_dim': [32, 48, 64]  # INR capacity when enabled
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    # FILTER CONFIGURATIONS
    filtered_combinations = []
    
    for combo in combinations:
        config = dict(zip(keys, combo))
        
        # Only test INR hidden dim when INR is enabled
        if not config['use_inr'] and config['inr_hidden_dim'] != 32:
            # Skip all INR variants when use_inr=False (we only need 1 baseline)
            continue
        
        filtered_combinations.append(combo)
    
    print("="*80)
    print(f"REFINED GRID SEARCH")
    print(f"Total combinations: {len(combinations)}")
    print(f"After filtering: {len(filtered_combinations)}")
    print("="*80)
    
    results = []
    
    for idx, combo in enumerate(filtered_combinations):
        config = dict(zip(keys, combo))
        inr_str = f"_INR{config['inr_hidden_dim']}" if config['use_inr'] else "_NoINR"
        config_str = (
            f"nnUNet_BC{config['base_channels']}_"
            f"D{config['unet_depth']}"
            f"{inr_str}_"
            f"LR{config['learning_rate']:.0e}_"
            f"WD{config['weight_decay']:.0e}"
        )
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(filtered_combinations)}] TESTING CONFIGURATION")
        print(f"{'='*80}")
        print(f"Config: {config_str}")
        print(f"Parameters:")
        for k, v in config.items():
            if k not in ['num_epochs', 'use_amp']:
                print(f"  • {k:20s}: {v}")
        print()
        
        try:
            metrics, history, model = train_single_config(config, train_loader, val_loader, device)
            
            result = {
                'config': config,
                'config_str': config_str,
                'metrics': metrics,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            results.append(result)
            
            print(f"\n{'='*80}")
            print(f"✓ TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"Results:")
            print(f"  • Best Dice Score:       {metrics['best_dice']:.4f}")
            print(f"  • Dice (Anterior):       {metrics['final_dice_anterior']:.4f}")
            print(f"  • Dice (Posterior):      {metrics['final_dice_posterior']:.4f}")
            print(f"  • Mean NSD:              {metrics['mean_nsd']:.4f}")
            print(f"  • NSD (Anterior):        {metrics['final_nsd_anterior']:.4f}")
            print(f"  • NSD (Posterior):       {metrics['final_nsd_posterior']:.4f}")
            print(f"  • Total Parameters:      {metrics['num_params']:,}")
            print(f"{'='*80}")
            
            # Save individual result
            result_file = os.path.join(save_dir, f'result_{config_str}.json')
            result_to_save = {
                'config': config,
                'config_str': config_str,
                'metrics': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                           for k, v in metrics.items()},
                'timestamp': result['timestamp']
            }
            with open(result_file, 'w') as f:
                json.dump(result_to_save, f, indent=2)
            
            # Save model if top 5
            if len(results) >= 5:
                top_5_dice = sorted([r['metrics']['best_dice'] for r in results])[-5:]
                if metrics['best_dice'] >= top_5_dice[0]:
                    model_file = os.path.join(save_dir, f'model_{config_str}.pth')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'metrics': metrics
                    }, model_file)
                    print(f"  💾 Model saved")
        
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save all results
    all_results_file = os.path.join(save_dir, 'all_results.json')
    results_to_save = []
    for r in results:
        result_copy = {
            'config': r['config'],
            'config_str': r['config_str'],
            'metrics': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                       for k, v in r['metrics'].items()},
            'timestamp': r['timestamp']
        }
        results_to_save.append(result_copy)
    
    with open(all_results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Print summary
    if results:
        print(f"\n\n{'='*80}")
        print("GRID SEARCH SUMMARY")
        print(f"{'='*80}")
        print(f"Total configurations tested: {len(results)}")
        print(f"\nTop 5 Configurations by Dice Score:")
        print(f"{'-'*80}")
        sorted_results = sorted(results, key=lambda x: x['metrics']['best_dice'], reverse=True)[:5]
        for i, r in enumerate(sorted_results, 1):
            print(f"\n{i}. {r['config_str']}")
            print(f"   Dice: {r['metrics']['best_dice']:.4f} | "
                  f"NSD: {r['metrics']['mean_nsd']:.4f} | "
                  f"Params: {r['metrics']['num_params']:,}")
        print(f"\n{'='*80}")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_grid_search_results(results, save_dir='grid_search_results'):
    """Create visualizations of grid search results"""
    
    valid_results = [r for r in results if 'error' not in r['metrics']]
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    configs = [r['config'] for r in valid_results]
    metrics = [r['metrics'] for r in valid_results]
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Dice vs Parameters
    ax1 = plt.subplot(2, 3, 1)
    params = [m['num_params'] for m in metrics]
    dice_scores = [m['best_dice'] for m in metrics]
    scatter = ax1.scatter(params, dice_scores, c=dice_scores, cmap='viridis', s=100, alpha=0.6)
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Best Dice')
    ax1.set_title('Dice vs Model Size')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1)
    
    # 2. Top 10 configurations
    ax2 = plt.subplot(2, 3, 2)
    sorted_results = sorted(valid_results, key=lambda x: x['metrics']['best_dice'], reverse=True)[:10]
    top_names = [r['config_str'][:30] for r in sorted_results]
    top_dice = [r['metrics']['best_dice'] for r in sorted_results]
    
    y_pos = np.arange(len(top_names))
    ax2.barh(y_pos, top_dice, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_dice))))
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_names, fontsize=8)
    ax2.set_xlabel('Best Dice')
    ax2.set_title('Top 10 Configurations')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grid_search_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\n📊 Analysis saved to {save_dir}/grid_search_analysis.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = {
        'batch_size': 16,
        'dataset_dir': '/kaggle/input/task4-hippocampus-segmentation/dataset_2d',
        'save_dir': 'grid_search_results'
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
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
    
    # Run grid search
    print("\n" + "="*80)
    print("STARTING REFINED GRID SEARCH")
    print("="*80)
    
    results = refined_grid_search(train_loader, val_loader, device, config['save_dir'])
    
    # Visualize
    plot_grid_search_results(results, config['save_dir'])
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()