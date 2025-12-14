import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import random

# =============================================================================
# CONFIG
# =============================================================================

IMG_DIR = "dataset_2d/images"
MASK_DIR = "dataset_2d/masks"

# Professional color scheme
COLORS = {
    'background': '#1a1a2e',
    'text': '#eaeaea',
    'accent': '#16213e',
    'highlight': '#0f3460'
}

# Hippocampus segmentation colormap
# 0: background (transparent/dark), 1: anterior (cyan), 2: posterior (magenta)
hippo_colors = ['#000000', '#00d9ff', '#ff006e']
hippo_cmap = ListedColormap(hippo_colors)

# =============================================================================
# LOAD SAMPLES
# =============================================================================

def load_sample(idx):
    """Load image and mask for a given index"""
    img_path = os.path.join(IMG_DIR, f"img_{idx:06d}.npy")
    mask_path = os.path.join(MASK_DIR, f"mask_{idx:06d}.npy")
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return None, None
    
    img = np.load(img_path)
    mask = np.load(mask_path)
    
    return img, mask


def get_random_samples(n_samples=9, ensure_variety=True):
    """Get random samples, optionally ensuring they have hippocampus"""
    all_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.npy')])
    total = len(all_files)
    
    samples = []
    
    if ensure_variety:
        # Try to get samples with hippocampus
        attempts = 0
        while len(samples) < n_samples and attempts < total:
            idx = random.randint(0, total - 1)
            img, mask = load_sample(idx)
            
            if img is not None and np.max(mask) > 0:  # Has hippocampus
                samples.append((idx, img, mask))
            
            attempts += 1
    else:
        # Just get random samples
        indices = random.sample(range(total), min(n_samples, total))
        for idx in indices:
            img, mask = load_sample(idx)
            if img is not None:
                samples.append((idx, img, mask))
    
    return samples


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_overlay(img, mask, alpha=0.5):
    """Create RGB overlay of image with colored mask"""
    # Normalize image to 0-1
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Create RGB image
    rgb = np.stack([img_norm] * 3, axis=-1)
    
    # Create mask overlay
    mask_rgb = np.zeros((*mask.shape, 3))
    mask_rgb[mask == 1] = [0, 0.85, 1.0]  # Cyan for anterior
    mask_rgb[mask == 2] = [1.0, 0, 0.43]  # Magenta for posterior
    
    # Blend
    overlay = rgb.copy()
    mask_present = mask > 0
    overlay[mask_present] = (1 - alpha) * rgb[mask_present] + alpha * mask_rgb[mask_present]
    
    return overlay


def plot_elegant_grid(samples, save_path=None):
    """Create elegant grid visualization of samples"""
    n_samples = len(samples)
    n_rows = int(np.ceil(n_samples / 3))
    
    # Create figure with dark background
    fig = plt.figure(figsize=(18, 6 * n_rows), facecolor=COLORS['background'])
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.3, wspace=0.15,
                  left=0.05, right=0.95, top=0.94, bottom=0.06)
    
    # Title
    fig.suptitle('Hippocampus 2D Segmentation Dataset', 
                 fontsize=28, fontweight='bold', color=COLORS['text'],
                 y=0.98)
    
    for idx, (sample_idx, img, mask) in enumerate(samples):
        row = idx // 3
        col = idx % 3
        
        # Create subplot with 1 row, 3 columns for this sample
        ax_main = fig.add_subplot(gs[row, col])
        
        # Create inner grid for this sample
        inner_gs = GridSpec(1, 3, figure=fig, 
                           hspace=0.02, wspace=0.05,
                           left=gs[row, col].get_position(fig).x0,
                           right=gs[row, col].get_position(fig).x1,
                           top=gs[row, col].get_position(fig).y1,
                           bottom=gs[row, col].get_position(fig).y0)
        
        # Remove main axis
        ax_main.remove()
        
        # Original image
        ax1 = fig.add_subplot(inner_gs[0, 0])
        ax1.imshow(img, cmap='gray', interpolation='bilinear')
        ax1.set_title('Original', fontsize=11, color=COLORS['text'], pad=8)
        ax1.axis('off')
        
        # Overlay
        ax2 = fig.add_subplot(inner_gs[0, 1])
        overlay = create_overlay(img, mask, alpha=0.6)
        ax2.imshow(overlay, interpolation='bilinear')
        ax2.set_title('Overlay', fontsize=11, color=COLORS['text'], pad=8)
        ax2.axis('off')
        
        # Mask only
        ax3 = fig.add_subplot(inner_gs[0, 2])
        ax3.imshow(mask, cmap=hippo_cmap, vmin=0, vmax=2, interpolation='nearest')
        ax3.set_title('Segmentation', fontsize=11, color=COLORS['text'], pad=8)
        ax3.axis('off')
        
        # Sample index label
        fig.text(gs[row, col].get_position(fig).x0 + 0.01,
                gs[row, col].get_position(fig).y0 + 0.01,
                f'Sample #{sample_idx:04d}',
                fontsize=9, color='#888888',
                ha='left', va='bottom')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#00d9ff', edgecolor='white', label='Anterior Hippocampus'),
        mpatches.Patch(facecolor='#ff006e', edgecolor='white', label='Posterior Hippocampus')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=2, fontsize=13, frameon=False,
              labelcolor=COLORS['text'], bbox_to_anchor=(0.5, 0.01))
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=COLORS['background'], 
                   bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


def plot_single_detailed(sample_idx, save_path=None):
    """Create detailed view of a single sample"""
    img, mask = load_sample(sample_idx)
    
    if img is None:
        print(f"Sample {sample_idx} not found!")
        return
    
    # Create figure
    fig = plt.figure(figsize=(20, 6), facecolor=COLORS['background'])
    gs = GridSpec(1, 4, figure=fig, wspace=0.25, 
                  left=0.05, right=0.95, top=0.88, bottom=0.12)
    
    # Title
    fig.suptitle(f'Detailed View - Sample #{sample_idx:04d}', 
                 fontsize=24, fontweight='bold', color=COLORS['text'])
    
    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(img, cmap='gray', interpolation='bilinear')
    ax1.set_title('MRI Slice', fontsize=16, color=COLORS['text'], pad=15)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Overlay with transparency
    ax2 = fig.add_subplot(gs[0, 1])
    overlay = create_overlay(img, mask, alpha=0.5)
    ax2.imshow(overlay, interpolation='bilinear')
    ax2.set_title('Overlay (50% opacity)', fontsize=16, color=COLORS['text'], pad=15)
    ax2.axis('off')
    
    # 3. Segmentation mask
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(mask, cmap=hippo_cmap, vmin=0, vmax=2, interpolation='nearest')
    ax3.set_title('Ground Truth Segmentation', fontsize=16, color=COLORS['text'], pad=15)
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar3.ax.set_yticklabels(['Background', 'Anterior', 'Posterior'])
    cbar3.ax.tick_params(colors=COLORS['text'])
    
    # 4. Statistics panel
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    
    # Compute statistics
    total_pixels = img.size
    anterior_pixels = np.sum(mask == 1)
    posterior_pixels = np.sum(mask == 2)
    hippo_pixels = anterior_pixels + posterior_pixels
    
    anterior_pct = (anterior_pixels / total_pixels) * 100
    posterior_pct = (posterior_pixels / total_pixels) * 100
    hippo_pct = (hippo_pixels / total_pixels) * 100
    
    stats_text = f"""
STATISTICS

Image Shape: {img.shape[0]} × {img.shape[1]}
Intensity Range: [{img.min():.3f}, {img.max():.3f}]

SEGMENTATION

Total Hippocampus: {hippo_pct:.2f}%
├─ Anterior: {anterior_pct:.2f}%
└─ Posterior: {posterior_pct:.2f}%

PIXEL COUNTS

Background: {total_pixels - hippo_pixels:,}
Anterior: {anterior_pixels:,}
Posterior: {posterior_pixels:,}
    """
    
    ax4.text(0.1, 0.5, stats_text.strip(), 
            fontsize=13, color=COLORS['text'], 
            verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=1', 
                     facecolor=COLORS['accent'], 
                     edgecolor=COLORS['text'],
                     linewidth=2))
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=COLORS['background'],
                   bbox_inches='tight')
        print(f"✓ Saved detailed view to {save_path}")
    
    plt.show()


def plot_dataset_overview(n_samples=20, save_path=None):
    """Create compact overview of many samples"""
    samples = get_random_samples(n_samples, ensure_variety=True)
    
    n_cols = 5
    n_rows = int(np.ceil(len(samples) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows),
                            facecolor=COLORS['background'])
    fig.suptitle('Dataset Overview - Random Samples', 
                fontsize=26, fontweight='bold', color=COLORS['text'], y=0.995)
    
    axes = axes.flatten()
    
    for idx, (sample_idx, img, mask) in enumerate(samples):
        ax = axes[idx]
        
        # Show overlay
        overlay = create_overlay(img, mask, alpha=0.7)
        ax.imshow(overlay, interpolation='bilinear')
        ax.set_title(f'#{sample_idx:04d}', fontsize=10, color=COLORS['text'])
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#00d9ff', edgecolor='white', label='Anterior'),
        mpatches.Patch(facecolor='#ff006e', edgecolor='white', label='Posterior')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=2, fontsize=14, frameon=False,
              labelcolor=COLORS['text'], bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=COLORS['background'],
                   bbox_inches='tight')
        print(f"✓ Saved overview to {save_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HIPPOCAMPUS 2D DATASET VISUALIZATION")
    print("=" * 70)
    
    # Set style
    plt.style.use('dark_background')
    random.seed(42)
    
    # Count total samples
    total_samples = len([f for f in os.listdir(IMG_DIR) if f.endswith('.npy')])
    print(f"\n📊 Total samples in dataset: {total_samples}")
    
    # 1. Elegant grid view (9 samples)
    print("\n🎨 Creating elegant grid view...")
    samples = get_random_samples(9, ensure_variety=True)
    plot_elegant_grid(samples, save_path="viz_elegant_grid.png")
    
    # 2. Detailed single view
    print("\n🔍 Creating detailed single sample view...")
    # Pick a random sample with hippocampus
    detailed_samples = get_random_samples(1, ensure_variety=True)
    if detailed_samples:
        sample_idx = detailed_samples[0][0]
        plot_single_detailed(sample_idx, save_path="viz_detailed_sample.png")
    
    # 3. Dataset overview (20 samples)
    print("\n📋 Creating dataset overview...")
    plot_dataset_overview(n_samples=20, save_path="viz_dataset_overview.png")
    
    print("\n" + "=" * 70)
    print("✓ All visualizations complete!")
    print("=" * 70)
