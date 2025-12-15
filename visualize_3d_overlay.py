"""
3D Hippocampus Segmentation - Overlay Visualization
Shows Ground Truth and Prediction overlaid to highlight differences
"""

import torch
import nibabel as nib
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import pyvista as pv
from pathlib import Path

from unet_inr_3d import UNet3DWithINRBalanced


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'checkpoint_path': 'results_hippocampus/best_model.pth',
    'test_volume': 'Task04_Hippocampus/imagesTr/hippocampus_008.nii.gz',
    'test_label': 'Task04_Hippocampus/labelsTr/hippocampus_008.nii.gz',
    'target_size': (128, 128, 128),
    'smoothing_sigma': 1.0,  # Increased smoothing
    'mesh_smoothing': 100,  # More iterations
    'save_image': True,
    'output_path': 'visualization_3d_overlay.png',
    'image_size': (1920, 1080)
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_model(checkpoint_path):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3DWithINRBalanced(in_channels=1, num_classes=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Use strict=False to ignore cached_coords buffer
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model, device


def normalize_intensity(img):
    """Normalize MRI intensity"""
    p1, p99 = np.percentile(img, [1, 99])
    img_clipped = np.clip(img, p1, p99)
    return (img_clipped - img_clipped.min()) / (img_clipped.max() - img_clipped.min() + 1e-8)


def run_inference(model, device, img_path):
    """Run model inference"""
    nii = nib.load(img_path)
    img = nii.get_fdata().astype(np.float32)  # Ensure float32!
    img_norm = normalize_intensity(img)
    # Make sure tensor is float32
    img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    
    return img_norm, pred


def create_smooth_mesh(mask, class_id, sigma=0.5, smooth_iter=30):
    """Create ultra-smooth 3D mesh"""
    binary_mask = (mask == class_id).astype(float)
    if binary_mask.sum() == 0:
        return None
    
    # Aggressive Gaussian smoothing
    smooth_mask = gaussian_filter(binary_mask, sigma=sigma)
    
    try:
        # Lower threshold for smoother surface
        verts, faces, _, _ = measure.marching_cubes(smooth_mask, level=0.3)
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, faces_pv)
        
        # Advanced smoothing pipeline
        mesh = mesh.smooth(
            n_iter=smooth_iter,
            relaxation_factor=0.1,
            feature_angle=120,
            boundary_smoothing=True,
            feature_smoothing=True
        )
        
        # Subdivide for smoother appearance
        mesh = mesh.subdivide(nsub=2, subfilter='butterfly')
        
        # Final smoothing pass
        mesh = mesh.smooth(n_iter=50, relaxation_factor=0.05)
        mesh = mesh.clean()
        
        return mesh
    except:
        return None


def compute_dice(pred, gt, class_id):
    """Compute Dice score"""
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)
    intersection = (pred_mask & gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    return (2.0 * intersection) / union if union > 0 else 1.0


# =============================================================================
# OVERLAY VISUALIZATION
# =============================================================================

def create_overlay_visualization(img, gt_mask, pred_mask, config):
    """
    Create overlay visualization showing:
    - True Positives (green): Correct predictions
    - False Positives (red): Over-segmentation
    - False Negatives (blue): Under-segmentation
    """
    from skimage.transform import resize
    
    target = config['target_size']
    img_resized = resize(img, target, order=1, preserve_range=True)
    gt_resized = resize(gt_mask, target, order=0, preserve_range=True)
    pred_resized = resize(pred_mask, target, order=0, preserve_range=True)
    
    # Compute metrics
    dice_ant = compute_dice(pred_mask, gt_mask, 1)
    dice_post = compute_dice(pred_mask, gt_mask, 2)
    dice_mean = (dice_ant + dice_post) / 2
    
    print(f"\n{'='*80}")
    print(f"SEGMENTATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Anterior Dice:  {dice_ant:.4f}")
    print(f"Posterior Dice: {dice_post:.4f}")
    print(f"Mean Dice:      {dice_mean:.4f}")
    print(f"{'='*80}\n")
    
    # Create separate masks for each class and category
    # Anterior (class 1)
    ant_tp = ((gt_resized == 1) & (pred_resized == 1)).astype(float)  # True positive
    ant_fp = ((gt_resized != 1) & (pred_resized == 1)).astype(float)  # False positive
    ant_fn = ((gt_resized == 1) & (pred_resized != 1)).astype(float)  # False negative
    
    # Posterior (class 2)
    post_tp = ((gt_resized == 2) & (pred_resized == 2)).astype(float)
    post_fp = ((gt_resized != 2) & (pred_resized == 2)).astype(float)
    post_fn = ((gt_resized == 2) & (pred_resized != 2)).astype(float)
    
    # Create plotter with 3 panels
    plotter = pv.Plotter(shape=(1, 3), window_size=(2400, 800))
    
    # =========================================================================
    # LEFT: Ground Truth
    # =========================================================================
    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth", position='upper_edge', font_size=14, color='black')
    
    gt_ant = create_smooth_mesh(gt_resized, 1, config['smoothing_sigma'], config['mesh_smoothing'])
    gt_post = create_smooth_mesh(gt_resized, 2, config['smoothing_sigma'], config['mesh_smoothing'])
    
    if gt_ant:
        plotter.add_mesh(gt_ant, color='#FF6B6B', opacity=0.9, smooth_shading=True)
    if gt_post:
        plotter.add_mesh(gt_post, color='#4ECDC4', opacity=0.9, smooth_shading=True)
    
    plotter.camera_position = [(250, 200, 200), (64, 64, 64), (0, 0, 1)]
    plotter.set_background('white')
    plotter.add_light(pv.Light(position=(200, 200, 200)))
    
    # =========================================================================
    # MIDDLE: Prediction
    # =========================================================================
    plotter.subplot(0, 1)
    plotter.add_text(f"Prediction (Dice: {dice_mean:.3f})", position='upper_edge', font_size=14, color='black')
    
    pred_ant = create_smooth_mesh(pred_resized, 1, config['smoothing_sigma'], config['mesh_smoothing'])
    pred_post = create_smooth_mesh(pred_resized, 2, config['smoothing_sigma'], config['mesh_smoothing'])
    
    if pred_ant:
        plotter.add_mesh(pred_ant, color='#FF6B6B', opacity=0.9, smooth_shading=True)
    if pred_post:
        plotter.add_mesh(pred_post, color='#4ECDC4', opacity=0.9, smooth_shading=True)
    
    plotter.camera_position = [(250, 200, 200), (64, 64, 64), (0, 0, 1)]
    plotter.set_background('white')
    plotter.add_light(pv.Light(position=(200, 200, 200)))
    
    # =========================================================================
    # RIGHT: Error Analysis
    # =========================================================================
    plotter.subplot(0, 2)
    plotter.add_text("Error Analysis", position='upper_edge', font_size=14, color='black')
    
    # True Positives (Green) - Correct
    ant_tp_mesh = create_smooth_mesh(ant_tp.astype(int), 1, config['smoothing_sigma'], 20)
    post_tp_mesh = create_smooth_mesh(post_tp.astype(int), 1, config['smoothing_sigma'], 20)
    
    # False Positives (Red) - Over-segmentation
    ant_fp_mesh = create_smooth_mesh(ant_fp.astype(int), 1, config['smoothing_sigma'], 20)
    post_fp_mesh = create_smooth_mesh(post_fp.astype(int), 1, config['smoothing_sigma'], 20)
    
    # False Negatives (Blue) - Under-segmentation
    ant_fn_mesh = create_smooth_mesh(ant_fn.astype(int), 1, config['smoothing_sigma'], 20)
    post_fn_mesh = create_smooth_mesh(post_fn.astype(int), 1, config['smoothing_sigma'], 20)
    
    # Add meshes with color coding
    if ant_tp_mesh or post_tp_mesh:
        if ant_tp_mesh:
            plotter.add_mesh(ant_tp_mesh, color='#2ECC71', opacity=0.9, smooth_shading=True)  # Green
        if post_tp_mesh:
            plotter.add_mesh(post_tp_mesh, color='#2ECC71', opacity=0.9, smooth_shading=True)
    
    if ant_fp_mesh or post_fp_mesh:
        if ant_fp_mesh:
            plotter.add_mesh(ant_fp_mesh, color='#E74C3C', opacity=0.9, smooth_shading=True)  # Red
        if post_fp_mesh:
            plotter.add_mesh(post_fp_mesh, color='#E74C3C', opacity=0.9, smooth_shading=True)
    
    if ant_fn_mesh or post_fn_mesh:
        if ant_fn_mesh:
            plotter.add_mesh(ant_fn_mesh, color='#3498DB', opacity=0.9, smooth_shading=True)  # Blue
        if post_fn_mesh:
            plotter.add_mesh(post_fn_mesh, color='#3498DB', opacity=0.9, smooth_shading=True)
    
    plotter.camera_position = [(250, 200, 200), (64, 64, 64), (0, 0, 1)]
    plotter.set_background('white')
    plotter.add_light(pv.Light(position=(200, 200, 200)))
    
    # Add legend
    plotter.add_legend(
        labels=[
            ['Correct (TP)', '#2ECC71'],
            ['Over-seg (FP)', '#E74C3C'],
            ['Under-seg (FN)', '#3498DB']
        ],
        bcolor='white',
        border=True,
        size=(0.2, 0.15)
    )
    
    # Link cameras
    plotter.link_views()
    
    return plotter


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("3D OVERLAY VISUALIZATION WITH ERROR ANALYSIS")
    print("="*80)
    
    print("\n[1/4] Loading model...")
    model, device = load_model(CONFIG['checkpoint_path'])
    
    print("\n[2/4] Running inference...")
    img, pred_mask = run_inference(model, device, CONFIG['test_volume'])
    
    print("\n[3/4] Loading ground truth...")
    gt_mask = nib.load(CONFIG['test_label']).get_fdata().astype(np.int64)
    
    print("\n[4/4] Creating overlay visualization...")
    plotter = create_overlay_visualization(img, gt_mask, pred_mask, CONFIG)
    
    if CONFIG['save_image']:
        print(f"\nSaving to {CONFIG['output_path']}...")
        plotter.show(screenshot=CONFIG['output_path'], window_size=CONFIG['image_size'])
        print(f"✓ Saved to {CONFIG['output_path']}")
    else:
        print("\nDisplaying interactive visualization...")
        plotter.show()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()