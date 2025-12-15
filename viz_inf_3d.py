"""
3D Hippocampus Segmentation Visualization
Compares Ground Truth vs Model Prediction side-by-side
Professional medical visualization
"""

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import pyvista as pv
from pathlib import Path

# Import your model
from unet_inr_3d import UNet3DWithINRBalanced


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'checkpoint_path': 'results_hippocampus/best_model.pth',
    'test_volume': 'Task04_Hippocampus/imagesTr/hippocampus_008.nii.gz',
    'test_label': 'Task04_Hippocampus/labelsTr/hippocampus_008.nii.gz',
    
    # Visualization - BALANCED SMOOTH (ORGANIC BUT SAFE)
    'target_size': (128, 128, 128),
    'smoothing_sigma': 1.5,  # Balanced (not too aggressive)
    'mesh_smoothing_iter': 100,
    'subdivide_levels': 2,  # 2 levels = 4x vertices
    'final_smooth_iter': 80,
    'marching_cubes_level': 0.35,  # Higher to preserve structure
    
    # Colors (medical imaging standard)
    'color_anterior': '#FF6B6B',
    'color_posterior': '#4ECDC4',
    'color_brain': '#95A5A6',
    
    # Save output
    'save_image': True,
    'output_path': 'visualization_3d_comparison.png',
    'image_size': (1920, 1080)
}


# =============================================================================
# MODEL INFERENCE
# =============================================================================

def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNet3DWithINRBalanced(in_channels=1, num_classes=3).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict with strict=False to ignore cached_coords
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"  Best Dice: {checkpoint['best_dice']:.4f}")
    
    return model, device


def normalize_intensity(img):
    """Normalize MRI intensity to [0, 1]"""
    p1, p99 = np.percentile(img, [1, 99])
    img_clipped = np.clip(img, p1, p99)
    return (img_clipped - img_clipped.min()) / (img_clipped.max() - img_clipped.min() + 1e-8)


def run_inference(model, device, img_path):
    """Run model inference on a volume"""
    # Load image
    nii = nib.load(img_path)
    img = nii.get_fdata().astype(np.float32)  # Ensure float32!
    
    # Normalize
    img_norm = normalize_intensity(img)
    
    # Convert to tensor (make sure it's float32)
    img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    
    return img_norm, pred


# =============================================================================
# 3D MESH GENERATION
# =============================================================================

def create_smooth_mesh(mask, class_id, config):
    """
    Create SMOOTH mesh for predictions
    Organic appearance without blocky edges
    
    Args:
        mask: 3D segmentation mask
        class_id: Which class to extract (1 or 2)
        config: Configuration dictionary
    
    Returns:
        PyVista mesh with smooth, organic appearance
    """
    # Extract binary mask for this class
    binary_mask = (mask == class_id).astype(float)
    
    if binary_mask.sum() == 0:
        return None
    
    # STEP 1: Strong Gaussian blur for smooth surface
    smooth_mask = gaussian_filter(binary_mask, sigma=config['smoothing_sigma'])
    
    try:
        # STEP 2: Marching cubes
        verts, faces, _, _ = measure.marching_cubes(
            smooth_mask, 
            level=config['marching_cubes_level']
        )
        
        # Safety check
        if len(verts) < 10:
            print(f"  Warning: Very few vertices for class {class_id}, skipping")
            return None
        
        # Create PyVista mesh
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, faces_pv)
        
        # STEP 3: Initial Taubin smoothing (preserves volume)
        mesh = mesh.smooth_taubin(
            n_iter=config['mesh_smoothing_iter'],
            pass_band=0.1
        )
        
        # Check mesh is still valid
        if mesh.n_points < 10:
            print(f"  Warning: Mesh collapsed after smoothing for class {class_id}")
            return None
        
        # STEP 4: Subdivide for smoother base
        for _ in range(config['subdivide_levels']):
            mesh = mesh.subdivide(1, subfilter='butterfly')
        
        # STEP 5: Heavy Laplacian smoothing (removes blocky look)
        mesh = mesh.smooth(
            n_iter=config['final_smooth_iter'],
            relaxation_factor=0.15,
            feature_angle=160,
            boundary_smoothing=True,
            feature_smoothing=False
        )
        
        # STEP 6: Additional Taubin for organic finish
        mesh = mesh.smooth_taubin(
            n_iter=60,
            pass_band=0.05
        )
        
        # STEP 7: Final gentle polish
        mesh = mesh.smooth(
            n_iter=40,
            relaxation_factor=0.1
        )
        
        # STEP 8: Clean
        mesh = mesh.clean()
        
        # Final safety check
        if mesh.n_points < 10:
            print(f"  Warning: Final mesh too small for class {class_id}")
            return None
        
        return mesh
        
    except Exception as e:
        print(f"  Error creating mesh for class {class_id}: {e}")
        return None


def create_blocky_mesh(mask, class_id):
    """
    Create BLOCKY mesh for ground truth
    Minimal smoothing - keeps voxel-like appearance
    
    Args:
        mask: 3D segmentation mask
        class_id: Which class to extract (1 or 2)
    
    Returns:
        PyVista mesh with blocky, lego-like appearance
    """
    # Extract binary mask for this class
    binary_mask = (mask == class_id).astype(float)
    
    if binary_mask.sum() == 0:
        return None
    
    # Very light smoothing (just to remove extreme jaggedness)
    smooth_mask = gaussian_filter(binary_mask, sigma=0.3)
    
    try:
        # Marching cubes with high threshold (preserves blocks)
        verts, faces, _, _ = measure.marching_cubes(smooth_mask, level=0.5)
        
        if len(verts) < 10:
            return None
        
        # Create PyVista mesh
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, faces_pv)
        
        # MINIMAL smoothing - just clean up artifacts
        mesh = mesh.smooth(
            n_iter=10,  # Very few iterations
            relaxation_factor=0.05,  # Very gentle
            feature_angle=90,  # Preserve edges
            boundary_smoothing=False
        )
        
        mesh = mesh.clean()
        
        if mesh.n_points < 10:
            return None
        
        return mesh
        
    except Exception as e:
        print(f"  Error creating blocky mesh for class {class_id}: {e}")
        return None


def create_brain_mesh(img, config):
    """Create smooth brain surface mesh with safety checks"""
    threshold = 0.15
    brain_mask = img > threshold
    brain_smooth = gaussian_filter(brain_mask.astype(float), sigma=1.5)
    
    try:
        verts, faces, _, _ = measure.marching_cubes(brain_smooth, level=0.4)
        
        if len(verts) < 10:
            return None
        
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, faces_pv)
        
        # Balanced smoothing
        mesh = mesh.smooth_taubin(n_iter=80, pass_band=0.1)
        mesh = mesh.subdivide(2, subfilter='butterfly')
        mesh = mesh.smooth_taubin(n_iter=60, pass_band=0.08)
        mesh = mesh.smooth(n_iter=60, relaxation_factor=0.15, feature_angle=160)
        mesh = mesh.clean()
        
        if mesh.n_points < 10:
            return None
        
        return mesh
    except:
        return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def compute_dice(pred, gt, class_id):
    """Compute Dice score for a specific class"""
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)
    
    intersection = (pred_mask & gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / union


def create_side_by_side_visualization(
    img, 
    gt_mask, 
    pred_mask,
    config
):
    """
    Create professional side-by-side comparison
    
    Left: Ground Truth
    Right: Model Prediction
    """
    
    # Resize volumes for visualization
    from skimage.transform import resize
    
    target = config['target_size']
    img_resized = resize(img, target, order=1, preserve_range=True)
    gt_resized = resize(gt_mask, target, order=0, preserve_range=True)
    pred_resized = resize(pred_mask, target, order=0, preserve_range=True)
    
    # Normalize image
    img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
    img_enhanced = np.clip(img_norm * 1.5, 0, 1)
    
    # Compute metrics
    dice_ant_gt = compute_dice(pred_mask, gt_mask, 1)
    dice_post_gt = compute_dice(pred_mask, gt_mask, 2)
    dice_mean = (dice_ant_gt + dice_post_gt) / 2
    
    print(f"\n{'='*80}")
    print(f"SEGMENTATION METRICS")
    print(f"{'='*80}")
    print(f"Anterior Hippocampus Dice:  {dice_ant_gt:.4f}")
    print(f"Posterior Hippocampus Dice: {dice_post_gt:.4f}")
    print(f"Mean Dice Score:            {dice_mean:.4f}")
    print(f"{'='*80}\n")
    
    # Create meshes with different styles
    print("Generating 3D meshes...")
    print("  Ground Truth: Blocky (voxel-like)")
    print("  Prediction: Smooth (organic)")
    
    brain_mesh = create_brain_mesh(img_enhanced, config)
    
    # Ground Truth meshes - BLOCKY/LEGO-LIKE
    gt_ant_mesh = create_blocky_mesh(gt_resized, 1)
    gt_post_mesh = create_blocky_mesh(gt_resized, 2)
    
    # Prediction meshes - SMOOTH/ORGANIC
    pred_ant_mesh = create_smooth_mesh(pred_resized, 1, config)
    pred_post_mesh = create_smooth_mesh(pred_resized, 2, config)
    
    # Create side-by-side plotter
    plotter = pv.Plotter(shape=(1, 2), window_size=config['image_size'])
    
    # =========================================================================
    # LEFT PANEL: GROUND TRUTH
    # =========================================================================
    plotter.subplot(0, 0)
    plotter.add_text(
        "Ground Truth (Original Voxels)",
        position='upper_edge',
        font_size=16,
        color='black',
        font='arial'
    )
    
    # Brain surface (semi-transparent)
    if brain_mesh is not None and brain_mesh.n_points > 0:
        plotter.add_mesh(
            brain_mesh,
            color=config['color_brain'],
            opacity=0.15,
            smooth_shading=True,
            show_edges=False
        )
    
    # Anterior hippocampus (red)
    if gt_ant_mesh is not None and gt_ant_mesh.n_points > 0:
        plotter.add_mesh(
            gt_ant_mesh,
            color=config['color_anterior'],
            opacity=0.95,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.7,
            specular=0.3,
            specular_power=15,
            label='Anterior'
        )
        # Contour
        try:
            edges = gt_ant_mesh.extract_feature_edges()
            if edges.n_points > 0:
                plotter.add_mesh(edges, color='darkred', line_width=2)
        except:
            pass
    
    # Posterior hippocampus (cyan)
    if gt_post_mesh is not None and gt_post_mesh.n_points > 0:
        plotter.add_mesh(
            gt_post_mesh,
            color=config['color_posterior'],
            opacity=0.95,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.7,
            specular=0.3,
            specular_power=15,
            label='Posterior'
        )
        # Contour
        try:
            edges = gt_post_mesh.extract_feature_edges()
            if edges.n_points > 0:
                plotter.add_mesh(edges, color='teal', line_width=2)
        except:
            pass
    
    # Lighting
    plotter.add_light(pv.Light(position=(200, 200, 200), light_type='scene light'))
    plotter.add_light(pv.Light(position=(-200, -200, 200), light_type='scene light', intensity=0.5))
    
    # Camera
    plotter.camera_position = [
        (250, 200, 200),
        (64, 64, 64),
        (0, 0, 1)
    ]
    
    plotter.set_background('white')
    
    # =========================================================================
    # RIGHT PANEL: PREDICTION
    # =========================================================================
    plotter.subplot(0, 1)
    plotter.add_text(
        f"Prediction - Smoothed (Dice: {dice_mean:.3f})",
        position='upper_edge',
        font_size=16,
        color='black',
        font='arial'
    )
    
    # Brain surface
    if brain_mesh is not None and brain_mesh.n_points > 0:
        plotter.add_mesh(
            brain_mesh,
            color=config['color_brain'],
            opacity=0.15,
            smooth_shading=True,
            show_edges=False
        )
    
    # Predicted anterior
    if pred_ant_mesh is not None and pred_ant_mesh.n_points > 0:
        plotter.add_mesh(
            pred_ant_mesh,
            color=config['color_anterior'],
            opacity=0.95,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.7,
            specular=0.3,
            specular_power=15,
            label=f'Anterior (Dice: {dice_ant_gt:.3f})'
        )
        try:
            edges = pred_ant_mesh.extract_feature_edges()
            if edges.n_points > 0:
                plotter.add_mesh(edges, color='darkred', line_width=2)
        except:
            pass
    
    # Predicted posterior
    if pred_post_mesh is not None and pred_post_mesh.n_points > 0:
        plotter.add_mesh(
            pred_post_mesh,
            color=config['color_posterior'],
            opacity=0.95,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.7,
            specular=0.3,
            specular_power=15,
            label=f'Posterior (Dice: {dice_post_gt:.3f})'
        )
        try:
            edges = pred_post_mesh.extract_feature_edges()
            if edges.n_points > 0:
                plotter.add_mesh(edges, color='teal', line_width=2)
        except:
            pass
    
    # Lighting
    plotter.add_light(pv.Light(position=(200, 200, 200), light_type='scene light'))
    plotter.add_light(pv.Light(position=(-200, -200, 200), light_type='scene light', intensity=0.5))
    
    # Camera (same as left panel)
    plotter.camera_position = [
        (250, 200, 200),
        (64, 64, 64),
        (0, 0, 1)
    ]
    
    plotter.set_background('white')
    
    # Add legend to right panel
    plotter.add_legend(
        labels=[
            ['Anterior Hippocampus', config['color_anterior']],
            ['Posterior Hippocampus', config['color_posterior']]
        ],
        bcolor='white',
        border=True,
        size=(0.15, 0.15)
    )
    
    # Link cameras for synchronized rotation
    plotter.link_views()
    
    return plotter


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("3D HIPPOCAMPUS SEGMENTATION VISUALIZATION")
    print("="*80)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, device = load_model(CONFIG['checkpoint_path'])
    
    # Run inference
    print("\n[2/4] Running inference...")
    img, pred_mask = run_inference(model, device, CONFIG['test_volume'])
    
    # Load ground truth
    print("\n[3/4] Loading ground truth...")
    gt_mask = nib.load(CONFIG['test_label']).get_fdata().astype(np.int64)
    
    # Create visualization
    print("\n[4/4] Creating ultra-smooth 3D visualization...")
    print("  ⚠ This may take 1-2 minutes due to aggressive smoothing...")
    print("  Creating organic, stone-like surfaces (no blocky edges)...")
    plotter = create_side_by_side_visualization(img, gt_mask, pred_mask, CONFIG)
    
    # Save or show
    if CONFIG['save_image']:
        print(f"\nSaving to {CONFIG['output_path']}...")
        plotter.show(screenshot=CONFIG['output_path'], window_size=CONFIG['image_size'])
        print(f"✓ Saved to {CONFIG['output_path']}")
    else:
        print("\nDisplaying interactive visualization...")
        print("  • Use mouse to rotate")
        print("  • Scroll to zoom")
        print("  • Press 'q' to quit")
        plotter.show()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()