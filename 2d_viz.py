import nibabel as nib 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from skimage import transform, filters
from scipy.ndimage import gaussian_filter

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================

# Load MRI and mask
img = nib.load("Task04_Hippocampus/imagesTr/hippocampus_008.nii.gz").get_fdata() 
mask = nib.load("Task04_Hippocampus/labelsTr/hippocampus_008.nii.gz").get_fdata() 

# Rescale to isotropic voxels
img_resized = transform.resize(img, (128,128,128), order=1, preserve_range=True) 
mask_resized = transform.resize(mask, (128,128,128), order=0, preserve_range=True) 

# Normalize image to [0, 1]
img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())

# Apply contrast enhancement using histogram equalization
from skimage import exposure
img_enhanced = exposure.equalize_adapthist(img_norm, clip_limit=0.03)

# Binary hippocampus mask
hippo_mask = (mask_resized > 0).astype(float)

# Smooth mask edges for better visualization
hippo_smooth = gaussian_filter(hippo_mask, sigma=0.8)

# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def create_professional_slices(img_data, mask_data, output_filename="brain_slices.png"):
    """
    Create professional medical imaging visualization with three orthogonal views.
    
    Parameters:
    -----------
    img_data : ndarray
        3D MRI image data (normalized 0-1)
    mask_data : ndarray
        3D segmentation mask (smoothed, 0-1)
    output_filename : str
        Output file name for saving
    """
    
    # Select slice positions (middle and key slices)
    axial_slices = [40, 64, 88]      # Bottom to top (Z-axis)
    coronal_slices = [50, 64, 78]    # Front to back (Y-axis)
    sagittal_slices = [50, 64, 78]   # Left to right (X-axis)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    fig.suptitle('Hippocampus Segmentation - Multi-Planar Reconstruction (MPR)', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # Create grid layout
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    # Color settings
    overlay_color = plt.cm.Reds  # Red colormap for hippocampus
    brain_cmap = 'gray'
    
    # ========================================================================
    # AXIAL VIEW (Top row)
    # ========================================================================
    for idx, slice_num in enumerate(axial_slices):
        ax = fig.add_subplot(gs[0, idx])
        
        # Get axial slice
        brain_slice = img_data[:, :, slice_num]
        mask_slice = mask_data[:, :, slice_num]
        
        # Display brain
        ax.imshow(brain_slice.T, cmap=brain_cmap, origin='lower', 
                  vmin=0, vmax=1, aspect='equal')
        
        # Overlay hippocampus with transparency
        mask_overlay = np.ma.masked_where(mask_slice < 0.1, mask_slice)
        im = ax.imshow(mask_overlay.T, cmap=overlay_color, origin='lower',
                       alpha=0.6, vmin=0, vmax=1, aspect='equal')
        
        # Add contour for clear boundaries
        if mask_slice.max() > 0.5:
            contours = ax.contour(mask_slice.T, levels=[0.5], colors='red', 
                                  linewidths=2, origin='lower')
        
        ax.set_title(f'Axial Slice {slice_num}/128', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (Left → Right)', fontsize=11)
        ax.set_ylabel('Y (Posterior → Anterior)', fontsize=11)
        ax.grid(False)
        
        # Add orientation markers
        ax.text(0.02, 0.98, 'L', transform=ax.transAxes, fontsize=12, 
                color='cyan', fontweight='bold', va='top')
        ax.text(0.98, 0.98, 'R', transform=ax.transAxes, fontsize=12, 
                color='cyan', fontweight='bold', va='top', ha='right')
    
    # Add 3D orientation diagram for axial
    ax_orient = fig.add_subplot(gs[0, 3])
    ax_orient.text(0.5, 0.7, '⊗', fontsize=60, ha='center', color='navy')
    ax_orient.text(0.5, 0.3, 'AXIAL VIEW\n(Looking Down)', fontsize=12, 
                   ha='center', fontweight='bold')
    ax_orient.text(0.5, 0.1, 'Superior → Inferior', fontsize=10, 
                   ha='center', style='italic', color='gray')
    ax_orient.axis('off')
    
    # ========================================================================
    # CORONAL VIEW (Middle row)
    # ========================================================================
    for idx, slice_num in enumerate(coronal_slices):
        ax = fig.add_subplot(gs[1, idx])
        
        # Get coronal slice
        brain_slice = img_data[:, slice_num, :]
        mask_slice = mask_data[:, slice_num, :]
        
        # Display brain
        ax.imshow(brain_slice.T, cmap=brain_cmap, origin='lower',
                  vmin=0, vmax=1, aspect='equal')
        
        # Overlay hippocampus
        mask_overlay = np.ma.masked_where(mask_slice < 0.1, mask_slice)
        ax.imshow(mask_overlay.T, cmap=overlay_color, origin='lower',
                  alpha=0.6, vmin=0, vmax=1, aspect='equal')
        
        # Add contour
        if mask_slice.max() > 0.5:
            ax.contour(mask_slice.T, levels=[0.5], colors='red', 
                      linewidths=2, origin='lower')
        
        ax.set_title(f'Coronal Slice {slice_num}/128', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (Left → Right)', fontsize=11)
        ax.set_ylabel('Z (Inferior → Superior)', fontsize=11)
        ax.grid(False)
        
        # Orientation markers
        ax.text(0.02, 0.98, 'L', transform=ax.transAxes, fontsize=12, 
                color='cyan', fontweight='bold', va='top')
        ax.text(0.98, 0.98, 'R', transform=ax.transAxes, fontsize=12, 
                color='cyan', fontweight='bold', va='top', ha='right')
    
    # Add 3D orientation diagram for coronal
    ax_orient = fig.add_subplot(gs[1, 3])
    ax_orient.text(0.5, 0.7, '◉', fontsize=60, ha='center', color='navy')
    ax_orient.text(0.5, 0.3, 'CORONAL VIEW\n(Looking Forward)', fontsize=12, 
                   ha='center', fontweight='bold')
    ax_orient.text(0.5, 0.1, 'Anterior → Posterior', fontsize=10, 
                   ha='center', style='italic', color='gray')
    ax_orient.axis('off')
    
    # ========================================================================
    # SAGITTAL VIEW (Bottom row)
    # ========================================================================
    for idx, slice_num in enumerate(sagittal_slices):
        ax = fig.add_subplot(gs[2, idx])
        
        # Get sagittal slice
        brain_slice = img_data[slice_num, :, :]
        mask_slice = mask_data[slice_num, :, :]
        
        # Display brain
        ax.imshow(brain_slice.T, cmap=brain_cmap, origin='lower',
                  vmin=0, vmax=1, aspect='equal')
        
        # Overlay hippocampus
        mask_overlay = np.ma.masked_where(mask_slice < 0.1, mask_slice)
        ax.imshow(mask_overlay.T, cmap=overlay_color, origin='lower',
                  alpha=0.6, vmin=0, vmax=1, aspect='equal')
        
        # Add contour
        if mask_slice.max() > 0.5:
            ax.contour(mask_slice.T, levels=[0.5], colors='red', 
                      linewidths=2, origin='lower')
        
        ax.set_title(f'Sagittal Slice {slice_num}/128', fontsize=14, fontweight='bold')
        ax.set_xlabel('Y (Posterior → Anterior)', fontsize=11)
        ax.set_ylabel('Z (Inferior → Superior)', fontsize=11)
        ax.grid(False)
        
        # Orientation markers
        ax.text(0.02, 0.98, 'P', transform=ax.transAxes, fontsize=12, 
                color='cyan', fontweight='bold', va='top')
        ax.text(0.98, 0.98, 'A', transform=ax.transAxes, fontsize=12, 
                color='cyan', fontweight='bold', va='top', ha='right')
    
    # Add 3D orientation diagram for sagittal
    ax_orient = fig.add_subplot(gs[2, 3])
    ax_orient.text(0.5, 0.7, '⊕', fontsize=60, ha='center', color='navy')
    ax_orient.text(0.5, 0.3, 'SAGITTAL VIEW\n(Looking Sideways)', fontsize=12, 
                   ha='center', fontweight='bold')
    ax_orient.text(0.5, 0.1, 'Left → Right', fontsize=10, 
                   ha='center', style='italic', color='gray')
    ax_orient.axis('off')
    
    # ========================================================================
    # ADD LEGEND AND INFORMATION
    # ========================================================================
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='darkgray', label='Brain Tissue (MRI T1)'),
        mpatches.Patch(facecolor='red', alpha=0.6, label='Hippocampus (Segmented)'),
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, 
                      label='Hippocampus Boundary')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              fontsize=12, frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, 0.01))
    
    # Add information box
    info_text = (
        "Patient ID: hippocampus_008\n"
        "Modality: T1-weighted MRI\n"
        "Dimensions: 128×128×128 voxels\n"
        "Segmentation: Automated hippocampus detection"
    )
    
    fig.text(0.98, 0.02, info_text, fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_filename}")
    
    plt.show()

# ============================================================================
# EXECUTE VISUALIZATION
# ============================================================================

create_professional_slices(img_enhanced, hippo_smooth, 
                          output_filename="hippocampus_mpr_visualization.png")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("\nFile saved: hippocampus_mpr_visualization.png")
print("Resolution: High-quality 300 DPI for publication/presentation")
print("\nOrthogonal views explained:")
print("  • AXIAL: Horizontal slices (like looking down at the head)")
print("  • CORONAL: Vertical slices from front to back")
print("  • SAGITTAL: Vertical slices from left to right")
print("\nColor coding:")
print("  • Grayscale: Original MRI brain tissue")
print("  • Red overlay: Segmented hippocampus")
print("  • Red contour: Precise hippocampus boundaries")
print("="*70)