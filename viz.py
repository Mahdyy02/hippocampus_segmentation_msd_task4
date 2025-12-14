import nibabel as nib 
import numpy as np 
from skimage import measure, transform, filters
import pyvista as pv 
 
# Charger IRM et masque 
img = nib.load("Task04_Hippocampus/imagesTr/hippocampus_008.nii.gz").get_fdata() 
mask = nib.load("Task04_Hippocampus/labelsTr/hippocampus_008.nii.gz").get_fdata() 
 
# Rescale pour rendre voxels isotropes (ex : 128x128x128) 
img_resized = transform.resize(img, (128,128,128), order=1, preserve_range=True) 
mask_resized = transform.resize(mask, (128,128,128), order=0, preserve_range=True) 
 
# Normaliser IRM 
img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min()) 

# Améliorer le contraste avec un seuillage adaptatif
img_enhanced = np.clip(img_norm * 1.5, 0, 1)  # Augmenter le contraste

# Hippocampe binaire avec lissage
hippo = mask_resized > 0

# Appliquer un léger lissage Gaussien pour réduire les artifacts
from scipy.ndimage import gaussian_filter
hippo_smooth = gaussian_filter(hippo.astype(float), sigma=0.5) > 0.5
 
# Marching cubes pour surface lisse de l'hippocampe
verts, faces, _, _ = measure.marching_cubes(hippo_smooth, level=0.5)
faces = np.hstack([np.full((faces.shape[0],1),3), faces]).flatten()

# Créer aussi un mesh pour le cerveau (contour externe)
brain_mask = img_norm > 0.15  # Seuil pour extraire le cerveau
brain_mask_smooth = gaussian_filter(brain_mask.astype(float), sigma=1.0)
brain_verts, brain_faces, _, _ = measure.marching_cubes(brain_mask_smooth, level=0.5)
brain_faces = np.hstack([np.full((brain_faces.shape[0],1),3), brain_faces]).flatten()
 
# PyVista plot avec meilleure configuration
plotter = pv.Plotter(window_size=[1200, 900])
plotter.set_background("white")

# Option 1: Volume rendering amélioré (décommenter si préféré)
# opacity = [0, 0.0, 0.03, 0.08, 0.15, 0.25]
# plotter.add_volume(img_enhanced, opacity=opacity, cmap="gray", shade=True)

# Option 2: Surface du cerveau semi-transparente (méthode recommandée)
brain_mesh = pv.PolyData(brain_verts, brain_faces)
brain_mesh = brain_mesh.smooth(n_iter=50)  # Lissage pour apparence plus naturelle
plotter.add_mesh(
    brain_mesh, 
    color="gray", 
    opacity=0.7,  # Très transparent pour voir à travers
    smooth_shading=True,
    show_edges=False
)

# Hippocampe avec couleur vive et contour
hippo_mesh = pv.PolyData(verts, faces)
hippo_mesh = hippo_mesh.smooth(n_iter=30)  # Lisser la surface

# Ajouter l'hippocampe principal
plotter.add_mesh(
    hippo_mesh, 
    color="crimson",  # Rouge vif
    opacity=0.95, 
    smooth_shading=True,
    ambient=0.3,
    diffuse=0.7,
    specular=0.3,
    specular_power=15
)

# Ajouter un contour pour mieux distinguer l'hippocampe
plotter.add_mesh(
    hippo_mesh.extract_feature_edges(), 
    color="darkred", 
    line_width=2,
    opacity=1.0
)

# Améliorer l'éclairage
plotter.add_light(pv.Light(position=(100, 100, 100), light_type='scene light'))
plotter.add_light(pv.Light(position=(-100, -100, 100), light_type='scene light', intensity=0.5))

# Meilleure position de caméra pour voir l'hippocampe
plotter.camera_position = [
    (200, 150, 150),  # Position de la caméra
    (64, 64, 64),     # Point focal (centre)
    (0, 0, 1)         # Vecteur up
]

# Ajouter des axes pour l'orientation
plotter.add_axes(
    xlabel='X', 
    ylabel='Y', 
    zlabel='Z',
    line_width=3,
    labels_off=False
)

# Activer la rotation interactive
plotter.enable_trackball_style()

# Afficher
plotter.show()

# Alternative: Sauvegarder une image
# plotter.screenshot("brain_hippocampus.png", window_size=[1920, 1080])