import os
import glob
import numpy as np
import nibabel as nib

from skimage.transform import resize, AffineTransform, warp
from scipy.ndimage import rotate

# =============================================================================
# CONFIG
# =============================================================================

DATASET_ROOT = "Task04_Hippocampus"
IMG_DIR = os.path.join(DATASET_ROOT, "imagesTr")
LBL_DIR = os.path.join(DATASET_ROOT, "labelsTr")

OUT_IMG_DIR = "dataset_2d/images"
OUT_MASK_DIR = "dataset_2d/masks"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

TARGET_SHAPE = (128, 128, 128)
SLICE_SIZE = 128

PLANES = ["axial", "coronal", "sagittal"]
MAX_SLICES_PER_PLANE = 15   # 🔥 contrôle principal (10–20 recommandé)

# Probabilités d’augmentation
P_NONE   = 0.40
P_ROT    = 0.25
P_AFFINE = 0.25
P_FLIP   = 0.10

np.random.seed(42)

# =============================================================================
# UTILS
# =============================================================================

def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def load_and_resample(img_path, lbl_path):
    img = nib.load(img_path).get_fdata()
    lbl = nib.load(lbl_path).get_fdata()

    img = resize(
        img, TARGET_SHAPE,
        order=1, preserve_range=True, anti_aliasing=True
    )

    lbl = resize(
        lbl, TARGET_SHAPE,
        order=0, preserve_range=True, anti_aliasing=False
    )

    return normalize(img), lbl.astype(np.int64)


def extract_slices(img, lbl, plane):
    slices = []

    if plane == "axial":
        for i in range(img.shape[2]):
            slices.append((img[:, :, i], lbl[:, :, i]))

    elif plane == "coronal":
        for i in range(img.shape[1]):
            slices.append((img[:, i, :], lbl[:, i, :]))

    elif plane == "sagittal":
        for i in range(img.shape[0]):
            slices.append((img[i, :, :], lbl[i, :, :]))

    return slices


def sample_slices(slices, k):
    valid = [(i, l) for i, l in slices if np.max(l) > 0]

    if len(valid) <= k:
        return valid

    idx = np.random.choice(len(valid), k, replace=False)
    return [valid[i] for i in idx]


def random_augment(img, lbl):
    r = np.random.rand()

    if r < P_NONE:
        return img, lbl

    elif r < P_NONE + P_ROT:
        angle = np.random.uniform(-7, 7)
        img = rotate(img, angle, reshape=False, order=1, mode="nearest")
        lbl = rotate(lbl, angle, reshape=False, order=0, mode="nearest")

    elif r < P_NONE + P_ROT + P_AFFINE:
        scale = np.random.uniform(0.95, 1.05)
        tx = np.random.uniform(-5, 5)
        ty = np.random.uniform(-5, 5)

        tform = AffineTransform(
            scale=(scale, scale),
            translation=(tx, ty)
        )

        img = warp(img, tform.inverse, order=1,
                   preserve_range=True, mode="constant")
        lbl = warp(lbl, tform.inverse, order=0,
                   preserve_range=True, mode="constant")

    else:
        img = np.fliplr(img)
        lbl = np.fliplr(lbl)

    return img.astype(np.float32), lbl.astype(np.int64)


# =============================================================================
# MAIN
# =============================================================================

counter = 0
img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.nii.gz")))

for img_path in img_files:
    case_id = os.path.basename(img_path)
    lbl_path = os.path.join(LBL_DIR, case_id)

    print(f"[INFO] Processing {case_id}")

    img, lbl = load_and_resample(img_path, lbl_path)

    for plane in PLANES:
        slices = extract_slices(img, lbl, plane)
        slices = sample_slices(slices, MAX_SLICES_PER_PLANE)

        for img2d, lbl2d in slices:

            img2d, lbl2d = random_augment(img2d, lbl2d)

            img2d = img2d[:SLICE_SIZE, :SLICE_SIZE]
            lbl2d = lbl2d[:SLICE_SIZE, :SLICE_SIZE]

            img_name = f"img_{counter:06d}.npy"
            lbl_name = f"mask_{counter:06d}.npy"

            np.save(os.path.join(OUT_IMG_DIR, img_name), img2d)
            np.save(os.path.join(OUT_MASK_DIR, lbl_name), lbl2d)

            counter += 1

    print(f"    saved so far: {counter}")

print("\n===================================")
print("DATASET GENERATION COMPLETE")
print(f"Total 2D samples: {counter}")
print("Image shape:", (128, 128))
print("Mask values:", "{0,1,2}")
print("===================================")
