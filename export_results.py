# -*- coding: utf-8 -*-
"""
export_results.py – Run AFTER training to generate fabrication files only
Generalized for any number of diffractive layers
"""

import torch
import torch.serialization
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from datetime import datetime

try:
    from stl import mesh
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    print("Warning: numpy-stl not installed. Skipping real .stl files.")

# ── Import your model classes ───────────────────────────────────────────
from model import Onn
from OpticalLayers import Diffraction, DiffLayer

# ── Allow-list all types used in the pickled model (fixes weights_only error) ──
torch.serialization.add_safe_globals([
    Onn,
    Diffraction,
    DiffLayer,
    torch.nn.ModuleList,
    torch.nn.modules.module.Module,
])

# ====================== CONFIG ======================
n_material = 1.634
base_thickness_mm = 2.0
lambda0_mm = 0.75
M = 256
L = 80.0
# ===================================================

# Find most recent run
run_folders = sorted(glob.glob("runs/run_*"), reverse=True)
if not run_folders:
    raise FileNotFoundError("No run_* folder found. Run onn_train.py first.")
latest_run = run_folders[0]
print(f"Using most recent run: {latest_run}")

model_files = sorted(
    glob.glob(f"{latest_run}/models/onn*.pt"),
    key=lambda x: int(''.join(filter(str.isdigit, x))) or 0,
    reverse=True
)
if not model_files:
    raise FileNotFoundError(f"No onn*.pt found in {latest_run}/models/")
model_path = model_files[0]
print(f"Loading: {model_path}")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

onn = torch.load(model_path, map_location=device)
onn.eval()

save_dir = f"fabrication_{datetime.now().strftime('%Y%m%d_%H%M')}"
os.makedirs(f'{save_dir}/phases', exist_ok=True)
os.makedirs(f'{save_dir}/lumerical', exist_ok=True)
os.makedirs(f'{save_dir}/stl', exist_ok=True)

print(f"Detected {onn.get_num_phase_layers()} phase layers")

phases = onn.get_phase_masks()
for idx, phase_np in enumerate(phases, 1):
    phase_mod = phase_np % (2 * np.pi)
    np.save(f'{save_dir}/phases/phase_layer{idx}.npy', phase_mod)
    
    fig = plt.figure(figsize=(6,6), dpi=300)
    im = plt.imshow(phase_mod, cmap='twilight_shifted')
    plt.colorbar(im, label='Phase (radians)')
    plt.title(f'Phase Mask - Layer {idx}')
    plt.axis('off')
    plt.savefig(f'{save_dir}/phases/phase_layer{idx}_color.png', bbox_inches='tight')
    plt.close(fig)

    height_rel_mm = phase_mod * lambda0_mm / (2 * np.pi * (n_material - 1))
    total_height_mm = base_thickness_mm + height_rel_mm

    np.save(f'{save_dir}/lumerical/height_layer{idx}_lumerical.npy', total_height_mm / 1000)

    if STL_AVAILABLE:
        x = np.linspace(-L/2, L/2, M)
        y = np.linspace(-L/2, L/2, M)
        X, Y = np.meshgrid(x, y)
        Z = total_height_mm

        vertices = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        faces = []
        for i in range(M-1):
            for j in range(M-1):
                v0 = i*M + j
                v1 = v0 + 1
                v2 = v0 + M
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        faces = np.array(faces)

        your_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                your_mesh.vectors[i, j] = vertices[f[j], :]
        your_mesh.save(f'{save_dir}/stl/layer_{idx}.stl')
        print(f"Saved STL: {save_dir}/stl/layer_{idx}.stl")
    else:
        plt.imsave(f'{save_dir}/stl/height_layer{idx}_preview.png', total_height_mm, cmap='viridis')
        print(f"Saved preview PNG (STL skipped): {save_dir}/stl/height_layer{idx}_preview.png")

print(f"\nExported to: {save_dir}/")
print("   • phases/phase_layer*_color.png")
print("   • lumerical/height_*_lumerical.npy")
if STL_AVAILABLE:
    print("   • stl/layer_*.stl")
else:
    print("   • stl/height_*_preview.png  (STL skipped — install numpy-stl)")