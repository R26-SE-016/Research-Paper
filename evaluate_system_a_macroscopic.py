#!/usr/bin/env python3
"""
System A: Macroscopic Aerial Surveillance Benchmark Script
Project: Multiscale Computer Vision Ecosystem for Coconut Pathology Detection

Evaluates:
- Canonical Excess-Green (ExG) chromatic canopy segmentation & Otsu binarization.
- Deterministic Euclidean Distance Transform (EDT) topological crown extraction (938 crowns).
- Moving-Window Local Spatial Z-Score Anomaly Engine for early canopy stress localization (21 hotspots).
- Dataset: High-resolution (5280x3956) DJI Multispectral orthomosaics from Ambakele Research Station, Sri Lanka.
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

def evaluate_system_a(ambakele_dir):
    print("=" * 70)
    print("EVALUATING SYSTEM A: MACROSCOPIC PIPELINE ON AMBAKELE UAV TILES")
    print("=" * 70)
    
    if not os.path.exists(ambakele_dir):
        print(f"Error: Directory {ambakele_dir} not found.")
        return None
        
    rgb_files = [os.path.join(ambakele_dir, f) for f in os.listdir(ambakele_dir) if f.endswith("_D.JPG")]
    nir_files = [os.path.join(ambakele_dir, f) for f in os.listdir(ambakele_dir) if f.endswith("_MS_NIR.TIF")]
    r_files = [os.path.join(ambakele_dir, f) for f in os.listdir(ambakele_dir) if f.endswith("_MS_R.TIF")]
    
    print(f"Found {len(rgb_files)} RGB drone orthophotos and {len(nir_files)} Multispectral NIR/R bands in Ambakele.")
    
    if not rgb_files:
        print("Error: No RGB drone tiles found.")
        return None
        
    sample_rgb_path = rgb_files[0]
    print(f"Processing sample tile: {os.path.basename(sample_rgb_path)}")
    
    img = Image.open(sample_rgb_path).convert('RGB')
    w_orig, h_orig = img.size
    print(f"Tile dimensions (full sensor resolution): {w_orig} x {h_orig} pixels")
    
    # Process at standard 1024x1024 scaled tile for edge/serverless execution
    scale_factor = 1024.0 / max(w_orig, h_orig)
    w, h = int(w_orig * scale_factor), int(h_orig * scale_factor)
    img_scaled = img.resize((w, h), Image.Resampling.BILINEAR)
    img_np = np.array(img_scaled, dtype=np.float32)
    
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]
    
    # 1. Chromatic coordinates & ExG
    sigma_rgb = np.maximum(R + G + B, 1e-6)
    r_norm = R / sigma_rgb
    g_norm = G / sigma_rgb
    b_norm = B / sigma_rgb
    exg = 2.0 * g_norm - r_norm - b_norm
    
    # 2. Otsu threshold on ExG
    t0 = time.perf_counter()
    exg_min, exg_max = np.min(exg), np.max(exg)
    exg_scaled = ((exg - exg_min) / (exg_max - exg_min + 1e-6) * 255).astype(np.uint8)
    
    hist, bin_edges = np.histogram(exg_scaled.ravel(), bins=256, range=(0, 256))
    total = exg_scaled.size
    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b = 0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * ((m_b - m_f) ** 2)
        if between_var > current_max:
            current_max = between_var
            threshold = t
            
    otsu_val = bin_edges[threshold] / 255.0 * (exg_max - exg_min) + exg_min
    canopy_mask = (exg >= otsu_val) & (g_norm > r_norm)
    t_mask = time.perf_counter() - t0
    
    canopy_pct = (np.sum(canopy_mask) / canopy_mask.size) * 100.0
    print(f"Canopy Coverage Mask: {canopy_pct:.2f}% canopy, {100.0 - canopy_pct:.2f}% ground soil (computed in {t_mask*1000.0:.1f} ms)")
    
    # 3. Euclidean Distance Transform + Gaussian Smooth + Peak Filter
    t0 = time.perf_counter()
    edt = ndimage.distance_transform_edt(canopy_mask)
    edt_smooth = ndimage.gaussian_filter(edt, sigma=2.2)
    
    d_min = int(26 * scale_factor)
    footprint_size = max(5, 2 * d_min + 1)
    footprint = np.ones((footprint_size, footprint_size), dtype=bool)
    local_max = ndimage.maximum_filter(edt_smooth, footprint=footprint) == edt_smooth
    
    min_depth = max(2.0, 6.0 * scale_factor)
    peaks = local_max & (edt_smooth >= min_depth) & canopy_mask
    peak_coords = np.argwhere(peaks)
    t_edt = time.perf_counter() - t0
    
    tree_count = len(peak_coords)
    print(f"Extracted {tree_count} discrete palm tree crowns via EDT Peak Filtering in {t_edt*1000.0:.1f} ms.")
    
    # 4. Multispectral NDVI & Moving-Window Z-Score
    outliers = 0
    if nir_files and r_files:
        nir_pil = Image.open(nir_files[0])
        red_pil = Image.open(r_files[0])
        
        nir_resized = np.array(nir_pil.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
        red_resized = np.array(red_pil.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
        ndvi = (nir_resized - red_resized) / np.maximum(nir_resized + red_resized, 1e-6)
        
        tree_ndvis = []
        tree_locs = []
        for (py, px) in peak_coords:
            rad = int(max(3, edt_smooth[py, px]))
            y0, y1 = max(0, py - rad), min(h, py + rad)
            x0, x1 = max(0, px - rad), min(w, px + rad)
            tree_ndvis.append(float(np.mean(ndvi[y0:y1, x0:x1])))
            tree_locs.append((px, py))
            
        tree_ndvis = np.array(tree_ndvis)
        R_spatial = int(250 * scale_factor)  # ~35m spatial radius
        z_scores = []
        for i in range(len(tree_ndvis)):
            px_i, py_i = tree_locs[i]
            dists = np.sqrt([(px_i - px_j)**2 + (py_i - py_j)**2 for (px_j, py_j) in tree_locs])
            neighbors = tree_ndvis[dists <= R_spatial]
            if len(neighbors) > 2:
                mu_i = np.mean(neighbors)
                sigma_i = np.std(neighbors)
                z = (tree_ndvis[i] - mu_i) / (sigma_i + 1e-6)
            else:
                z = 0.0
            z_scores.append(z)
            if z < -2.0:
                outliers += 1
                
        print(f"Spatial Z-Score Anomaly Engine: identified {outliers} early stress hotspots (Z < -2.0) across {tree_count} palms.")
        
    result = {
        "tile": os.path.basename(sample_rgb_path),
        "dimensions": f"{w}x{h}",
        "full_sensor_resolution": f"{w_orig}x{h_orig}",
        "canopy_pct": round(canopy_pct, 2),
        "tree_count": int(tree_count),
        "edt_time_ms": round(t_edt * 1000.0, 2),
        "mask_time_ms": round(t_mask * 1000.0, 2),
        "total_macroscopic_time_ms": round((t_mask + t_edt) * 1000.0, 2),
        "outliers_z2": int(outliers)
    }
    
    out_path = r"f:\GitHub\Research\Research-Paper\system_a_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"\nSaved System A benchmark results to: {out_path}")
    return result

if __name__ == "__main__":
    ambakele_dir = r"f:\GitHub\Research\Datasets\Raw Images\Ambakele"
    evaluate_system_a(ambakele_dir)
