#!/usr/bin/env python3
"""
Comprehensive Benchmark & Empirical Verification Script
Project: Multiscale Computer Vision Ecosystem for Coconut Pathology Detection

Evaluates:
1. System B: Real INT8 TFLite MobileNetV2 on the official multiclass test dataset (240 images).
2. System A: Macroscopic Aerial Pipeline on Ambakele DJI Multispectral Tiles.
"""

import os
import sys
import time
import json
import csv
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from ai_edge_litert.interpreter import Interpreter

def evaluate_system_b(test_dir, model_path):
    print("=" * 70)
    print("EVALUATING SYSTEM B: INT8 TFLITE MODEL ON TEST DATASET")
    print("=" * 70)
    
    classes_csv_path = os.path.join(test_dir, "_classes.csv")
    if not os.path.exists(classes_csv_path):
        print(f"Error: {classes_csv_path} not found.")
        return None

    # Load TFLite Model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_shape = input_details['shape']  # [1, H, W, C]
    input_dtype = input_details['dtype']
    output_dtype = output_details['dtype']
    input_scale, input_zero_point = input_details.get('quantization', (0.0, 0))
    output_scale, output_zero_point = output_details.get('quantization', (0.0, 0))
    
    print(f"Model Input : shape={input_shape}, dtype={input_dtype}, scale={input_scale}, zero_point={input_zero_point}")
    print(f"Model Output: shape={output_details['shape']}, dtype={output_dtype}, scale={output_scale}, zero_point={output_zero_point}")
    
    # Read classes CSV
    rows = []
    with open(classes_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Class columns in _classes.csv:
        # ['bud root dropping', 'bud rot', 'gray leaf spot', 'healthy leaves', 'leaf rot', 'stembleeding']
        class_names = [c.strip() for c in header[1:]]
        for row in reader:
            if len(row) >= 7:
                rows.append(row)
                
    num_classes = len(class_names)
    print(f"Found {len(rows)} test samples across {num_classes} classes: {class_names}")
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    latencies = []
    in_dist_entropies = []
    in_dist_max_probs = []
    
    T = 1.5  # Temperature scaling
    
    def get_class_from_filename(f):
        fl = f.lower()
        if fl.startswith('budrootdropping'): return 0
        elif fl.startswith('budrot'): return 1
        elif fl.startswith('grayleafspot'): return 2
        elif fl.startswith('healthy_leaf'): return 3
        elif fl.startswith('leafrot'): return 4
        elif fl.startswith('stembleeding'): return 5
        return -1

    files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    print(f"Evaluating {len(files)} test images across 6 consolidated classes: {class_names}")

    for filename in files:
        true_class = get_class_from_filename(filename)
        if true_class == -1:
            continue
            
        img_path = os.path.join(test_dir, filename)
        if not os.path.exists(img_path):
            continue
            
        # Load image in RGB
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((input_shape[1], input_shape[2]), Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.uint8)
        input_data = np.expand_dims(img_array, axis=0)
            
        # Run inference
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        
        raw_output = interpreter.get_tensor(output_details['index'])[0]
        
        # Dequantize output
        if output_dtype == np.uint8:
            probs = raw_output.astype(np.float32) / 255.0
            probs = probs / np.sum(probs)
        elif output_dtype == np.int8:
            probs = (raw_output.astype(np.float32) + 128.0) / 255.0
            probs = probs / np.sum(probs)
        else:
            # Apply Temperature-scaled Softmax if raw logits
            scaled_logits = raw_output.astype(np.float32) / T
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_logits / np.sum(exp_logits)
            
        # Compute Shannon Entropy
        safe_probs = np.clip(probs, 1e-9, 1.0)
        entropy = -np.sum(safe_probs * np.log2(safe_probs))
        in_dist_entropies.append(entropy)
        in_dist_max_probs.append(np.max(probs))
        
        pred_class = int(np.argmax(probs))
        confusion_matrix[true_class, pred_class] += 1

    # Per-Class Metrics
    print("\n" + "-" * 70)
    print("GENUINE LOGGED PER-CLASS EVALUATION METRICS (TEST SET N=240)")
    print("-" * 70)
    
    results = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for i, cname in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        support = np.sum(confusion_matrix[i, :])
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        
        results[cname] = {
            "precision": round(prec * 100.0, 2),
            "recall": round(rec * 100.0, 2),
            "f1": round(f1 * 100.0, 2),
            "support": int(support),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)
        }
        print(f"{cname:<20}: Precision={prec*100.0:6.2f}%, Recall={rec*100.0:6.2f}%, F1={f1*100.0:6.2f}%, Support={support:4d}")
        
    total_tp = np.trace(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    overall_accuracy = (total_tp / total_samples) * 100.0 if total_samples > 0 else 0.0
    macro_prec = np.mean(precisions) * 100.0
    macro_rec = np.mean(recalls) * 100.0
    macro_f1 = np.mean(f1s) * 100.0
    
    # Cohen's Kappa
    p_o = overall_accuracy / 100.0
    p_e = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / (total_samples ** 2)
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) != 0 else 0.0
    
    avg_latency = np.mean(latencies[10:]) if len(latencies) > 10 else np.mean(latencies)
    
    print("-" * 70)
    print(f"Overall Accuracy : {overall_accuracy:.2f}%")
    print(f"Macro Precision  : {macro_prec:.2f}%")
    print(f"Macro Recall     : {macro_rec:.2f}%")
    print(f"Macro F1-Score   : {macro_f1:.2f}%")
    print(f"Cohen's Kappa (k): {kappa:.4f}")
    print(f"Average Latency  : {avg_latency:.2f} ms/frame on CPU")
    print(f"Mean In-Dist H   : {np.mean(in_dist_entropies):.2f} bits")
    print("-" * 70)
    
    # Benchmark OOD Filter on 500 Out-of-Distribution Noise Samples
    print("\nBENCHMARKING OUT-OF-DISTRIBUTION (OOD) GATING ON NOISE CHALLENGE SET (N=500)")
    np.random.seed(42)
    ood_rejections = 0
    H_thresh = 2.1
    tau = 0.40
    
    for _ in range(500):
        noise_type = np.random.choice(['soil', 'flare', 'skin', 'dark', 'random'])
        H, W = input_shape[1], input_shape[2]
        if noise_type == 'soil':
            base = np.array([120, 80, 50], dtype=np.int16)
            noise = base + np.random.randint(-20, 20, (H, W, 3))
        elif noise_type == 'flare':
            base = np.array([240, 240, 180], dtype=np.int16)
            noise = base + np.random.randint(-15, 15, (H, W, 3))
        elif noise_type == 'skin':
            base = np.array([210, 160, 130], dtype=np.int16)
            noise = base + np.random.randint(-15, 15, (H, W, 3))
        elif noise_type == 'dark':
            noise = np.random.randint(0, 40, (H, W, 3))
        else:
            noise = np.random.randint(0, 256, (H, W, 3))
            
        noise_uint8 = np.clip(noise, 0, 255).astype(np.uint8)
        
        if input_dtype == np.uint8:
            noise_input = np.expand_dims(noise_uint8, axis=0)
        elif input_dtype == np.int8:
            noise_input = np.expand_dims(noise_uint8.astype(np.int8) - 128, axis=0)
        else:
            noise_input = np.expand_dims(noise_uint8.astype(np.float32), axis=0)
            
        interpreter.set_tensor(input_details['index'], noise_input)
        interpreter.invoke()
        
        raw_out = interpreter.get_tensor(output_details['index'])[0]
        # Quantized probabilities p in [0, 1]
        p_raw = np.maximum(raw_out.astype(np.float32) / 255.0, 1e-6)
        p_raw = p_raw / np.sum(p_raw)
        
        # Temperature Calibration: z = ln(p), p_calibrated = softmax(z / T)
        z = np.log(p_raw)
        z_scaled = z / T
        exp_z = np.exp(z_scaled - np.max(z_scaled))
        p = exp_z / np.sum(exp_z)
            
        safe_p = np.clip(p, 1e-9, 1.0)
        h = -np.sum(safe_p * np.log2(safe_p))
        max_p = np.max(p)
        
        if h > H_thresh or max_p < tau:
            ood_rejections += 1
            
    ood_rejection_rate = (ood_rejections / 500.0) * 100.0
    print(f"OOD Challenge Rejection Rate: {ood_rejections}/500 ({ood_rejection_rate:.2f}%) successfully filtered as 'Ambiguous Foliage'")
    
    return {
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class": results,
        "overall_accuracy": round(overall_accuracy, 2),
        "macro_precision": round(macro_prec, 2),
        "macro_recall": round(macro_rec, 2),
        "macro_f1": round(macro_f1, 2),
        "cohens_kappa": round(kappa, 4),
        "latency_ms": round(avg_latency, 2),
        "ood_rejection_rate": round(ood_rejection_rate, 2)
    }


def evaluate_system_a(ambakele_dir):
    print("\n" + "=" * 70)
    print("EVALUATING SYSTEM A: MACROSCOPIC PIPELINE ON AMBAKELE UAV TILES")
    print("=" * 70)
    
    rgb_files = [os.path.join(ambakele_dir, f) for f in os.listdir(ambakele_dir) if f.endswith("_D.JPG")]
    nir_files = [os.path.join(ambakele_dir, f) for f in os.listdir(ambakele_dir) if f.endswith("_MS_NIR.TIF")]
    r_files = [os.path.join(ambakele_dir, f) for f in os.listdir(ambakele_dir) if f.endswith("_MS_R.TIF")]
    
    print(f"Found {len(rgb_files)} RGB drone orthophotos and {len(nir_files)} Multispectral NIR/R bands in Ambakele.")
    
    if not rgb_files:
        return None
        
    sample_rgb_path = rgb_files[0]
    print(f"Processing sample tile: {os.path.basename(sample_rgb_path)}")
    
    img = Image.open(sample_rgb_path).convert('RGB')
    w_orig, h_orig = img.size
    print(f"Tile dimensions (full): {w_orig} x {h_orig} pixels")
    
    # Process at standard 1024x1024 crop/tile
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
        R_spatial = int(250 * scale_factor)
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
                
        print(f"Spatial Z-Score Anomaly Engine: identified {outliers} pre-symptomatic stress hotspots (Z < -2.0) across {tree_count} palms.")
        
    return {
        "tile": os.path.basename(sample_rgb_path),
        "dimensions": f"{w}x{h}",
        "canopy_pct": round(canopy_pct, 2),
        "tree_count": int(tree_count),
        "edt_time_ms": round(t_edt * 1000.0, 2),
        "mask_time_ms": round(t_mask * 1000.0, 2),
        "outliers_z2": int(outliers)
    }

if __name__ == "__main__":
    test_dir = r"f:\GitHub\Research\Datasets\Microscopic Mobile Diagnosis of Coconut related Diseases.v2-microscopic_baseline_v2_pp1.multiclass\test"
    model_path = r"f:\GitHub\Research\Coconut-Pathology-Detection-Service\system_b_baseline_int8.tflite"
    ambakele_dir = r"f:\GitHub\Research\Datasets\Raw Images\Ambakele"
    
    sys_b_res = evaluate_system_b(test_dir, model_path)
    sys_a_res = evaluate_system_a(ambakele_dir)
    
    output_summary = {
        "system_b": sys_b_res,
        "system_a": sys_a_res
    }
    
    with open(r"f:\GitHub\Research\Research-Paper\genuine_benchmark_results.json", "w") as f:
        json.dump(output_summary, f, indent=2)
        
    print("\nSaved genuine logged results to f:\\GitHub\\Research\\Research-Paper\\genuine_benchmark_results.json")
