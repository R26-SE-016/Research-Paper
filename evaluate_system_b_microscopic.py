#!/usr/bin/env python3
"""
System B: Microscopic Mobile Edge Diagnostics Benchmark Script
Project: Multiscale Computer Vision Ecosystem for Coconut Pathology Detection

Evaluates:
- Full-Integer INT8 Quantized MobileNetV2 on the leak-free, un-augmented V3 test set (N=233).
- Generates Confusion Matrix, Per-Class Precision, Recall, F1-Score, and Support.
- Calculates Top-1 Accuracy and Cohen's Kappa.
- Computes 1,000-Iteration Bootstrap 95% Confidence Intervals for all primary metrics.
- Evaluates CPU Inference Latency (LiteRT with XNNPACK).
- Evaluates Temperature-Scaled Shannon Entropy OOD Filter on 500 noise samples.
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

def evaluate_system_b(test_dir, model_path):
    print("=" * 70)
    print("EVALUATING SYSTEM B: INT8 TFLITE MODEL ON UN-AUGMENTED V3 TEST DATASET")
    print("=" * 70)
    
    if not os.path.exists(test_dir):
        print(f"Error: Directory {test_dir} not found.")
        return None
        
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None

    # Load TFLite Model
    print(f"Loading Model: {model_path}", flush=True)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_shape = input_details['shape']  # [1, H, W, C]
    input_dtype = input_details['dtype']
    output_dtype = output_details['dtype']
    
    print(f"Model Input : shape={input_shape}, dtype={input_dtype}")
    print(f"Model Output: shape={output_details['shape']}, dtype={output_dtype}")
    
    classes = ['BudRootDropping', 'BudRot', 'GrayLeafSpot', 'Healthy', 'LeafRot', 'StemBleeding']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_true = []
    y_pred = []
    latencies = []
    in_dist_entropies = []
    in_dist_max_probs = []
    
    T = 1.5  # Temperature scaling factor for calibration
    
    print(f"\nProcessing test samples from: {test_dir}")
    for cname in classes:
        cdir = os.path.join(test_dir, cname)
        if not os.path.exists(cdir):
            print(f"Warning: {cdir} not found.")
            continue
        fnames = [f for f in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, f))]
        print(f"Class {cname:<18}: {len(fnames)} test images")
        
        for fname in fnames:
            fpath = os.path.join(cdir, fname)
            img = Image.open(fpath).convert('RGB').resize((input_shape[1], input_shape[2]))
            arr = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
            
            t0 = time.perf_counter()
            interpreter.set_tensor(input_details['index'], arr)
            interpreter.invoke()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
            raw_out = interpreter.get_tensor(output_details['index'])[0]
            
            # Dequantize output probabilities
            p_raw = np.maximum(raw_out.astype(np.float32) / 255.0, 1e-6)
            p_raw = p_raw / np.sum(p_raw)
            
            # Calibrated probabilities via Temperature Softmax
            z = np.log(p_raw)
            z_scaled = z / T
            exp_z = np.exp(z_scaled - np.max(z_scaled))
            probs = exp_z / np.sum(exp_z)
            
            safe_probs = np.clip(probs, 1e-9, 1.0)
            entropy = -np.sum(safe_probs * np.log2(safe_probs))
            in_dist_entropies.append(entropy)
            in_dist_max_probs.append(np.max(probs))
            
            pred_idx = int(np.argmax(raw_out))
            y_true.append(class_to_idx[cname])
            y_pred.append(pred_idx)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_n = len(y_true)
    
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    print("\n" + "-" * 70)
    print(f"GENUINE LOGGED PER-CLASS EVALUATION METRICS (UN-AUGMENTED TEST SET N={total_n})")
    print("-" * 70)
    
    results = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for i, cname in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
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
        print(f"{cname:<18}: Precision={prec*100.0:6.2f}%, Recall={rec*100.0:6.2f}%, F1={f1*100.0:6.2f}%, Support={support:3d}")
        
    acc = np.mean(y_true == y_pred) * 100.0
    macro_prec = np.mean(precisions) * 100.0
    macro_rec = np.mean(recalls) * 100.0
    macro_f1 = np.mean(f1s) * 100.0
    
    # Cohen's Kappa
    p_o = acc / 100.0
    p_e = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total_n ** 2)
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) != 0 else 0.0
    
    # 1,000-Iteration Bootstrap 95% Confidence Intervals
    np.random.seed(42)
    b_accs, b_f1s, b_kappas = [], [], []
    for _ in range(1000):
        idx = np.random.choice(total_n, size=total_n, replace=True)
        b_yt, b_yp = y_true[idx], y_pred[idx]
        b_accs.append(np.mean(b_yt == b_yp) * 100.0)
        
        b_cm = np.zeros((6, 6), dtype=int)
        for bt, bp in zip(b_yt, b_yp):
            b_cm[bt, bp] += 1
            
        b_f1_list = []
        for bi in range(6):
            btp = b_cm[bi, bi]
            bfp = np.sum(b_cm[:, bi]) - btp
            bfn = np.sum(b_cm[bi, :]) - btp
            bp = btp / (btp + bfp) if (btp + bfp) > 0 else 0.0
            br = btp / (btp + bfn) if (btp + bfn) > 0 else 0.0
            bf = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0.0
            b_f1_list.append(bf)
        b_f1s.append(np.mean(b_f1_list) * 100.0)
        
        bp_o = np.mean(b_yt == b_yp)
        bp_e = np.sum(np.sum(b_cm, axis=0) * np.sum(b_cm, axis=1)) / (total_n ** 2)
        b_k = (bp_o - bp_e) / (1.0 - bp_e) if (1.0 - bp_e) != 0 else 0.0
        b_kappas.append(b_k)
        
    ci_acc = [round(float(np.percentile(b_accs, 2.5)), 2), round(float(np.percentile(b_accs, 97.5)), 2)]
    ci_f1 = [round(float(np.percentile(b_f1s, 2.5)), 2), round(float(np.percentile(b_f1s, 97.5)), 2)]
    ci_kappa = [round(float(np.percentile(b_kappas, 2.5)), 4), round(float(np.percentile(b_kappas, 97.5)), 4)]
    
    avg_latency = np.mean(latencies[10:]) if len(latencies) > 10 else np.mean(latencies)
    
    print("-" * 70)
    print(f"Overall Accuracy : {acc:.2f}% [95% CI: {ci_acc[0]}% - {ci_acc[1]}%]")
    print(f"Macro Precision  : {macro_prec:.2f}%")
    print(f"Macro Recall     : {macro_rec:.2f}%")
    print(f"Macro F1-Score   : {macro_f1:.2f}% [95% CI: {ci_f1[0]}% - {ci_f1[1]}%]")
    print(f"Cohen's Kappa (k): {kappa:.4f} [95% CI: {ci_kappa[0]} - {ci_kappa[1]}]")
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
        noise_input = np.expand_dims(noise_uint8, axis=0)
            
        interpreter.set_tensor(input_details['index'], noise_input)
        interpreter.invoke()
        
        raw_out = interpreter.get_tensor(output_details['index'])[0]
        p_raw = np.maximum(raw_out.astype(np.float32) / 255.0, 1e-6)
        p_raw = p_raw / np.sum(p_raw)
        
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
    
    result = {
        "test_support": total_n,
        "confusion_matrix": cm.tolist(),
        "per_class": results,
        "overall_accuracy": round(acc, 2),
        "ci_accuracy_95": ci_acc,
        "macro_precision": round(macro_prec, 2),
        "macro_recall": round(macro_rec, 2),
        "macro_f1": round(macro_f1, 2),
        "ci_macro_f1_95": ci_f1,
        "cohens_kappa": round(kappa, 4),
        "ci_kappa_95": ci_kappa,
        "latency_ms": round(avg_latency, 2),
        "ood_rejection_rate": round(ood_rejection_rate, 2)
    }
    
    out_path = r"f:\GitHub\Research\Research-Paper\system_b_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"\nSaved System B benchmark results to: {out_path}")
    return result

if __name__ == "__main__":
    v3_test_dir = r"f:\GitHub\Research\Datasets\Microscopic_Mobile_Diagnosis_v3_Scientifically_Augmented.multiclass\test"
    model_path = r"f:\GitHub\Research\Coconut-Pathology-Detection-Service\system_b_baseline_int8.tflite"
    evaluate_system_b(v3_test_dir, model_path)
