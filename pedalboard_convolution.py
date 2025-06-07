import os
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Convolution
from scipy import signal
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


def pedalboard_convolution(input_wav: str, ir_wav: str) -> np.ndarray:

    x, sr_x = sf.read(input_wav)
    h, sr_h = sf.read(ir_wav)
    if sr_x != sr_h:
        raise ValueError(f"Sampling rates do not match: {sr_x} vs {sr_h}")

    if x.ndim > 1:
        x = x[:, 0]
    if h.ndim > 1:
        h = h[:, 0]

    board = Pedalboard([Convolution(ir_wav)])
   
    y = board.process(x, sr_x, reset=True)
    zeros = np.zeros(len(h), dtype=x.dtype)
    tail = board.process(zeros, sr_x, reset=False)

    y_full = np.concatenate([y, tail])
    return y_full, sr_x


def normalize_to_reference(ref_wav: str, target: np.ndarray, sr: int, method: str = 'peak') -> np.ndarray:
  
    ref, sr_r = sf.read(ref_wav)
    if sr_r != sr:
        raise ValueError(f"Sampling rates do not match: {sr_r} vs {sr}")
    if ref.ndim > 1:
        ref = ref[:, 0]

    if method == 'peak':
        peak_ref = np.max(np.abs(ref))
        peak_tgt = np.max(np.abs(target))
        if peak_tgt > 0:
            target = target * (peak_ref / peak_tgt)
    elif method == 'rms':
        rms_ref = np.sqrt(np.mean(ref**2))
        rms_tgt = np.sqrt(np.mean(target**2))
        if rms_tgt > 0:
            target = target * (rms_ref / rms_tgt)
    else:
        raise ValueError("method must be 'peak' or 'rms',or 'both'")
    return target


def level_difference_db(ref: np.ndarray, target: np.ndarray) -> tuple[float, float]:
   
    min_len = min(len(ref), len(target))
    ref, target = ref[:min_len], target[:min_len]

    # Peak diff
    peak_ref = np.max(np.abs(ref))
    peak_tgt = np.max(np.abs(target))
    peak_diff_db = 20 * np.log10(peak_tgt / peak_ref) if peak_ref > 0 else 0.0
    
    # RMS diff
    rms_ref = np.sqrt(np.mean(ref**2))
    rms_tgt = np.sqrt(np.mean(target**2))
    rms_diff_db = 20 * np.log10(rms_tgt / rms_ref) if rms_ref > 0 else 0.0
    return peak_diff_db, rms_diff_db


def compute_level_differences(reference_dir: str, target_dir: str, output_json: str):
  
    results = {}
    for fn in sorted(os.listdir(reference_dir)):
        if not fn.lower().endswith('.wav'):
            continue
        base = os.path.splitext(fn)[0]
        ref_path = os.path.join(reference_dir, fn)
        tgt_path = os.path.join(target_dir, fn)
        if not os.path.exists(tgt_path):
            print(f"Target not found for {fn} in {target_dir}")
            continue
       
        ref, sr = sf.read(ref_path)
        if ref.ndim > 1:
            ref = ref[:, 0]
        
        tgt, sr2 = sf.read(tgt_path)
        if tgt.ndim > 1:
            tgt = tgt[:, 0]
        if sr2 != sr:
            raise ValueError(f"Sampling rate mismatch for '{fn}': {sr2} vs {sr}")
       
        peak_db, rms_db = level_difference_db(ref, tgt)
        results[base] = {
            'peak_diff_db': round(peak_db, 2),
            'rms_diff_db': round(rms_db, 2)
        }
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {output_json}")

