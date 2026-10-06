import numpy as np
from sklearn.metrics import r2_score

def dynamic_range(clip_scores):
    return float(np.max(clip_scores) - np.min(clip_scores))

def linearity_r2(weight_values, clip_scores):
    weight_values = np.asarray(weight_values, dtype=float)
    clip_scores = np.asarray(clip_scores, dtype=float)
    coeffs = np.polyfit(weight_values, clip_scores, 1)
    pred = np.polyval(coeffs, weight_values)
    return float(r2_score(clip_scores, pred)), coeffs

def compression_rate(weight_values, clip_scores):
    weight_values = np.asarray(weight_values, dtype=float)
    clip_scores = np.asarray(clip_scores, dtype=float)
    coeffs = np.polyfit(weight_values, clip_scores, 1)
    theoretical_min = np.polyval(coeffs, weight_values.min())
    theoretical_max = np.polyval(coeffs, weight_values.max())
    theoretical_range = theoretical_max - theoretical_min
    actual_range = clip_scores.max() - clip_scores.min()
    if theoretical_range == 0:
        return 0.0
    return float(1.0 - actual_range / theoretical_range)

def saturation_point(weight_values, clip_scores, threshold=0.005):
    diffs = np.diff(clip_scores)
    idx = np.where(np.abs(diffs) < threshold)[0]
    if len(idx) == 0:
        return None
    return weight_values[idx[0] + 1]
