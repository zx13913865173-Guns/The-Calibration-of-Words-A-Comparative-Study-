import json
import numpy as np

def compute_matched_value(responses):
    """
    responses: list of dict, each has 'reference' and 'matched_comfy'
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in responses:
        groups[r["reference"]].append(r["matched_comfy"])
    result = {}
    for ref, vals in groups.items():
        arr = np.array(vals, dtype=float)
        result[ref] = {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)),
            "ci95": [float(arr.mean() - 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))),
                     float(arr.mean() + 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))]
        }
    return result

if __name__ == "__main__":
    # 从 subjective 评分表读取
    pass
