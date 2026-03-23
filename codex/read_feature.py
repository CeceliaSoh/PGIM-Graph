from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


FEATURE_PATH = Path("dataset/feature.npy")
FEATURE_MASK_PATH = Path("dataset/feature_mask.npy")
GRAPH_PATH = Path("dataset/graph.pt")
OUTPUT_PATH = Path("dataset/nonzero_node_feature.txt")


features = np.load(FEATURE_PATH)
feature_mask = np.load(FEATURE_MASK_PATH) if FEATURE_MASK_PATH.exists() else np.any(features != 0, axis=2)
node_names = None

if torch is not None and GRAPH_PATH.exists():
    graph_data = torch.load(GRAPH_PATH, map_location="cpu")
    node_names = graph_data.get("node_names")

matches = np.argwhere(feature_mask)

print(f"feature shape: {features.shape}")
print(f"feature mask shape: {feature_mask.shape}")
print(f"observed non-zero node-time pairs: {len(matches)}")

if len(matches) == 0:
    message = "No non-zero node feature row found."
    print(message)
    OUTPUT_PATH.write_text(message + "\n", encoding="utf-8")
else:
    time_idx, node_idx = matches[0]
    node_feature = features[time_idx, node_idx]
    lines = [
        f"feature shape: {features.shape}",
        f"feature mask shape: {feature_mask.shape}",
        f"observed non-zero node-time pairs: {len(matches)}",
        f"time index: {time_idx}",
        f"node index: {node_idx}",
    ]
    if node_names is not None:
        lines.append(f"node name: {node_names[node_idx]}")
    lines.append("node feature:")
    lines.extend(f"{value:.6f}" for value in node_feature)

    report = "\n".join(lines) + "\n"
    print(report, end="")
    OUTPUT_PATH.write_text(report, encoding="utf-8")
