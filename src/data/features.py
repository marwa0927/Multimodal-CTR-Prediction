import numpy as np
import pandas as pd
from typing import List

from loader import load_item_info

EMB_DIM = 128


def build_item_feature_matrix(item_info: pd.DataFrame | None = None) -> dict[int, np.ndarray]:
    """Returns {item_id: 128-dim ndarray} using the pre-trained item_emb_d128 column."""
    if item_info is None:
        item_info = load_item_info()
    lookup = {
        int(row.item_id): np.asarray(row.item_emb_d128, dtype=np.float32)
        for row in item_info.itertuples(index=False)
    }
    print(f"build_item_feature_matrix: {len(lookup)} items, emb dim {EMB_DIM}")
    return lookup


def encode_tags(item_info: pd.DataFrame | None = None) -> dict[int, np.ndarray]:
    """Returns {item_id: mean-pooled tag embedding (128-dim)} via one-hot mean pooling over tag IDs."""
    if item_info is None:
        item_info = load_item_info()

    all_tags: list[list[int]] = item_info["item_tags"].tolist()
    flat = [t for tags in all_tags for t in tags if t > 0]
    n_tags = max(flat) + 1 if flat else 1

    lookup: dict[int, np.ndarray] = {}
    for row in item_info.itertuples(index=False):
        tags = [t for t in row.item_tags if t > 0]
        if tags:
            one_hot = np.zeros(n_tags, dtype=np.float32)
            for t in tags:
                one_hot[t] += 1.0
            emb = one_hot / one_hot.sum()
        else:
            emb = np.zeros(n_tags, dtype=np.float32)
        # Truncate or pad to EMB_DIM so downstream modules see a fixed width
        if len(emb) >= EMB_DIM:
            emb = emb[:EMB_DIM]
        else:
            emb = np.pad(emb, (0, EMB_DIM - len(emb)))
        lookup[int(row.item_id)] = emb

    print(f"encode_tags: {len(lookup)} items, tag vocab {n_tags}, output dim {EMB_DIM}")
    return lookup


def seq_stats(
    item_seq: List[int],
    emb_lookup: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns (mean_emb, last_emb, seq_len) for a user's item sequence.

    Pads of 0 are skipped. Falls back to zero vectors if the sequence is empty.
    """
    valid_ids = [iid for iid in item_seq if iid != 0 and iid in emb_lookup]
    seq_len = len(valid_ids)

    if seq_len == 0:
        zero = np.zeros(EMB_DIM, dtype=np.float32)
        return zero, zero, 0

    embs = np.stack([emb_lookup[iid] for iid in valid_ids])  # (seq_len, 128)
    mean_emb = embs.mean(axis=0)
    last_emb  = embs[-1]
    return mean_emb, last_emb, seq_len


if __name__ == "__main__":
    item_info = load_item_info()

    print("\n--- build_item_feature_matrix ---")
    emb_map = build_item_feature_matrix(item_info)
    sample_id = next(iter(emb_map))
    print(f"  item {sample_id}: shape={emb_map[sample_id].shape}  "
          f"first3={emb_map[sample_id][:3]}")

    print("\n--- encode_tags ---")
    tag_map = encode_tags(item_info)
    print(f"  item {sample_id}: shape={tag_map[sample_id].shape}  "
          f"first3={tag_map[sample_id][:3]}")

    print("\n--- seq_stats ---")
    example_seq = list(item_info["item_id"].iloc[:10]) + [0, 0, 0]
    mean_e, last_e, slen = seq_stats(example_seq, emb_map)
    print(f"  seq_len={slen}")
    print(f"  mean_emb shape={mean_e.shape}  first3={mean_e[:3]}")
    print(f"  last_emb shape={last_e.shape}  first3={last_e[:3]}")
