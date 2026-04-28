"""
Train the tag encoder (Task 1) and produce blended item embeddings.

Output:
    outputs/features/item_emb_task1.npy  — shape (91718, 128), indexed by item_id
    outputs/checkpoints/task1.pt         — model state dict
"""

import sys
from pathlib import Path

# make project root and src/data importable from any working directory
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src" / "data"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from src.data.loader import load_item_info
from src.models.task1_emb import TagEncoder


def _pad_tags(tag_list: list[int], max_len: int) -> list[int]:
    tags = [t for t in tag_list if t > 0][:max_len]
    return tags + [0] * (max_len - len(tags))


def build_tensors(item_info, max_tags: int):
    """Return (item_ids, tag_tensor, pretrained_emb_tensor) excluding item_id=0."""
    rows = item_info[item_info["item_id"] != 0].reset_index(drop=True)

    item_ids = rows["item_id"].to_numpy(dtype=np.int32)

    tag_matrix = np.array(
        [_pad_tags(tags, max_tags) for tags in rows["item_tags"]],
        dtype=np.int64,
    )

    emb_matrix = np.stack(
        rows["item_emb_d128"].apply(lambda e: np.asarray(e, dtype=np.float32)).values
    )  # (N, 128)

    return item_ids, tag_matrix, emb_matrix


def train(epochs: int = config.T1_EPOCHS,
          lr: float = config.T1_LR,
          batch_size: int = config.T1_BATCH_SIZE,
          device_str: str = "auto") -> None:

    torch.manual_seed(config.SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    print(f"device: {device}")

    # ── data ──────────────────────────────────────────────────────────────
    item_info = load_item_info()
    item_ids, tag_matrix, emb_matrix = build_tensors(item_info, config.MAX_TAGS)

    tag_t   = torch.from_numpy(tag_matrix).to(device)     # (N, MAX_TAGS)
    target_t = torch.from_numpy(emb_matrix).to(device)    # (N, 128)

    dataset = TensorDataset(tag_t, target_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── model ─────────────────────────────────────────────────────────────
    model = TagEncoder(
        vocab_size=config.TAG_VOCAB_SIZE,
        tag_dim=config.TAG_EMB_DIM,
        out_dim=config.ITEM_EMB_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for tags_b, target_b in loader:
            optimizer.zero_grad()
            pred = model(tags_b)
            loss = criterion(pred, target_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(tags_b)
        avg = total_loss / len(dataset)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs}  mse={avg:.6f}")

    # ── save checkpoint ───────────────────────────────────────────────────
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.CHECKPOINTS_DIR / "task1.pt"
    torch.save({"model_state": model.state_dict(), "epoch": epochs}, ckpt_path)
    print(f"checkpoint → {ckpt_path}")

    # ── produce final embeddings ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        tag_out = model(tag_t).cpu().numpy()   # (N, 128)  tag-model output

    # detect items whose pre-trained embedding is all-zero
    zero_mask = (emb_matrix == 0).all(axis=1)  # (N,)

    blended = np.where(
        zero_mask[:, None],
        tag_out,                                           # tag model only
        config.T1_BLEND_ALPHA * tag_out + (1 - config.T1_BLEND_ALPHA) * emb_matrix,
    ).astype(np.float32)

    # build full array indexed by item_id  (91718 rows, row 0 = padding = zeros)
    n_items = int(item_info["item_id"].max()) + 1   # should be 91718
    final_emb = np.zeros((n_items, config.ITEM_EMB_DIM), dtype=np.float32)
    for i, iid in enumerate(item_ids):
        final_emb[iid] = blended[i]

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FEATURES_DIR / "item_emb_task1.npy"
    np.save(out_path, final_emb)
    print(f"embeddings → {out_path}  shape={final_emb.shape}")

    # ── verification ──────────────────────────────────────────────────────
    loaded = np.load(out_path)
    assert loaded.shape == (91718, 128), f"unexpected shape {loaded.shape}"
    print(f"verification OK: loaded shape = {loaded.shape}")


if __name__ == "__main__":
    train()
