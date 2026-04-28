"""
Evaluate the best Task 2 checkpoint on valid.parquet and print AUC.

Usage:
    python src/evaluate.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import config
from src.data.loader import load_valid
from src.models.task2_ctr import SASRecCTR
from src.training.train_task2 import CTRDataset

BATCH_SIZE = 2048


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = config.CHECKPOINTS_DIR / "task2_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = SASRecCTR().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']} (saved auc={ckpt['auc']:.4f})")

    valid_df = load_valid()
    valid_ds = CTRDataset(valid_df)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds, targets = [], []
    with torch.no_grad():
        for item_seq, item_id, likes, views, label in valid_loader:
            item_seq = item_seq.to(device)
            item_id = item_id.to(device)
            likes = likes.to(device)
            views = views.to(device)
            p = model(item_seq, item_id, likes, views).cpu().numpy()
            preds.append(p)
            targets.append(label.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    auc = roc_auc_score(targets, preds)
    print(f"valid AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
