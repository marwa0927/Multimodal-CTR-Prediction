"""
Run inference with the best task1_eval checkpoint on test.parquet.
Saves raw pCTR scores to outputs/submissions/scores_task1_v1.npy.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from src.data.loader import load_test
from src.models.task2_ctr import SASRecCTR

BATCH_SIZE = 4096
CKPT_PATH = config.CHECKPOINTS_DIR / "task1_eval_best.pt"
OUT_PATH = config.OUTPUTS_DIR / "submissions" / "scores_task1_v1.npy"


class TestDataset(Dataset):
    def __init__(self, df):
        seqs = np.stack(df["item_seq"].apply(
            lambda s: np.array(s, dtype=np.int64)
        ).values)
        self.item_seq = torch.from_numpy(seqs)
        self.item_id = torch.from_numpy(df["item_id"].to_numpy(dtype=np.int64))
        self.likes = torch.from_numpy(df["likes_level"].to_numpy(dtype=np.int64))
        self.views = torch.from_numpy(df["views_level"].to_numpy(dtype=np.int64))

    def __len__(self):
        return len(self.item_id)

    def __getitem__(self, idx):
        return self.item_seq[idx], self.item_id[idx], self.likes[idx], self.views[idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    test_df = load_test()

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = SASRecCTR(emb_path=config.FEATURES_DIR / "item_emb_task1.npy").to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']}  (val AUC={ckpt['auc']:.4f})")

    dataset = TestDataset(test_df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    scores = []
    with torch.no_grad():
        for item_seq, item_id, likes, views in tqdm(loader, desc="inference", unit="batch"):
            item_seq = item_seq.to(device)
            item_id = item_id.to(device)
            likes = likes.to(device)
            views = views.to(device)
            preds = model(item_seq, item_id, likes, views).cpu().numpy()
            scores.append(preds)

    scores = np.concatenate(scores).astype(np.float32)
    print(f"scores shape: {scores.shape}  min={scores.min():.4f}  max={scores.max():.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, scores)
    print(f"saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
