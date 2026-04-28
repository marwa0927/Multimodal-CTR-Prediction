"""
Train a frozen-embedding CTR model to simulate the Task 1 fixed downstream evaluator.

Item embeddings from outputs/features/item_emb_task1.npy are frozen for ALL epochs.
Best checkpoint saved to outputs/checkpoints/task1_eval_best.pt.
"""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

torch.set_num_threads(8)

import config
from src.data.loader import load_train, load_valid
from src.models.task2_ctr import SASRecCTR

BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 20
PATIENCE = 3
SUBSAMPLE_SIZE = 400_000


class CTRDataset(Dataset):
    def __init__(self, df):
        seqs = np.stack(df["item_seq"].apply(
            lambda s: np.array(s, dtype=np.int64)
        ).values)
        self.item_seq = torch.from_numpy(seqs)
        self.item_id = torch.from_numpy(df["item_id"].to_numpy(dtype=np.int64))
        self.likes = torch.from_numpy(df["likes_level"].to_numpy(dtype=np.int64))
        self.views = torch.from_numpy(df["views_level"].to_numpy(dtype=np.int64))
        self.labels = torch.from_numpy(df["label"].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.item_seq[idx],
            self.item_id[idx],
            self.likes[idx],
            self.views[idx],
            self.labels[idx],
        )


def stratified_subsample(dataset: CTRDataset, n: int, rng: np.random.Generator) -> Subset:
    labels = dataset.labels.numpy()
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos = min(len(pos_idx), n // 2)
    n_neg = min(len(neg_idx), n - n_pos)
    n_pos = min(len(pos_idx), n - n_neg)

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    indices = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(indices)
    return Subset(dataset, indices.tolist())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for item_seq, item_id, likes, views, label in loader:
        item_seq = item_seq.to(device)
        item_id = item_id.to(device)
        likes = likes.to(device)
        views = views.to(device)
        p = model(item_seq, item_id, likes, views).cpu().numpy()
        preds.append(p)
        targets.append(label.numpy())
    return roc_auc_score(np.concatenate(targets), np.concatenate(preds))


def train():
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  threads: {torch.get_num_threads()}")
    print("item embeddings FROZEN for all epochs (task1 eval mode)")

    train_df = load_train()
    valid_df = load_valid()

    train_ds = CTRDataset(train_df)
    valid_ds = CTRDataset(valid_df)

    subsample_size = min(SUBSAMPLE_SIZE, len(train_ds))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # item_emb starts frozen (freeze=True in SASRecCTR __init__) — never unfreeze
    model = SASRecCTR(emb_path=config.FEATURES_DIR / "item_emb_task1.npy").to(device)

    # Only optimise non-embedding parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()

    best_auc = 0.0
    no_improve = 0
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.CHECKPOINTS_DIR / "task1_eval_best.pt"

    epoch_times = []

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_subset = stratified_subsample(train_ds, subsample_size, rng)
        train_loader = DataLoader(
            epoch_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        bar = tqdm(train_loader, desc=f"epoch {epoch:>2}/{MAX_EPOCHS}", unit="batch", leave=False)
        for item_seq, item_id, likes, views, label in bar:
            item_seq = item_seq.to(device)
            item_id = item_id.to(device)
            likes = likes.to(device)
            views = views.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(item_seq, item_id, likes, views)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(label)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_elapsed = time.perf_counter() - t0
        epoch_times.append(epoch_elapsed)

        avg_loss = total_loss / subsample_size
        auc = evaluate(model, valid_loader, device)

        eta_str = ""
        if epoch == 1:
            eta_s = epoch_times[0] * (MAX_EPOCHS - 1)
            eta_str = f"  (est. remaining: {eta_s/60:.1f} min for {MAX_EPOCHS - 1} more epochs)"

        print(f"epoch {epoch:>2}/{MAX_EPOCHS}  loss={avg_loss:.4f}  valid_auc={auc:.4f}  "
              f"[{epoch_elapsed:.0f}s]{eta_str}")

        if auc > best_auc:
            best_auc = auc
            no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "auc": auc}, ckpt_path)
            print(f"  ✓ best checkpoint saved (auc={best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    total_min = sum(epoch_times) / 60
    print(f"\ntraining complete in {total_min:.1f} min. best valid AUC: {best_auc:.4f}")
    print(f"checkpoint → {ckpt_path}")


if __name__ == "__main__":
    train()
