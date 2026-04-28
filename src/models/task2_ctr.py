"""
SASRec-style CTR model (Task 2).

Input:
    item_seq       (B, 64)  — left-zero-padded history
    target_item_id (B,)     — item whose pCTR we predict
    likes_level    (B,)     — popularity signal, values 1-10
    views_level    (B,)     — popularity signal, values 1-10

Output: sigmoid scalar (B,)  — pCTR
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).parents[2]
_EMB_PATH = _ROOT / "outputs" / "features" / "item_emb_task1.npy"

SEQ_LEN = 64
ITEM_DIM = 128
POP_DIM = 16
MLP_IN = ITEM_DIM + ITEM_DIM + POP_DIM + POP_DIM  # 288


class SASRecCTR(nn.Module):
    def __init__(self, emb_path: Path = _EMB_PATH, dropout: float = 0.2):
        super().__init__()

        emb_matrix = np.load(emb_path).astype(np.float32)  # (91718, 128)
        emb_tensor = torch.from_numpy(emb_matrix)
        # freeze=True initially; call unfreeze_item_emb() after 2 epochs
        self.item_emb = nn.Embedding.from_pretrained(emb_tensor, freeze=True, padding_idx=0)

        self.pos_emb = nn.Embedding(SEQ_LEN, ITEM_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ITEM_DIM, nhead=2, dim_feedforward=256,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.likes_emb = nn.Embedding(11, POP_DIM)
        self.views_emb = nn.Embedding(11, POP_DIM)

        self.mlp = nn.Sequential(
            nn.Linear(MLP_IN, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def unfreeze_item_emb(self):
        self.item_emb.weight.requires_grad_(True)

    def forward(
        self,
        item_seq: torch.Tensor,       # (B, 64)
        target_item_id: torch.Tensor, # (B,)
        likes_level: torch.Tensor,    # (B,)
        views_level: torch.Tensor,    # (B,)
    ) -> torch.Tensor:                # (B,)
        B, L = item_seq.shape

        seq_emb = self.item_emb(item_seq)  # (B, L, 128)
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0)
        seq_emb = seq_emb + self.pos_emb(positions)

        # padding mask: True = ignore (PyTorch convention)
        pad_mask = item_seq == 0  # (B, L)

        # Prevent all-pad rows from producing NaN in attention softmax
        all_pad = pad_mask.all(dim=1)  # (B,)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, -1] = False

        seq_out = self.transformer(seq_emb, src_key_padding_mask=pad_mask)  # (B, L, 128)

        # For left-zero-padded sequences the most recent item is always at position L-1
        last_out = seq_out[:, -1, :]  # (B, 128)

        target_emb = self.item_emb(target_item_id)   # (B, 128)
        likes_e = self.likes_emb(likes_level)         # (B, 16)
        views_e = self.views_emb(views_level)         # (B, 16)

        x = torch.cat([last_out, target_emb, likes_e, views_e], dim=-1)  # (B, 288)
        logit = self.mlp(x).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)
