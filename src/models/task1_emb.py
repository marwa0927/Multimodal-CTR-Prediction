import torch
import torch.nn as nn


class TagEncoder(nn.Module):
    """
    tag_ids (B, MAX_TAGS) → mean-pool non-zero tags → 2-layer MLP → 128-dim output.
    padding_idx=0 keeps the pad token at zero and excludes it from gradients.
    """

    def __init__(self, vocab_size: int = 11_740, tag_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, tag_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(tag_dim, tag_dim),
            nn.ReLU(),
            nn.Linear(tag_dim, out_dim),
        )

    def forward(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tag_ids: LongTensor (B, T), zero-padded on the left or right.
        Returns:
            FloatTensor (B, out_dim)
        """
        mask = (tag_ids != 0).float()                          # (B, T)
        emb  = self.embedding(tag_ids)                         # (B, T, tag_dim)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)   # (B, 1)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, tag_dim)
        return self.mlp(pooled)                                 # (B, out_dim)
