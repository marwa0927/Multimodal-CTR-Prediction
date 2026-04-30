# Multimodal CTR Prediction — WWW 2025 EReL@MIR Challenge


WWW 2025 EReL@MIR Workshop MM-CTR Challenge.

---

## Overview

This project implements a cascaded multimodal CTR prediction pipeline on the MicroLens-1M dataset (1M users, 91K items). The pipeline consists of two separate, disjoint models:

- **Task 1** — Multimodal item embedding learning (128-dim output)
- **Task 2** — Sequential CTR prediction using those embeddings

---

## Architecture

### Task 1 — Item Embedding Model
- Tag embedding table (vocab=11,740, dim=128) with mean pooling over variable-length tag lists
- Trained with combined MSE + InfoNCE contrastive loss
- InfoNCE uses co-occurrence pairs extracted from user sequences (5.4M unique pairs from 1.8M sequences)
- Final embedding: `0.4 × tag_model_output + 0.6 × pretrained_item_emb_d128`
- Output: 128-dim embedding per item, saved to `outputs/features/item_emb_task1.npy`

### Task 2 — CTR Model (SASRec + DCNv2 + Target Attention)

Inspired by the 1st place solution (arxiv: 2505.03543), with the following architecture:

**Embedding Layer**
- Learnable item ID embedding (vocab=91,719, dim=64)
- Frozen multimodal embedding from Task 1 (dim=128)
- Item representation: concat(ID_emb, mm_emb) = 192-dim
- Learnable user ID embedding (vocab=1,000,001, dim=64)
- Popularity embeddings: likes_level + views_level (dim=16 each)
- Learnable positional encoding (max_len=64)

**Sequential Feature Learning**
- 4-layer Transformer encoder, 4 attention heads, dim=192, feedforward=768
- Padding mask for zero item IDs
- DIN-style target attention over all sequence positions (query = target item)
- Sequence representation: concat(last_position, target_attention_output) = 384-dim

**Feature Interaction — DCNv2 (Parallel)**
- Input: concat(seq_384, target_item_192, user_64, likes_16, views_16) = 672-dim
- Cross network: 3 cross layers
- Deep network: 672 → 1024 → 512 → 256, Dice activation, dropout=0.2
- Output: concat(cross_672, deep_256) = 928-dim

**Prediction Layer**
- 928 → 64 → 32 → 1, ReLU, sigmoid

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 2e-4 (linear warmup 3 epochs, cosine decay) |
| Batch size | 2048 × N_GPUs |
| Training data | 50% stratified sample (1.8M rows) |
| Max epochs | 40 |
| Early stopping | patience=8 |
| Grad clip | 0.5 |
| Hardware | 2× Tesla T4 |

---

## Results

| Metric | Local Valid AUC | Leaderboard AUC |
|--------|----------------|-----------------|
| Task 1 | 0.8713 | 0.8867 |
| Task 2 | 0.9333 | 0.9479 |
| Combined | — | **0.9489** |

---

## Dataset

MicroLens-1M ([Westlake University](https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/)):
- 3.6M training interactions, 10K validation, 379K test
- 91,718 items with pre-trained 128-dim embeddings and tag metadata
- User sequences up to length 64 (left-zero-padded)

---

## Key Design Decisions

1. **Target attention over plain self-attention** — DIN-style attention between target item and sequence positions captures item-specific user interest more effectively than using only the last sequence position
2. **DCNv2 over MLP** — Explicit cross-feature interactions outperform standard deep networks for CTR tasks
3. **Cascaded training** — Task 1 embeddings are frozen in Task 2, preventing joint training while still benefiting from richer item representations
4. **Co-occurrence pretraining** — InfoNCE loss on sequential item pairs gives the tag encoder a collaborative filtering signal beyond pure tag reconstruction

---

## References

- [1st Place Solution WWW 2025 MM-CTR Challenge](https://arxiv.org/abs/2505.03543)
- [DCNv2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [DIN: Deep Interest Network](https://arxiv.org/abs/1706.06978)
