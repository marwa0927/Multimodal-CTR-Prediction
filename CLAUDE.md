# Web Mining Competition — Project Rules

## Competition Constraints

- **Tree models are banned**: No LightGBM, XGBoost, CatBoost, or any gradient-boosted tree model.
- **Ensemble methods are banned**: No blending, stacking, or bagging across multiple models.

## Two-Task Structure

- **Task 1** (`task1_emb.py`): Learn item embeddings from tag data → output `outputs/features/item_emb_task1.npy`
- **Task 2** (`task2_ctr.py`): SASRec-style CTR model using item embeddings as input

## Approved Module Structure

```
src/
├── data/
│   ├── loader.py       # load_train(), load_valid(), load_test(), load_item_info()
│   └── features.py     # build_item_feature_matrix(), encode_tags(), seq_stats()
├── models/
│   ├── task1_emb.py    # tag encoder → 128-dim item embeddings (Task 1)
│   └── task2_ctr.py    # SASRec-style CTR model using item embeddings (Task 2)
├── training/
│   ├── train_task1.py  # train embedding model, save outputs/features/item_emb_task1.npy
│   └── train_task2.py  # train CTR model, checkpoint to outputs/checkpoints/
├── inference/
│   ├── predict.py      # score test.parquet
│   └── submit.py       # format → {ID, label} CSV
└── evaluate.py         # AUC on valid.parquet, works for both tasks

config.py               # all paths, seeds, hyperparameters
```

## Output Paths

- `outputs/features/item_emb_task1.npy` — Task 1 learned embeddings
- `outputs/checkpoints/` — Task 2 model weights
- `outputs/submissions/` — submission CSVs
