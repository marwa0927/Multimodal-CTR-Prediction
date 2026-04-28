"""
Build Task 1 submission CSV from scores_task1_v1.npy.
Saves to outputs/submissions/sub_task1_v1.csv with columns: ID, label.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

import config
from src.data.loader import load_test

SCORES_PATH = config.OUTPUTS_DIR / "submissions" / "scores_task1_v1.npy"
OUT_PATH = config.OUTPUTS_DIR / "submissions" / "sub_task1_v1.csv"
EXPECTED_ROWS = 379_142


def main():
    test_df = load_test()
    scores = np.load(SCORES_PATH)

    assert len(test_df) == EXPECTED_ROWS, (
        f"test.parquet has {len(test_df)} rows, expected {EXPECTED_ROWS}"
    )
    assert len(scores) == EXPECTED_ROWS, (
        f"scores_task1_v1.npy has {len(scores)} rows, expected {EXPECTED_ROWS}"
    )

    sub = pd.DataFrame({"ID": test_df["ID"].values, "label": scores})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)

    print(sub.head())
    print(f"\nrow count: {len(sub)}  ✓")
    print(f"saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
