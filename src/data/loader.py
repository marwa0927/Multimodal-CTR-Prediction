import pandas as pd
from pathlib import Path

_DATA = Path(__file__).parents[2] / "data"


def load_train() -> pd.DataFrame:
    df = pd.read_parquet(_DATA / "train.parquet")
    print(f"train:     {df.shape}")
    return df


def load_valid() -> pd.DataFrame:
    df = pd.read_parquet(_DATA / "valid.parquet")
    print(f"valid:     {df.shape}")
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_parquet(_DATA / "test.parquet")
    print(f"test:      {df.shape}")
    return df


def load_item_info() -> pd.DataFrame:
    df = pd.read_parquet(_DATA / "item_info.parquet")
    print(f"item_info: {df.shape}")
    return df


if __name__ == "__main__":
    train = load_train()
    valid = load_valid()
    test  = load_test()
    items = load_item_info()

    print("\n--- train sample ---")
    print(train.head(2).to_string())
    print("\n--- item_info sample ---")
    print(items.head(2).to_string())
    print("\n--- dtypes ---")
    print(train.dtypes)
