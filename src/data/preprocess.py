"""
Preprocessing pipeline for CICIoT2023 data.

- Loads raw CSVs (train, test, validation).
- Replaces infinities, drops NaNs, clips outliers.
- Creates binary label column (0 = benign, 1 = attack).
- StandardScaler normalisation (fit on train, transform all splits).
- Saves to data/processed/ as parquet files + scaler.joblib.

CLI: python -m src.data.preprocess --config config/experiment.yaml [--nrows N]
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config/experiment.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _clean(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=feat_cols, inplace=True)
    for col in feat_cols:
        lo, hi = df[col].quantile(0.001), df[col].quantile(0.999)
        df[col] = df[col].clip(lo, hi)
    return df


def run_preprocessing(config_path: str, nrows: int = None) -> None:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = cfg["data"]["flow_feature_columns"]
    label_col = cfg["data"]["label_column"]
    benign = set(cfg["data"]["benign_labels"])

    splits = {}
    for split in ("train", "test", "validation"):
        p = raw_dir / f"{split}.csv"
        if not p.exists():
            logger.warning("Missing %s — skipping", p)
            continue
        logger.info("Loading %s ...", p)
        df = pd.read_csv(p, nrows=nrows)
        df["binary_label"] = df[label_col].apply(lambda x: 0 if x in benign else 1)
        df = _clean(df, feat_cols)
        splits[split] = df
        logger.info("  %s: %d rows (benign=%d, attack=%d)",
                     split, len(df),
                     (df["binary_label"] == 0).sum(),
                     (df["binary_label"] == 1).sum())

    if "train" not in splits:
        logger.error("No train split found in %s", raw_dir)
        return

    scaler = StandardScaler()
    scaler.fit(splits["train"][feat_cols])
    for split, df in splits.items():
        df[feat_cols] = scaler.transform(df[feat_cols])
        df.to_parquet(out_dir / f"{split}.parquet", index=False)
        logger.info("Saved %s -> %s", split, out_dir / f"{split}.parquet")

    joblib.dump(scaler, out_dir / "scaler.joblib")
    logger.info("Saved scaler -> %s", out_dir / "scaler.joblib")


def main():
    ap = argparse.ArgumentParser(description="Preprocess CICIoT2023")
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_preprocessing(args.config, args.nrows)


if __name__ == "__main__":
    main()
