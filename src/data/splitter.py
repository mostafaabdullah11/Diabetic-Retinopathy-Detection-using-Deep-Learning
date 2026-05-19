"""
splitter.py — Stratified K-Fold split utility
Returns DataFrames (not just indices) for cleaner dataset creation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple


def get_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Splits a DataFrame into stratified train/val folds.

    Args:
        df           : full DataFrame with 'diagnosis' column
        n_splits     : number of folds
        random_state : reproducibility seed

    Returns:
        List of (train_df, val_df) tuples, one per fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = df['diagnosis'].values
    folds = []

    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        folds.append((train_df, val_df))

    return folds
