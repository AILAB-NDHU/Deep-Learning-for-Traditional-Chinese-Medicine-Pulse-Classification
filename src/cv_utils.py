"""
Cross-validation utilities for inner-layer subject-level validation splitting.
Used to prevent test set leakage into early stopping signal.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_inner_split(y_tr_full, groups_tr_full, n_inner=5, seed=42):
    """
    Create an inner validation split from training fold using StratifiedGroupKFold.
    
    This ensures:
    1. Same subject never appears in both inner_train and inner_val
    2. Stratified split preserves label distribution
    3. Rare classes (Tight) are likely represented in inner_val
    
    Parameters
    ----------
    y_tr_full : array-like
        Labels for the full training fold
    groups_tr_full : array-like
        Group/subject IDs for the full training fold
    n_inner : int, default=5
        Number of inner folds (inner_val will be ~1/n_inner of data)
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    inner_tr_idx : ndarray
        Indices (relative to training fold) for inner training set
    inner_val_idx : ndarray
        Indices (relative to training fold) for inner validation set
        
    Examples
    --------
    >>> inner_tr, inner_val = make_inner_split(y_tr, groups_tr, n_inner=5, seed=42)
    >>> assert set(groups_tr[inner_tr]) & set(groups_tr[inner_val]) == set()  # No overlap
    """
    sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    inner_tr_idx, inner_val_idx = next(sgkf.split(
        np.zeros(len(y_tr_full)), y_tr_full, groups_tr_full))
    return inner_tr_idx, inner_val_idx


def validate_inner_split(inner_tr_idx, inner_val_idx, groups_tr_full, y_tr_full):
    """
    Validate that inner split is correct.
    
    Parameters
    ----------
    inner_tr_idx : ndarray
        Indices for inner training set
    inner_val_idx : ndarray
        Indices for inner validation set
    groups_tr_full : array-like
        Subject IDs for full training fold
    y_tr_full : array-like
        Labels for full training fold
        
    Returns
    -------
    validation_results : dict
        Dictionary with validation checks:
        - 'no_subject_overlap': bool, whether same subject appears in both sets
        - 'inner_tr_subjects': int, count of unique subjects in inner_train
        - 'inner_val_subjects': int, count of unique subjects in inner_val
        - 'inner_val_has_rare_class': bool, whether inner_val contains rare class
    """
    groups_inner_tr = groups_tr_full[inner_tr_idx]
    groups_inner_val = groups_tr_full[inner_val_idx]
    
    # Check for subject overlap
    overlap = set(groups_inner_tr) & set(groups_inner_val)
    no_overlap = len(overlap) == 0
    
    # Count unique subjects
    n_subjects_inner_tr = len(set(groups_inner_tr))
    n_subjects_inner_val = len(set(groups_inner_val))
    
    # Check for rare class in inner_val (assuming rarest class is 'Tight')
    y_inner_val = y_tr_full[inner_val_idx]
    # Most common class (assuming binary or multiclass)
    unique_classes = np.unique(y_tr_full)
    rare_class = None
    min_count = float('inf')
    for cls in unique_classes:
        count = (y_tr_full == cls).sum()
        if count < min_count:
            min_count = count
            rare_class = cls
    
    has_rare_class = (y_inner_val == rare_class).sum() > 0
    
    results = {
        'no_subject_overlap': no_overlap,
        'inner_tr_subjects': n_subjects_inner_tr,
        'inner_val_subjects': n_subjects_inner_val,
        'inner_val_has_rare_class': has_rare_class,
        'subject_overlap': overlap,
        'inner_tr_n_samples': len(inner_tr_idx),
        'inner_val_n_samples': len(inner_val_idx),
    }
    
    return results
