"""Random Forest classifier wrapper. Trains on PaySim with SMOTE oversampling
to address the ~0.13% class imbalance, returns the fitted estimator plus
training metadata.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE, RF_PARAMS


@dataclass
class RFTrainingResult:
    model: RandomForestClassifier
    feature_importances: dict[str, float]
    n_train: int
    n_train_after_smote: int
    fraud_rate_before: float
    fraud_rate_after: float


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    use_smote: bool = True,
) -> RFTrainingResult:
    """Fit a RandomForest, optionally with SMOTE to balance fraud class."""
    n_before = len(y_train)
    fraud_rate_before = float(np.mean(y_train))

    if use_smote and np.sum(y_train == 1) > 5:
        # SMOTE needs at least k_neighbors+1 minority samples (default k=5)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, np.sum(y_train == 1) - 1))
        x_train, y_train = smote.fit_resample(x_train, y_train)

    n_after = len(y_train)
    fraud_rate_after = float(np.mean(y_train))

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(x_train, y_train)

    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    return RFTrainingResult(
        model=model,
        feature_importances=importances,
        n_train=n_before,
        n_train_after_smote=n_after,
        fraud_rate_before=fraud_rate_before,
        fraud_rate_after=fraud_rate_after,
    )
