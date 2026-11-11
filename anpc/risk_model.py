"""Random-forest risk model and its held-out evaluation (Sections 4.2 & 6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Config


@dataclass
class RiskMetrics:
    """Container for every quantity reported in Table 2."""

    test_nodes: int
    failure_proportion: float
    roc_auc: float
    average_precision: float
    brier: float
    true_healthy: int
    false_alarms: int
    missed_failures: int
    found_failures: int
    precision: float
    recall: float
    threshold: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "test_nodes": self.test_nodes,
            "failure_proportion": self.failure_proportion,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "brier": self.brier,
            "true_healthy": self.true_healthy,
            "false_alarms": self.false_alarms,
            "missed_failures": self.missed_failures,
            "found_failures": self.found_failures,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
        }


class RiskModel:
    """Thin wrapper around scikit-learn's random forest."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model = RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_features=cfg.rf_max_features,
            class_weight=cfg.rf_class_weight,
            random_state=cfg.rf_random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "RiskModel":
        self.model.fit(features, labels)
        self._fitted = True
        return self

    def predict_risk(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.asarray([], dtype=float)
        return self.model.predict_proba(features)[:, 1]

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def evaluate(
        self, features: np.ndarray, labels: np.ndarray
    ) -> RiskMetrics:
        """Compute the metrics reported in Table 2."""
        probabilities = self.predict_risk(features)
        predictions = (
            probabilities >= self.cfg.classification_threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()

        return RiskMetrics(
            test_nodes=int(len(labels)),
            failure_proportion=float(np.mean(labels)),
            roc_auc=float(roc_auc_score(labels, probabilities)),
            average_precision=float(average_precision_score(labels, probabilities)),
            brier=float(brier_score_loss(labels, probabilities)),
            true_healthy=int(tn),
            false_alarms=int(fp),
            missed_failures=int(fn),
            found_failures=int(tp),
            precision=float(precision_score(labels, predictions, zero_division=0)),
            recall=float(recall_score(labels, predictions, zero_division=0)),
            threshold=float(self.cfg.classification_threshold),
        )
