"""Normalized recoverability per prereg §3.

R(F, c) on a normalized [0,1] scale:
    categorical  -> (accuracy - chance) / (1 - chance)
    continuous   -> R^2   (NOT clipped at 0 — clipping biases encoder-gain upward)

Phase 1 fits a SINGLE linear probe per factor (the quality gate). The full
capacity ladder (MLP rungs) and the G / S / epsilon_G metric layer are Phase 2.
Regularization is tuned on a validation split, never on test (prereg §0).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

LOGISTIC_C_GRID = (0.1, 1.0, 10.0)
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)


def normalized_accuracy(acc: float, chance: float) -> float:
    return (acc - chance) / (1.0 - chance)


def linear_recoverability_categorical(
    h_train, y_train, h_val, y_val, h_test, y_test, chance: float
) -> dict:
    """Logistic probe; C tuned on val; returns normalized accuracy on test."""
    scaler = StandardScaler().fit(h_train)
    Xtr, Xva, Xte = scaler.transform(h_train), scaler.transform(h_val), scaler.transform(h_test)

    best_c, best_val = None, -np.inf
    for c in LOGISTIC_C_GRID:
        clf = LogisticRegression(C=c, max_iter=1000)
        clf.fit(Xtr, y_train)
        val_acc = accuracy_score(y_val, clf.predict(Xva))
        if val_acc > best_val:
            best_val, best_c = val_acc, c

    clf = LogisticRegression(C=best_c, max_iter=1000).fit(Xtr, y_train)
    test_acc = accuracy_score(y_test, clf.predict(Xte))
    return {
        "recoverability": float(normalized_accuracy(test_acc, chance)),
        "raw_test_acc": float(test_acc),
        "chance": float(chance),
        "best_hparam": float(best_c),
        "metric": "norm_acc",
    }


def linear_recoverability_continuous(
    h_train, y_train, h_val, y_val, h_test, y_test
) -> dict:
    """Ridge probe; alpha tuned on val; returns R^2 (unclipped) on test."""
    scaler = StandardScaler().fit(h_train)
    Xtr, Xva, Xte = scaler.transform(h_train), scaler.transform(h_val), scaler.transform(h_test)

    best_a, best_val = None, -np.inf
    for a in RIDGE_ALPHA_GRID:
        reg = Ridge(alpha=a).fit(Xtr, y_train)
        val_r2 = r2_score(y_val, reg.predict(Xva))
        if val_r2 > best_val:
            best_val, best_a = val_r2, a

    reg = Ridge(alpha=best_a).fit(Xtr, y_train)
    test_r2 = r2_score(y_test, reg.predict(Xte))
    return {
        "recoverability": float(test_r2),  # unclipped
        "raw_test_acc": float("nan"),
        "best_hparam": float(best_a),
        "metric": "r2",
    }
