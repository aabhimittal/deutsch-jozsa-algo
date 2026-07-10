"""Quantum-inspired binary classifier.

This classifier borrows the *global interference* idea from Deutsch-Jozsa. Each
feature value is turned into a phase, all phases for a sample are summed as
complex amplitudes (a mimic of quantum superposition), and the real part of the
resulting amplitude -- the interference pattern -- is used as a decision score.

Just like Deutsch-Jozsa reads out a *global* property of a function rather than
individual outputs, this classifier makes a decision from a global interference
value rather than a single feature.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class QuantumInspiredClassifier:
    """Binary classifier driven by an interference score.

    Parameters
    ----------
    n_features : int
        Informational only; kept for API symmetry with the rest of the project.
    """

    def __init__(self, n_features=4):
        self.n_features = n_features
        self.threshold = 0.0
        self.scaler = None
        self._sign = 1.0

    # -- quantum-inspired transforms ------------------------------------------
    @staticmethod
    def _quantum_phase_encoding(X):
        """Map each feature to a phase in ``[-pi/2, pi/2]`` via ``arctan``."""
        return np.arctan(X)

    @staticmethod
    def _interference_pattern(phases):
        """Sum per-sample phases as complex amplitudes: ``sum(e^{i*phi}) / sqrt(n)``."""
        n = phases.shape[1]
        amplitudes = np.exp(1j * phases)
        return np.sum(amplitudes, axis=1) / np.sqrt(n)

    def _decision_function(self, X):
        phases = self._quantum_phase_encoding(X)
        psi = self._interference_pattern(phases)
        return np.real(psi)

    # -- scikit-learn style API -----------------------------------------------
    def fit(self, X, y):
        """Learn the scaler, threshold and score orientation from training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        scores = self._decision_function(X_scaled)
        self.threshold = float(np.median(scores))

        # Orient the score so "above threshold" lines up with class 1. Without
        # this the interference sign is arbitrary and accuracy can invert.
        above = scores > self.threshold
        if above.sum() and (~above).sum():
            mean_above = y[above].mean()
            mean_below = y[~above].mean()
            self._sign = 1.0 if mean_above >= mean_below else -1.0
        else:
            self._sign = 1.0
        return self

    def decision_scores(self, X):
        """Return oriented, threshold-centred interference scores."""
        self._check_fitted()
        X_scaled = self.scaler.transform(np.asarray(X, dtype=float))
        return self._sign * (self._decision_function(X_scaled) - self.threshold)

    def predict(self, X):
        """Predict class labels (0 or 1)."""
        return (self.decision_scores(X) > 0).astype(int)

    def predict_proba(self, X):
        """Probability estimates via a sigmoid of the interference score."""
        scores = self.decision_scores(X)
        prob_class_1 = 1.0 / (1.0 + np.exp(-scores))
        return np.vstack([1.0 - prob_class_1, prob_class_1]).T

    def score(self, X, y):
        """Mean accuracy on ``(X, y)``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_fitted(self):
        if self.scaler is None:
            raise RuntimeError("classifier is not fitted yet; call fit() first")
