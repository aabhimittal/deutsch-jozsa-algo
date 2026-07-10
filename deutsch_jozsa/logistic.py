"""Quantum-enhanced logistic regression.

Standard logistic regression models ``P(y=1|x) = sigma(w . x + b)`` on the raw
features. Here we first expand the features through a *Deutsch-Jozsa inspired
feature map*: phases ``arctan(x)`` are pushed through cosine ("constant"-like)
and sine ("balanced"-like) interference layers, plus a few entanglement-style
cross terms. Logistic regression is then trained with plain gradient descent on
that richer, non-linear feature space.
"""

from __future__ import annotations

import numpy as np


class QuantumEnhancedLogisticRegression:
    """Logistic regression on a Deutsch-Jozsa inspired feature map.

    Parameters
    ----------
    quantum_reps : int
        Number of interference layers to stack in the feature map.
    random_state : int, optional
        Seed for reproducible weight initialisation.
    """

    def __init__(self, quantum_reps=2, random_state=None):
        self.quantum_reps = quantum_reps
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _deutsch_jozsa_feature_map(self, X):
        """Expand raw features into interference-based features."""
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]

        feature_blocks = [X]
        phases = np.arctan(X)

        for rep in range(self.quantum_reps):
            # cos -> "constant"-like patterns, sin -> "balanced"-like patterns.
            cos_features = np.cos(phases * (rep + 1))
            sin_features = np.sin(phases * (rep + 1))
            feature_blocks.extend([cos_features, sin_features])

            # Entanglement-inspired cross terms between neighbouring features.
            if n_features >= 2:
                cross_terms = cos_features[:, :-1] * sin_features[:, 1:]
                feature_blocks.append(cross_terms)

        return np.hstack(feature_blocks)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y, learning_rate=0.01, epochs=1000, verbose=False):
        """Train with batch gradient descent on the expanded features."""
        y = np.asarray(y, dtype=float)
        X_quantum = self._deutsch_jozsa_feature_map(X)
        n_samples, n_features = X_quantum.shape

        rng = np.random.default_rng(self.random_state)
        self.weights = rng.standard_normal(n_features) * 0.01
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(epochs):
            z = X_quantum @ self.weights + self.bias
            predictions = self._sigmoid(z)

            loss = -np.mean(
                y * np.log(predictions + 1e-15)
                + (1 - y) * np.log(1 - predictions + 1e-15)
            )
            self.loss_history.append(loss)

            dz = predictions - y
            dw = X_quantum.T @ dz / n_samples
            db = float(np.mean(dz))

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | loss: {loss:.6f}")
        return self

    def predict_proba(self, X):
        self._check_fitted()
        X_quantum = self._deutsch_jozsa_feature_map(X)
        proba = self._sigmoid(X_quantum @ self.weights + self.bias)
        return np.vstack([1.0 - proba, proba]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_fitted(self):
        if self.weights is None:
            raise RuntimeError("model is not fitted yet; call fit() first")
