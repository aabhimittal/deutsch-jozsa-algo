import numpy as np
from sklearn.datasets import make_classification

from deutsch_jozsa import QuantumEnhancedLogisticRegression


def _data():
    return make_classification(
        n_samples=300, n_features=6, n_informative=4, n_redundant=1,
        n_classes=2, random_state=1,
    )


def test_feature_map_expands_features():
    X, _ = _data()
    model = QuantumEnhancedLogisticRegression(quantum_reps=2)
    mapped = model._deutsch_jozsa_feature_map(X)
    assert mapped.shape[0] == X.shape[0]
    assert mapped.shape[1] > X.shape[1]


def test_training_reduces_loss_and_learns():
    X, y = _data()
    model = QuantumEnhancedLogisticRegression(quantum_reps=2, random_state=0)
    model.fit(X, y, epochs=300)
    assert model.loss_history[-1] < model.loss_history[0]
    # Should comfortably beat chance on the training set.
    assert model.score(X, y) > 0.7


def test_predict_proba_valid():
    X, y = _data()
    model = QuantumEnhancedLogisticRegression(random_state=0).fit(X, y, epochs=100)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_requires_fit():
    model = QuantumEnhancedLogisticRegression()
    try:
        model.predict(np.zeros((2, 4)))
    except RuntimeError:
        return
    raise AssertionError("predict before fit should raise RuntimeError")
