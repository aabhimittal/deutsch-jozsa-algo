import numpy as np
from sklearn.datasets import make_classification

from deutsch_jozsa import QuantumInspiredClassifier


def _data():
    return make_classification(
        n_samples=200, n_features=4, n_informative=3, n_redundant=1,
        n_classes=2, random_state=0,
    )


def test_fit_predict_shapes():
    X, y = _data()
    clf = QuantumInspiredClassifier(n_features=4).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_proba_valid():
    X, y = _data()
    clf = QuantumInspiredClassifier(n_features=4).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_orientation_beats_chance():
    X, y = _data()
    clf = QuantumInspiredClassifier(n_features=4).fit(X, y)
    # Score orientation must ensure we do at least as well as chance.
    assert clf.score(X, y) >= 0.5


def test_requires_fit():
    clf = QuantumInspiredClassifier()
    try:
        clf.predict(np.zeros((2, 4)))
    except RuntimeError:
        return
    raise AssertionError("predict before fit should raise RuntimeError")
