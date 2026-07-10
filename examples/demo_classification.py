"""Compare the quantum-inspired models against classical logistic regression."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from deutsch_jozsa import (
    QuantumEnhancedLogisticRegression,
    QuantumInspiredClassifier,
)


def main():
    print("=" * 70)
    print("QUANTUM-INSPIRED vs CLASSICAL CLASSIFICATION")
    print("=" * 70)

    X, y = make_classification(
        n_samples=600,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    results = []

    start = time.time()
    qic = QuantumInspiredClassifier(n_features=X.shape[1]).fit(X_train, y_train)
    results.append(("Quantum-Inspired Classifier", qic.score(X_test, y_test),
                    time.time() - start))

    start = time.time()
    qelr = QuantumEnhancedLogisticRegression(quantum_reps=2, random_state=42)
    qelr.fit(X_train, y_train, epochs=500)
    results.append(("Quantum-Enhanced Logistic Reg.", qelr.score(X_test, y_test),
                    time.time() - start))

    start = time.time()
    clf = LogisticRegression(max_iter=500).fit(X_train, y_train)
    results.append(("Classical Logistic Reg. (base)",
                    float(np.mean(clf.predict(X_test) == y_test)),
                    time.time() - start))

    print(f"\n{'Method':<34}{'Accuracy':<12}{'Time (s)':<10}")
    print("-" * 56)
    for name, acc, secs in results:
        print(f"{name:<34}{acc:<12.4f}{secs:<10.4f}")


if __name__ == "__main__":
    main()
