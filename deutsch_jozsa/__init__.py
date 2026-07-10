"""Deutsch-Jozsa: from quantum oracles to classical intelligence.

A small teaching library that connects the Deutsch-Jozsa algorithm to three
downstream domains:

* ``quantum_dj``  -- the pure quantum algorithm (Qiskit).
* ``classifier``  -- a quantum-inspired binary classifier (NumPy).
* ``logistic``    -- quantum-enhanced logistic regression (NumPy).
* ``qkd``         -- Deutsch-Jozsa based key verification (Qiskit).
* ``complexity``  -- classical vs quantum query-complexity comparison.
"""

from .classifier import QuantumInspiredClassifier
from .complexity import complexity_table, print_complexity_comparison
from .logistic import QuantumEnhancedLogisticRegression

__all__ = [
    "QuantumInspiredClassifier",
    "QuantumEnhancedLogisticRegression",
    "complexity_table",
    "print_complexity_comparison",
    "create_deutsch_jozsa_circuit",
    "run_deutsch_jozsa",
    "QuantumKeyDistribution",
]

__version__ = "0.1.0"


def __getattr__(name):
    # Lazily import Qiskit-backed pieces so the NumPy-only parts work even if
    # Qiskit is not installed.
    if name in ("create_deutsch_jozsa_circuit", "run_deutsch_jozsa"):
        from . import quantum_dj

        return getattr(quantum_dj, name)
    if name == "QuantumKeyDistribution":
        from .qkd import QuantumKeyDistribution

        return QuantumKeyDistribution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
