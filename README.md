# Deutsch-Jozsa: From Quantum Oracles to Classical Intelligence

A small, well-tested project that starts from the **Deutsch-Jozsa algorithm**
— the first algorithm to prove an exponential quantum speedup over classical
computation — and follows a single idea, *global interference*, into three
downstream domains:

1. **Quantum computing** — the pure Deutsch-Jozsa algorithm on a Qiskit simulator.
2. **Machine learning** — a quantum-*inspired* classifier and a quantum-enhanced
   logistic regression that run on plain NumPy.
3. **Security** — Deutsch-Jozsa based key verification and tamper detection.

> The classical detective needs `2ⁿ⁻¹ + 1` questions in the worst case. The
> quantum detective needs exactly one. This repository shows *why*, and what
> that idea buys you elsewhere.

## The problem in one paragraph

You are given a function `f: {0,1}ⁿ → {0,1}` promised to be either **constant**
(same output everywhere) or **balanced** (0 on half the inputs, 1 on the other
half). Deutsch-Jozsa decides which with a **single** oracle query by putting all
`2ⁿ` inputs into superposition, using *phase kickback* to stamp each function
value onto a quantum phase, and then interfering those phases so the answer
concentrates on a single measurement outcome. Full derivation in
[`docs/MATHEMATICS.md`](docs/MATHEMATICS.md).

## Installation

```bash
git clone https://github.com/aabhimittal/deutsch-jozsa-algo.git
cd deutsch-jozsa-algo
pip install -r requirements.txt      # numpy, scikit-learn, qiskit, qiskit-aer
# or install the package itself:
pip install -e ".[quantum,dev]"
```

The NumPy-only parts (classifier, logistic regression, complexity table) work
without Qiskit; the quantum modules need the `quantum` extra.

## Quick start

```python
from deutsch_jozsa.quantum_dj import run_deutsch_jozsa

# One query tells constant from balanced:
run_deutsch_jozsa(n_qubits=4, oracle_type="balanced")
```

```python
from deutsch_jozsa import QuantumInspiredClassifier, QuantumEnhancedLogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=400, n_features=8, random_state=0)
clf = QuantumInspiredClassifier().fit(X, y)
print("accuracy:", clf.score(X, y))
```

## Run the demos

```bash
python examples/demo_quantum_dj.py       # pure quantum algorithm
python examples/demo_classification.py   # quantum-inspired vs classical ML
python examples/demo_qkd.py              # key verification + tamper detection
python examples/run_all.py               # everything, end to end
```

## Project layout

```
deutsch_jozsa/
  quantum_dj.py     Pure Deutsch-Jozsa circuit + runner (Qiskit)
  classifier.py     QuantumInspiredClassifier (interference-based, NumPy)
  logistic.py       QuantumEnhancedLogisticRegression (DJ feature map, NumPy)
  qkd.py            QuantumKeyDistribution — DJ-based key verification (Qiskit)
  complexity.py     Classical vs quantum query-complexity comparison
examples/           Runnable demonstrations
tests/              pytest suite (Qiskit tests auto-skip if unavailable)
docs/MATHEMATICS.md Symbol primer + full step-by-step derivation
```

## The connecting idea

| Component | Deutsch-Jozsa concept it borrows |
| --- | --- |
| `QuantumInspiredClassifier` | Encode features as phases, decide on the **global interference** value rather than individual features |
| `QuantumEnhancedLogisticRegression` | A cosine/sine "interference" **feature map** (constant-like vs balanced-like patterns) before linear training |
| `QuantumKeyDistribution` | One-query **global read-out** recovers the key; any tampered bit changes the recovered string |

## Complexity

| Qubits `n` | Classical (worst case) | Quantum | Speedup |
| --- | --- | --- | --- |
| 4 | 9 | 1 | 9× |
| 8 | 129 | 1 | 129× |
| 12 | 2049 | 1 | 2049× |

Classical: `O(2ⁿ)` worst case. Quantum: `O(1)` — always one query.

## Tests

```bash
pytest -q
```

Qiskit-dependent tests skip automatically when Qiskit is not installed, so the
classical suite always runs.

## A note on scope

The quantum modules run the genuine algorithm on a simulator. The machine-learning
and QKD modules are **teaching models**: they carry the Deutsch-Jozsa intuition
into classical code, not a claim of quantum advantage on real hardware. They are
meant to make the connection between quantum interference, feature engineering,
and tamper detection concrete and runnable.

## License

MIT — see [`LICENSE`](LICENSE).
