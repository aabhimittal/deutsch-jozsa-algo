"""Quantum key verification built on the Deutsch-Jozsa circuit.

A bit string ``s`` is encoded as an oracle ``f(x) = s . x (mod 2)`` using CNOTs.
Running the Deutsch-Jozsa / Bernstein-Vazirani circuit on that oracle makes the
input register collapse onto exactly ``s`` -- one query recovers the whole
string. We use this as a tamper check: if an eavesdropper flips bits of the key,
the recovered string no longer matches the expected one and the integrity score
drops below 1.0.

This is a teaching model of the intuition behind quantum key distribution, not a
production QKD stack: it shows how a global quantum read-out exposes tampering
that a classical bit-by-bit copy could hide.
"""

from __future__ import annotations

import numpy as np
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)
from qiskit_aer import AerSimulator


class QuantumKeyDistribution:
    """Distribute and verify a key using a Deutsch-Jozsa style oracle."""

    def __init__(self, n_bits=8, seed=None):
        self.n_bits = n_bits
        self.simulator = AerSimulator()
        self._rng = np.random.default_rng(seed)

    def generate_key(self, length=None):
        """Return a random bit string of the given length (defaults to ``n_bits``)."""
        length = self.n_bits if length is None else length
        return "".join(self._rng.choice(["0", "1"]) for _ in range(length))

    def _build_verification_circuit(self, key_string):
        n = len(key_string)
        qr = QuantumRegister(n + 1, "q")
        cr = ClassicalRegister(n, "c")
        qc = QuantumCircuit(qr, cr)

        # Prepare |0...0>|1> then Hadamard everything.
        qc.x(qr[n])
        for i in range(n + 1):
            qc.h(qr[i])

        # Oracle f(x) = key . x : one CNOT per set bit of the key.
        for i, bit in enumerate(key_string):
            if bit == "1":
                qc.cx(qr[i], qr[n])

        # Interference layer on the input register, then measure.
        for i in range(n):
            qc.h(qr[i])
        qc.measure(qr[:n], cr)
        return qc

    def verify_key_integrity(self, key_string, expected_key=None, shots=1000):
        """Recover the key with one query and compare against ``expected_key``.

        Returns
        -------
        integrity_score : float
            Fraction of shots whose recovered string matches ``expected_key``
            (defaults to ``key_string`` itself). 1.0 means untouched.
        counts : dict
            Raw measurement histogram.
        """
        expected = key_string if expected_key is None else expected_key
        qc = self._build_verification_circuit(key_string)
        transpiled = transpile(qc, self.simulator)
        counts = transpiled_counts = self.simulator.run(transpiled, shots=shots).result().get_counts()

        # Qiskit returns little-endian strings; reverse to compare against the key.
        matches = sum(
            count for state, count in transpiled_counts.items()
            if state[::-1] == expected
        )
        integrity_score = matches / shots
        return integrity_score, counts

    def simulate_eavesdropping(self, key_string, tampering_rate=0.3):
        """Return a copy of ``key_string`` with a fraction of its bits flipped."""
        tampered = list(key_string)
        n_flip = max(1, int(round(len(key_string) * tampering_rate)))
        flip_indices = self._rng.choice(len(key_string), n_flip, replace=False)
        for idx in flip_indices:
            tampered[idx] = "1" if tampered[idx] == "0" else "0"
        return "".join(tampered)
