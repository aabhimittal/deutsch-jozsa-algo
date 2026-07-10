"""Pure quantum implementation of the Deutsch-Jozsa algorithm.

The Deutsch-Jozsa algorithm decides whether a promised function
``f: {0, 1}^n -> {0, 1}`` is *constant* (same output for every input) or
*balanced* (output 0 for exactly half the inputs and 1 for the other half).

Classically this needs ``2^(n-1) + 1`` queries in the worst case. The quantum
algorithm needs exactly **one** oracle query, regardless of ``n``.

Circuit outline::

    |0>^n  --H^n--|      |--H^n--|M|
                  |  Uf  |
    |1>    --H----|      |----------

The ancilla qubit prepared in the ``|->`` state turns the oracle's output into
a relative phase (``phase kickback``). A final layer of Hadamard gates converts
those phases into an interference pattern: if ``f`` is constant every amplitude
concentrates on ``|0...0>``; if ``f`` is balanced that amplitude cancels to
zero and we always measure a non-zero string.
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


def create_deutsch_jozsa_circuit(n_qubits, oracle_type="balanced", secret_string=None):
    """Build a complete Deutsch-Jozsa circuit.

    Parameters
    ----------
    n_qubits : int
        Number of input qubits (the ancilla is added automatically).
    oracle_type : {"constant", "balanced"}
        Which family of oracle to construct.
    secret_string : str, optional
        For a balanced oracle, a length-``n_qubits`` bit string selecting which
        input qubits are wired into the oracle. A random non-zero string is
        generated when omitted. Ignored for a constant oracle.

    Returns
    -------
    circuit : qiskit.QuantumCircuit
        The assembled circuit including measurement of the input register.
    secret_string : str or None
        The bit string actually used (``None`` for a constant oracle).
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if oracle_type not in ("constant", "balanced"):
        raise ValueError("oracle_type must be 'constant' or 'balanced'")

    qr = QuantumRegister(n_qubits + 1, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr)

    # Step 1: put the ancilla into |1> so that H makes it |->.
    qc.x(qr[n_qubits])
    qc.barrier()

    # Step 2: Hadamard everything -> uniform superposition + phase-kickback ancilla.
    for i in range(n_qubits + 1):
        qc.h(qr[i])
    qc.barrier()

    # Step 3: the oracle U_f.
    used_secret = None
    if oracle_type == "constant":
        # Constant-0 does nothing; constant-1 flips the ancilla (global phase only).
        if np.random.random() > 0.5:
            qc.x(qr[n_qubits])
    else:  # balanced
        if secret_string is None:
            # A non-zero string guarantees a genuinely balanced function.
            while True:
                secret_string = "".join(np.random.choice(["0", "1"]) for _ in range(n_qubits))
                if "1" in secret_string:
                    break
        else:
            _validate_secret(secret_string, n_qubits)
            if "1" not in secret_string:
                raise ValueError("a balanced oracle needs a non-zero secret_string")
        used_secret = secret_string
        for i, bit in enumerate(secret_string):
            if bit == "1":
                qc.cx(qr[i], qr[n_qubits])
    qc.barrier()

    # Step 4: interference layer on the input register.
    for i in range(n_qubits):
        qc.h(qr[i])
    qc.barrier()

    # Step 5: measure the input register (little-endian in the returned string).
    qc.measure(qr[:n_qubits], cr)

    return qc, used_secret


def run_deutsch_jozsa(n_qubits=3, oracle_type="balanced", secret_string=None,
                      shots=1024, verbose=True):
    """Execute a Deutsch-Jozsa circuit and classify the oracle.

    Returns
    -------
    result : dict
        Keys: ``counts`` (measurement histogram), ``detected`` ("constant" or
        "balanced"), ``correct`` (bool), ``secret_string`` and ``circuit``.
    """
    qc, used_secret = create_deutsch_jozsa_circuit(n_qubits, oracle_type, secret_string)

    simulator = AerSimulator()
    transpiled = transpile(qc, simulator)
    counts = simulator.run(transpiled, shots=shots).result().get_counts()

    all_zeros = "0" * n_qubits
    prob_zero = counts.get(all_zeros, 0) / shots
    detected = "constant" if prob_zero > 0.95 else "balanced"
    correct = detected == oracle_type

    if verbose:
        print(f"\n{'=' * 60}")
        print("Deutsch-Jozsa Algorithm")
        print(f"{'=' * 60}")
        print(f"Input qubits : {n_qubits}")
        print(f"Oracle type  : {oracle_type}")
        if used_secret:
            print(f"Secret string: {used_secret}")
        print(f"\nMeasurement results ({shots} shots):")
        for state, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  |{state}>: {count:5d} ({count / shots * 100:5.1f}%)")
        verdict = "correct" if correct else "WRONG"
        print(f"\nDetected: {detected.upper()}  ->  {verdict}")

    return {
        "counts": counts,
        "detected": detected,
        "correct": correct,
        "secret_string": used_secret,
        "circuit": qc,
    }


def _validate_secret(secret_string, n_qubits):
    if len(secret_string) != n_qubits:
        raise ValueError(
            f"secret_string length {len(secret_string)} != n_qubits {n_qubits}"
        )
    if any(c not in "01" for c in secret_string):
        raise ValueError("secret_string must contain only '0' and '1'")
