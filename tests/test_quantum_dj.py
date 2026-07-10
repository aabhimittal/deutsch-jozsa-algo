import pytest

qiskit = pytest.importorskip("qiskit")

from deutsch_jozsa.quantum_dj import (  # noqa: E402
    create_deutsch_jozsa_circuit,
    run_deutsch_jozsa,
)


def test_constant_oracle_detected():
    result = run_deutsch_jozsa(n_qubits=3, oracle_type="constant",
                               shots=512, verbose=False)
    assert result["detected"] == "constant"
    assert result["correct"]


def test_balanced_oracle_detected():
    result = run_deutsch_jozsa(n_qubits=3, oracle_type="balanced",
                               secret_string="101", shots=512, verbose=False)
    assert result["detected"] == "balanced"
    assert result["correct"]


def test_circuit_dimensions():
    qc, secret = create_deutsch_jozsa_circuit(4, "balanced", secret_string="1010")
    assert qc.num_qubits == 5  # 4 input + 1 ancilla
    assert qc.num_clbits == 4
    assert secret == "1010"


def test_invalid_secret_length():
    with pytest.raises(ValueError):
        create_deutsch_jozsa_circuit(3, "balanced", secret_string="10")


def test_zero_secret_rejected():
    with pytest.raises(ValueError):
        create_deutsch_jozsa_circuit(3, "balanced", secret_string="000")
