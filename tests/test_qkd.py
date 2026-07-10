import pytest

pytest.importorskip("qiskit")

from deutsch_jozsa import QuantumKeyDistribution  # noqa: E402


def test_untampered_key_is_secure():
    qkd = QuantumKeyDistribution(n_bits=5, seed=7)
    key = qkd.generate_key()
    score, _ = qkd.verify_key_integrity(key, expected_key=key, shots=512)
    assert score > 0.95


def test_tampered_key_is_detected():
    qkd = QuantumKeyDistribution(n_bits=6, seed=3)
    key = qkd.generate_key()
    tampered = qkd.simulate_eavesdropping(key, tampering_rate=0.5)
    assert tampered != key
    score, _ = qkd.verify_key_integrity(tampered, expected_key=key, shots=512)
    assert score < 0.95


def test_generate_key_length():
    qkd = QuantumKeyDistribution(n_bits=8, seed=1)
    assert len(qkd.generate_key()) == 8
    assert len(qkd.generate_key(4)) == 4
