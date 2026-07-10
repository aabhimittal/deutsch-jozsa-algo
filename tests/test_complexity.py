from deutsch_jozsa import complexity_table


def test_table_length_and_values():
    rows = complexity_table(max_qubits=5)
    assert len(rows) == 5
    first = rows[0]
    assert first["qubits"] == 1
    assert first["classical_worst_case"] == 2 ** 0 + 1  # == 2
    assert first["quantum"] == 1


def test_speedup_is_exponential():
    rows = complexity_table(max_qubits=10)
    last = rows[-1]
    assert last["classical_worst_case"] == 2 ** 9 + 1
    assert last["speedup"] == last["classical_worst_case"]
