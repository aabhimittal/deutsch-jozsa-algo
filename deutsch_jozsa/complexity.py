"""Query-complexity comparison: classical vs Deutsch-Jozsa."""

from __future__ import annotations


def complexity_table(max_qubits=15):
    """Return per-``n`` query complexity as a list of dictionaries.

    Classical (worst case): ``2^(n-1) + 1`` queries.
    Quantum (Deutsch-Jozsa): always ``1`` query.
    """
    rows = []
    for n in range(1, max_qubits + 1):
        classical = 2 ** (n - 1) + 1
        quantum = 1
        rows.append(
            {
                "qubits": n,
                "classical_worst_case": classical,
                "quantum": quantum,
                "speedup": classical / quantum,
            }
        )
    return rows


def print_complexity_comparison(max_qubits=15):
    """Pretty-print the classical vs quantum query complexity table."""
    print("=" * 70)
    print("COMPLEXITY: CLASSICAL vs QUANTUM (Deutsch-Jozsa)")
    print("=" * 70)
    print(f"{'Qubits':<10}{'Classical (worst)':<22}{'Quantum':<12}{'Speedup':<12}")
    print("-" * 70)
    for row in complexity_table(max_qubits):
        print(
            f"{row['qubits']:<10}{row['classical_worst_case']:<22}"
            f"{row['quantum']:<12}{row['speedup']:.0f}x"
        )
    print("\nClassical: O(2^n) worst case | Quantum: O(1) -- exactly one query.")
