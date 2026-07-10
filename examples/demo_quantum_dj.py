"""Run the pure quantum Deutsch-Jozsa algorithm on constant and balanced oracles."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deutsch_jozsa.quantum_dj import run_deutsch_jozsa


def main():
    print("\n" + "#" * 70)
    print("DEUTSCH-JOZSA ALGORITHM DEMONSTRATION")
    print("#" * 70)

    print("\n>>> TEST 1: constant oracle")
    run_deutsch_jozsa(n_qubits=4, oracle_type="constant", shots=1000)

    print("\n>>> TEST 2: balanced oracle")
    run_deutsch_jozsa(n_qubits=4, oracle_type="balanced", shots=1000)


if __name__ == "__main__":
    main()
