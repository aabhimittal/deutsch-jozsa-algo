"""Run every demonstration end to end."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from deutsch_jozsa.complexity import print_complexity_comparison

from demo_classification import main as classification_main
from demo_qkd import main as qkd_main
from demo_quantum_dj import main as quantum_dj_main


def main():
    print("\n" + "=" * 70)
    print("DEUTSCH-JOZSA: FROM QUANTUM ORACLES TO CLASSICAL INTELLIGENCE")
    print("=" * 70)

    print("\n[1/4] Pure quantum algorithm")
    quantum_dj_main()

    print("\n[2/4] Classification benchmark")
    classification_main()

    print("\n[3/4] Quantum key distribution")
    qkd_main()

    print("\n[4/4] Complexity comparison")
    print_complexity_comparison(max_qubits=12)

    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
