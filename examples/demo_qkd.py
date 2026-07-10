"""Demonstrate Deutsch-Jozsa based key verification and tamper detection."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deutsch_jozsa import QuantumKeyDistribution


def main():
    print("=" * 70)
    print("QUANTUM KEY DISTRIBUTION WITH DEUTSCH-JOZSA VERIFICATION")
    print("=" * 70)

    qkd = QuantumKeyDistribution(n_bits=6, seed=42)
    alice_key = qkd.generate_key()
    print(f"\nAlice's original key: {alice_key}")

    print("\n--- Scenario 1: secure channel (no eavesdropping) ---")
    score, _ = qkd.verify_key_integrity(alice_key, expected_key=alice_key)
    print(f"Integrity score: {score:.1%}  -> "
          f"{'SECURE' if score > 0.95 else 'COMPROMISED'}")

    print("\n--- Scenario 2: Eve tampers with the key ---")
    tampered = qkd.simulate_eavesdropping(alice_key, tampering_rate=0.34)
    flipped = sum(a != b for a, b in zip(alice_key, tampered))
    print(f"Eve's tampered key:   {tampered}  ({flipped} bits changed)")
    score, _ = qkd.verify_key_integrity(tampered, expected_key=alice_key)
    print(f"Integrity score: {score:.1%}  -> "
          f"{'SECURE' if score > 0.95 else 'COMPROMISED'}")

    print("\nDeutsch-Jozsa recovers the whole key in one query, so any flipped")
    print("bit makes the recovered string diverge from the expected one.")


if __name__ == "__main__":
    main()
