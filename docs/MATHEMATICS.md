# The Mathematics of Deutsch-Jozsa

A step-by-step derivation of why the algorithm works, plus a primer on every
symbol involved. Read this alongside `deutsch_jozsa/quantum_dj.py`.

## 1. Notation primer

| Symbol | Name | Meaning |
| --- | --- | --- |
| `|ψ⟩` | ket | A quantum state (a column vector) |
| `⟨ψ|` | bra | The conjugate transpose of `|ψ⟩` (a row vector) |
| `⟨ψ|φ⟩` | bra-ket | Inner product / overlap between two states |
| `|0⟩`, `|1⟩` | computational basis | `[1,0]ᵀ` and `[0,1]ᵀ` |
| `⊗` | tensor product | Combines subsystems: `|0⟩⊗|1⟩ = |01⟩` |
| `∑` | summation | Sum over the indicated index |
| `H` | Hadamard gate | Creates / removes superposition |
| `Uf` | oracle | Unitary encoding the function `f` |
| `⊕` | XOR | Addition modulo 2 |
| `x·y` | bitwise dot product | `x₀y₀ ⊕ x₁y₁ ⊕ … ⊕ x_{n-1}y_{n-1}` |

A single qubit is `|ψ⟩ = α|0⟩ + β|1⟩` with complex amplitudes constrained by
`|α|² + |β|² = 1`. Measurement yields `0` with probability `|α|²` and `1` with
probability `|β|²`.

The Hadamard gate is

```
H = (1/√2) [ 1  1 ]      H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩
           [ 1 -1 ]      H|1⟩ = (|0⟩ - |1⟩)/√2 = |-⟩
```

## 2. The problem

We are promised that `f: {0,1}ⁿ → {0,1}` is either

* **constant** — `f(x)` is the same for every `x`, or
* **balanced** — `f(x) = 0` for exactly half the inputs and `1` for the rest.

Goal: decide which, with as few oracle queries as possible.

* Classical worst case: `2ⁿ⁻¹ + 1` queries (you may see half the inputs all
  agree before a disagreement appears).
* Deutsch-Jozsa: **exactly one** query.

## 3. The circuit, state by state

```
|0⟩^⊗n ──H^⊗n──┤      ├──H^⊗n──┤M├
               │  Uf  │
|1⟩    ──H─────┤      ├──────────
```

**Step 0 — initialise.** `|ψ₀⟩ = |0⟩^⊗n ⊗ |1⟩`.

**Step 1 — Hadamard all qubits.**

```
|ψ₁⟩ = (1/√2ⁿ) ∑ₓ |x⟩ ⊗ |-⟩
```

The input register is now a uniform superposition over all `2ⁿ` inputs, and the
ancilla sits in `|-⟩`.

**Step 2 — the oracle and phase kickback.** The oracle acts as
`Uf|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩`. Applied to `|x⟩|-⟩`:

```
Uf|x⟩|-⟩ = |x⟩ (|f(x)⟩ - |1 ⊕ f(x)⟩)/√2
         = (-1)^{f(x)} |x⟩|-⟩
```

* If `f(x) = 0`: the state is unchanged, sign `+1`.
* If `f(x) = 1`: the two terms swap, producing an overall `-1`.

So the function value becomes a **relative phase** on the input register:

```
|ψ₂⟩ = (1/√2ⁿ) ∑ₓ (-1)^{f(x)} |x⟩ ⊗ |-⟩
```

**Step 3 — Hadamard the input register again.** Using
`H^⊗n |x⟩ = (1/√2ⁿ) ∑_y (-1)^{x·y} |y⟩`,

```
|ψ₃⟩ = (1/2ⁿ) ∑_y [ ∑ₓ (-1)^{f(x) + x·y} ] |y⟩ ⊗ |-⟩
```

The amplitude of a given output `|y⟩` is `α_y = (1/2ⁿ) ∑ₓ (-1)^{f(x) + x·y}`.

**Step 4 — read the `|0…0⟩` amplitude.** For `y = 0`, `x·y = 0` for all `x`, so

```
α₀ = (1/2ⁿ) ∑ₓ (-1)^{f(x)}
```

* **Constant `f`:** every term has the same sign, so `α₀ = ±1`. Probability of
  measuring `|0…0⟩` is `|α₀|² = 1`.
* **Balanced `f`:** exactly half the terms are `+1` and half are `-1`, so they
  cancel and `α₀ = 0`. We can **never** measure `|0…0⟩`.

**Step 5 — measure.**

* All zeros → the function is **constant**.
* Anything else → the function is **balanced**.

One query settles it. ∎

## 4. Why each trick matters

* **Ancilla in `|-⟩`.** Turns the oracle's bit output into a phase, so it acts
  on the input register without needing to be measured (phase kickback).
* **Two Hadamard layers.** The first spreads amplitude across all inputs
  (superposition); the second is a discrete Fourier-style transform that turns
  the phase pattern into an interference pattern. The `|0…0⟩` amplitude is
  exactly the *average* of `(-1)^{f(x)}` — the DC / zero-frequency component.
* **Global, not local.** We never learn individual values `f(x)`; we read a
  single global property (constant vs balanced) that lives in the interference.

## 5. The bridge to this repository

| Quantum idea | Classical analogue in code |
| --- | --- |
| Phase encoding `(-1)^{f(x)}` | `arctan(x)` → phase, `exp(i·φ)` amplitudes (`classifier.py`) |
| Interference of amplitudes | Real part of the summed amplitude as a decision score |
| Constant vs balanced read-out | Threshold on the global interference value |
| Cosine/sine interference layers | Deutsch-Jozsa feature map (`logistic.py`) |
| One-query global verification | Key recovery / tamper check (`qkd.py`) |

The quantum modules run the real thing on a simulator; the NumPy modules borrow
the *interference-as-decision* intuition for classical machine learning.
