#!/usr/bin/env python3
"""
unit_circle_basel_gate.py

Möbius unit circle gate with explicit square‑free filtering.
Includes Basel checksum error detection.
Only indices with μ(|i|) ≠ 0 are used; n=0 is skipped.
"""

import math
import numpy as np
import matplotlib.pyplot as pltz

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Möbius sieve ----------
def mobius_sieve(K):
    mu = [0] * (K + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def build_square_free_coeffs(K):
    """
    Build coefficient array where C_i = μ(|i|) if |i| is square‑free,
    and C_0 = 0 (since μ(0)=0). Returns a dict {i: coeff} and a list of indices.
    """
    mu = mobius_sieve(K)
    coeffs = {}
    for i in range(-K, K+1):
        if i == 0:
            continue   # skip n=0 (μ(0)=0)
        if mu[abs(i)] != 0:
            coeffs[i] = mu[abs(i)]
    return coeffs

# ---------- Unit circle gate ----------
class MobiusUnitCircleGate:
    def __init__(self, K):
        self.K = K
        self.coeffs = build_square_free_coeffs(K)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.period = 2 * PI * ALPHA   # fundamental period

    def zeta(self, t):
        """Evaluate ζ(t) = Σ_{i square‑free} μ(|i|) * exp(i * t * i / α)."""
        if len(self.indices) == 0:
            return 0.0 + 0.0j
        # Vectorized calculation
        phases = t * self.indices / ALPHA
        coeff_vals = np.array([self.coeffs[i] for i in self.indices])
        return np.sum(coeff_vals * np.exp(1j * phases))

    def power_spectrum(self, t):
        """Return |ζ(t)|²."""
        z = self.zeta(t)
        return np.abs(z)**2

    def integrate_power_spectrum(self, N=1000):
        """Numerically integrate |ζ(t)|² over one period."""
        t_vals = np.linspace(0, self.period, N)
        power = np.array([self.power_spectrum(t) for t in t_vals])
        integral = np.trapz(power, t_vals)
        average = integral / self.period
        return integral, average

    def basel_theoretical(self):
        """
        Theoretical average: 2 * (6/π²) * K.
        This is the expected average of |ζ(t)|² if the coefficients are
        μ(n) for square‑free n.
        """
        return 2 * (6 / (PI * PI)) * self.K

    def basel_error(self, N=1000):
        """Relative error between measured average and Basel theoretical."""
        _, avg = self.integrate_power_spectrum(N)
        theo = self.basel_theoretical()
        return abs(avg - theo) / theo if theo != 0 else 0.0

    def check_basel(self, tolerance=1e-2, N=1000):
        """Return True if relative error < tolerance."""
        err = self.basel_error(N)
        return err < tolerance, err

# ---------- Demo ----------
def demo():
    K = 20
    gate = MobiusUnitCircleGate(K)

    # Print coefficients (only non-zero)
    print(f"K = {K}")
    print(f"Number of square‑free indices: {len(gate.indices)}")
    print("Coefficients (i -> C_i):")
    for i in gate.indices:
        print(f"  C_{i} = {gate.coeffs[i]}")

    # Integrate power spectrum
    integral, avg = gate.integrate_power_spectrum(N=2000)
    print(f"\nIntegral over [0, {gate.period:.3f}] : {integral:.6f}")
    print(f"Average |ζ(t)|² : {avg:.6f}")

    # Basel theoretical
    theo = gate.basel_theoretical()
    print(f"  Theoretical average (2*(6/π²)*K): {theo:.6f}")

    # Basel error check
    ok, err = gate.check_basel(tolerance=0.02, N=2000)
    print(f"  Basel error: {err:.6f}  -> {'OK' if ok else 'FAIL'}")

    # Plot the power spectrum
    t_vals = np.linspace(0, gate.period, 500)
    power_vals = [gate.power_spectrum(t) for t in t_vals]
    plt.figure(figsize=(10, 4))
    plt.plot(t_vals, power_vals)
    plt.axhline(y=avg, color='r', linestyle='--', label=f'Average = {avg:.4f}')
    plt.axhline(y=theo, color='g', linestyle=':', label=f'Basel = {theo:.4f}')
    plt.xlabel('t')
    plt.ylabel('|ζ(t)|²')
    plt.title(f'Power spectrum on the unit circle (K={K})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    demo()