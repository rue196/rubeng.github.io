import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)   # unused here, but kept for consistency

# ---------- Linear sieve for Möbius (O(K)) ----------
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

# ---------- Harmonic numbers (O(K)) ----------
def harmonic_numbers(K):
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n-1] + 1.0 / n
    return H

# ---------- Dirichlet convolution (μ * H) in O(K log K) ----------
def mu_convolution_H(K):
    mu = mobius_sieve(K)
    H = harmonic_numbers(K)

    F = np.zeros(K + 1, dtype=float)   # F[n] = (μ * H)(n)

    # For each d, add μ(d) * H(m) to F[m*d]
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        for m in range(1, K // d + 1):
            F[d * m] += mu[d] * H[m]

    return F, mu, H

# ---------- Main ----------
def main():
    K = 200
    print(f"Computing (μ * H)(n) for n = 1..{K} in O(K log K) time...")
    F, mu, H = mu_convolution_H(K)

    # Print some values
    print("\nFirst 20 values of F(n) = (μ * H)(n):")
    for n in range(1, min(21, K+1)):
        print(f"F({n}) = {F[n]:.6f}")

    # Summatory function: S(K) = Σ_{n=1}^K F(n)
    S = np.sum(F[1:])
    print(f"\nSummatory S({K}) = Σ_{1}^{K} F(n) = {S:.6f}")

    # Also compute the summatory of μ(n) * H_{⌊K/n⌋} for comparison
    S2 = 0.0
    for n in range(1, K+1):
        S2 += mu[n] * H[K // n]
    print(f"Alternative sum Σ μ(n) H_{K/n} = {S2:.6f}")

    # Check identity: Σ_{d|n} μ(d) = 1 if n=1 else 0, but with H it's different.
    # We can verify that F[1] = 1.
    print(f"F(1) = {F[1]:.6f} (should be 1)")

if __name__ == "__main__":
    main()