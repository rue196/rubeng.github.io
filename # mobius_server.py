# mobius_server.py
import math
import random
import numpy as np
from numpy.fft import fft, ifft
import cmath

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)   # ≈ 2.362

# ---------- Möbius sieve (O(K)) ----------
def mobius_sieve(K):
    """
    Compute μ(n) for 1 ≤ n ≤ K.
    Returns list mu of length K+1 (mu[0] unused).
    """
    if K < 1:
        return [0] * (K + 1)
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

def count_squarefree_upto(K, mu):
    """Count numbers n ≤ K with μ(n) != 0."""
    return sum(1 for n in range(1, K+1) if mu[n] != 0)

def nth_squarefree(n, start=1):
    """
    Find the n‑th positive square‑free number.
    Uses binary search with Möbius sieve.
    """
    if n < 1:
        raise ValueError("n must be ≥ 1")
    # Upper bound: n * (π^2/6) + 10 (asymptotic density 6/π^2 ≈ 0.6079)
    limit = int(n * (PI*PI / 6) + 10)
    mu = mobius_sieve(limit)
    while count_squarefree_upto(limit, mu) < n:
        limit *= 2
        mu = mobius_sieve(limit)
    # Binary search
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi) // 2
        # We may need to recompute mu up to mid; but we already have mu up to limit.
        # Use the full mu (covers hi) and count up to mid.
        cnt = count_squarefree_upto(mid, mu)
        if cnt >= n:
            hi = mid
        else:
            lo = mid + 1
    return lo

# ---------- Supertrace and entropy (optional) ----------
def supertrace_from_coeffs(C):
    S = 0.0
    for idx, coeff in enumerate(C):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(coeff)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def invariant_scalar(C):
    S = supertrace_from_coeffs(C)
    H = entropy_from_supertrace(S, len(C))
    return abs(S) * math.exp(-H)

# ---------- Integral kernel (Toeplitz) ----------
def integral_kernel(K, alpha=ALPHA):
    norm = 1.0 - math.exp(-alpha * (PI + E))
    kernel = np.zeros(2*K - 1, dtype=float)
    for d in range(-(K-1), K):
        val = (1.0 - math.exp(-alpha * abs(d))) / norm
        kernel[d + (K-1)] = val
    return kernel

# ---------- FFT convolution ----------
def apply_convolution(signal, kernel):
    L = len(signal)
    N = 1 << (2*L - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - L), mode='constant')
    ker_pad = np.pad(kernel, (0, N - len(kernel)), mode='constant')
    conv = ifft(fft(sig_pad) * fft(ker_pad))[:L]
    return conv

# ---------- Server with Möbius compression ----------
class MobiusServer:
    def __init__(self, squarefree_index=50, seed=42):
        """
        K is chosen as the squarefree_index‑th square‑free number.
        """
        self.K = nth_squarefree(squarefree_index)
        self.squarefree_index = squarefree_index
        self.points, self.weights = self._generate_data(seed)
        self.client = None
        # Pre‑compute Möbius values up to K
        self.mu = mobius_sieve(self.K)
        # List of square‑free indices (1‑based)
        self.squarefree_indices = [i for i in range(1, self.K+1) if self.mu[i] != 0]
        # Compressed storage
        self.compressed = {}
        self.compressed_info = {}

    def _generate_data(self, seed):
        random.seed(seed)
        points = [(random.uniform(-5,5), random.uniform(-5,5)) for _ in range(self.K)]
        weights = [complex(random.gauss(0,1), random.gauss(0,1)) for _ in range(self.K)]
        return points, weights

    def set_client(self, client):
        self.client = client

    def get_algebraic_data(self):
        return self.points, self.weights

    def run_compression(self, sigma=None, method='mobius_supertrace'):
        """
        Compression pipeline:
          1. Sort points by angle.
          2. Client computes the integral kernel.
          3. Convolve weights with kernel (FFT).
          4. Keep only coefficients whose (1‑based) index is square‑free.
          5. Optionally, further reduce to top M by magnitude (if method='mobius_supertrace').
        """
        points_sorted, weights_sorted = self._sort_by_angle(self.points, self.weights)
        K = len(weights_sorted)

        if self.client is None:
            raise ValueError("Client not set.")
        kernel = self.client.compute_kernel(points_sorted, sigma)

        conv = apply_convolution(np.array(weights_sorted), kernel)

        # Filter by Möbius: keep only square‑free indices
        # Convert to 0‑based: index i (0‑based) corresponds to n = i+1
        squarefree_mask = [self.mu[i+1] != 0 for i in range(K)]
        filtered_indices = [i for i, ok in enumerate(squarefree_mask) if ok]
        filtered_values = conv[filtered_indices]

        if method == 'mobius_supertrace':
            # Compute supertrace on the full convolved signal (or on filtered?)
            # We'll use the full signal to determine M, then select top M among square‑free.
            S = supertrace_from_coeffs(conv)
            H = entropy_from_supertrace(S, K, ALPHA)
            m = invariant_scalar(conv)
            M = max(1, int(abs(S)))
            if M > len(filtered_indices):
                M = len(filtered_indices)
            # Among square‑free coefficients, keep top M by magnitude
            mags = np.abs(filtered_values)
            # Sort indices by magnitude descending
            sorted_order = np.argsort(mags)[::-1][:M]
            kept_indices = [filtered_indices[i] for i in sorted_order]
            kept_values = filtered_values[sorted_order]
        else:
            # Fallback: keep all square‑free coefficients (no further reduction)
            kept_indices = filtered_indices
            kept_values = filtered_values
            M = len(kept_indices)
            S = supertrace_from_coeffs(conv)
            H = entropy_from_supertrace(S, K, ALPHA)
            m = invariant_scalar(conv)

        # Store compressed data
        self.compressed = {int(i): v for i, v in zip(kept_indices, kept_values)}
        self.compressed_info = {
            'M': M,
            'S': S,
            'H': H,
            'm': m,
            'conv_full': conv,
            'kept_indices': kept_indices,
            'squarefree_indices': self.squarefree_indices
        }
        return self.compressed, points_sorted, weights_sorted, M, S, H, m

    def _sort_by_angle(self, points, weights):
        data = sorted(zip(points, weights), key=lambda p: cmath.phase(complex(p[0][0], p[0][1])))
        points_sorted = [p for p, w in data]
        weights_sorted = [w for p, w in data]
        return points_sorted, weights_sorted

    def reconstruct(self):
        """Reconstruct full convolved signal from compressed dict."""
        if not self.compressed:
            raise ValueError("No compressed data. Run compression first.")
        recon = np.zeros(self.K, dtype=complex)
        for i, val in self.compressed.items():
            recon[i] = val
        return recon

    def get_compression_ratio(self):
        return len(self.compressed) / self.K if self.K > 0 else 0.0

# ---------- Client (unchanged) ----------
class Client:
    def compute_kernel(self, points_sorted, sigma=None):
        K = len(points_sorted)
        return integral_kernel(K, ALPHA)

# ---------- Example ----------
if __name__ == "__main__":
    squarefree_index = 30          # 30th square‑free number = 46 (approx)
    server = MobiusServer(squarefree_index=squarefree_index, seed=42)
    client = Client()
    server.set_client(client)

    print(f"Server K = {server.K} (square‑free index {squarefree_index})")
    print(f"Number of square‑free indices ≤ K: {len(server.squarefree_indices)}")

    compressed, pts, wts, M, S, H, m = server.run_compression(method='mobius_supertrace')
    print(f"Compressed: kept {M} coefficients out of {server.K} (ratio {M/server.K:.3f})")
    print(f"Supertrace S = {S:.4f}, Entropy H = {H:.4f}, Mass m = {m:.4f}")

    recon = server.reconstruct()
    conv_full = server.compressed_info['conv_full']
    error = np.linalg.norm(conv_full - recon) / np.linalg.norm(conv_full)
    print(f"Reconstruction relative L2 error = {error:.4e}")

    print("First 5 compressed entries:", list(compressed.items())[:5])