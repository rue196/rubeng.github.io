import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from scipy.ndimage import zoom
import cmath

# ------------------------------------------------------------
# 1. Generate a synthetic image from M(x,y) = x^(n-1) + y^(n-1)
# ------------------------------------------------------------
def generate_matrix_image(size=128, n=1, scale=2.0):
    """
    Create a 2D array where pixel (i,j) = (x^(n-1) + y^(n-1))
    with x, y scaled to the range [-scale, scale].
    n = 1, 5, 15, etc.
    """
    x = np.linspace(-scale, scale, size)
    y = np.linspace(-scale, scale, size)
    X, Y = np.meshgrid(x, y)
    # Avoid negative bases for fractional powers: take absolute value, or use integer exponents.
    # For integer n, we can use X**(n-1) + Y**(n-1) safely (negative bases yield negative values).
    # For n=1, we get X^0 + Y^0 = 1+1 = 2 (constant).
    # For better visual, we can shift and normalize.
    if n == 1:
        img = np.ones((size, size)) * 2.0   # constant 2
    else:
        img = np.abs(X)**(n-1) + np.abs(Y)**(n-1)   # use abs to avoid complex
    # Normalize to [0,1]
    img = img - img.min()
    img = img / img.max()
    return img

def load_image(n=1, size=128):
    """
    Load (or generate) image. Here we generate a matrix‑trace image.
    """
    return generate_matrix_image(size, n)

# ------------------------------------------------------------
# 2. Prepare data: flatten image, pack coordinates as complex
# ------------------------------------------------------------
def flatten_image(img):
    H, W = img.shape
    K = H * W
    coords = np.array([(i, j) for i in range(H) for j in range(W)])
    z = coords[:,0] + 1j * coords[:,1]   # complex coordinates
    intensities = img.flatten()
    return z, intensities, (H, W)

# ------------------------------------------------------------
# 3. Sort by angle (arg) – O(K log K)
# ------------------------------------------------------------
def sort_by_angle(z, intensities):
    angles = np.angle(z)
    idx = np.argsort(angles)
    z_sorted = z[idx]
    intensities_sorted = intensities[idx]
    return z_sorted, intensities_sorted, idx

# ------------------------------------------------------------
# 4. Build Toeplitz kernel (Gaussian in index difference)
# ------------------------------------------------------------
def toeplitz_kernel(K, sigma):
    kernel = np.zeros(2*K-1, dtype=float)
    for d in range(-(K-1), K):
        kernel[d + (K-1)] = np.exp(-(d*d) / (2*sigma*sigma))
    return kernel

# ------------------------------------------------------------
# 5. Apply convolution (FFT) – O(K log K)
# ------------------------------------------------------------
def apply_convolution(signal, kernel):
    K = len(signal)
    N = 1 << (2*K - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - K), mode='constant')
    ker_pad = np.pad(kernel, (0, N - (2*K - 1)), mode='constant')
    S = fft(sig_pad)
    K_fft = fft(ker_pad)
    Y = S * K_fft
    y = ifft(Y)
    return y[:K]

# ------------------------------------------------------------
# 6. Magnitude compression: keep top M magnitudes
# ------------------------------------------------------------
def compress_magnitude(conv_result, M):
    idx_sorted = np.argsort(np.abs(conv_result))[::-1][:M]
    compressed = {int(i): conv_result[i] for i in idx_sorted}
    return compressed

# ------------------------------------------------------------
# 7. Decompress: reconstruct sequence from compressed dict
# ------------------------------------------------------------
def decompress(compressed, K):
    recon = np.zeros(K, dtype=complex)
    for i, val in compressed.items():
        recon[i] = val
    return recon

# ------------------------------------------------------------
# 8. Full compression pipeline
# ------------------------------------------------------------
def compress_image(img, sigma=2.0, M=None, collatz_mask=False):
    H, W = img.shape
    z, intensities, shape = flatten_image(img)
    K = len(intensities)
    if M is None:
        M = K // 4

    # Sort by angle (TSP routing)
    z_sorted, intensities_sorted, sort_idx = sort_by_angle(z, intensities)

    # Collatz mask (zero out odd indices)
    if collatz_mask:
        for i in range(1, K, 2):
            intensities_sorted[i] = 0

    kernel = toeplitz_kernel(K, sigma)
    conv_result = apply_convolution(intensities_sorted, kernel)

    compressed = compress_magnitude(conv_result, M)

    meta = {
        'shape': shape,
        'sort_idx': sort_idx,
        'K': K,
        'sigma': sigma,
        'M': M,
        'collatz_mask': collatz_mask
    }
    return compressed, meta

# ------------------------------------------------------------
# 9. Decompress and reconstruct image
# ------------------------------------------------------------
def decompress_image(compressed, meta):
    K = meta['K']
    shape = meta['shape']
    sort_idx = meta['sort_idx']

    recon_sorted = decompress(compressed, K)

    unsorted = np.zeros(K, dtype=complex)
    unsorted[sort_idx] = recon_sorted
    recon_img = unsorted.reshape(shape).real
    recon_img = np.clip(recon_img, 0, 1)
    return recon_img

# ------------------------------------------------------------
# 10. Demonstration with different exponents n
# ------------------------------------------------------------
if __name__ == "__main__":
    # Choose exponent n = 1, 5, 15
    for n in [1, 5, 15]:
        print(f"\n--- Exponent n = {n} ---")
        img = load_image(n, size=128)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original (n={n})")
        plt.show()

        sigma = 2.0
        M = 1000   # keep ~6% of coefficients
        compressed, meta = compress_image(img, sigma, M, collatz_mask=False)

        print(f"Original pixels: {meta['K']}")
        print(f"Compressed size: {len(compressed)} (ratio: {len(compressed)/meta['K']:.2%})")

        recon = decompress_image(compressed, meta)

        plt.imshow(recon, cmap='gray')
        plt.title(f"Reconstructed (n={n}, M={len(compressed)})")
        plt.show()

        mse = np.mean((img - recon)**2)
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            print(f"PSNR: {psnr:.2f} dB")
        else:
            print("PSNR: infinite")

input('Press ENTER to exit')