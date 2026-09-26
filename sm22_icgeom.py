"""IC-level geometry of ellipticity loss for sm22_grow2d seeds.
For each seed: A_Z (fixed-shape), and the Hessian of the ignorable component at its
critical point in the field-aligned frame (t along in-plane field, n = rays):
aniso = |w_tt| / |w_nn|  (large => W-trough elongated along rays => long Z_bad segments)."""
import sys, numpy as np
from sm22_grow2d import seed_state
th = np.deg2rad(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
N = 128
k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
KL, KZ = np.meshgrid(k, k, indexing="ij")
d = lambda f, K: np.real(np.fft.ifft2(1j * K * np.fft.fft2(f)))
for s in range(1, 8):
    B = seed_state(N, th, 1.0, s, 0.05, 1)
    wbar = -np.sin(th); dw = B[2] - wbar
    f = dw * np.sign(-wbar)
    i = np.unravel_index(np.argmax(f), f.shape)
    AZ = 0.05 * abs(wbar) / f[i]
    # at small A the in-plane field is ~ mean in-plane direction l-hat; use actual local direction
    m = B[:2, i[0], i[1]]; t = m / np.linalg.norm(m); n = np.array([-t[1], t[0]])
    H = np.array([[d(d(dw, KL), KL)[i], d(d(dw, KL), KZ)[i]], [d(d(dw, KZ), KL)[i], d(d(dw, KZ), KZ)[i]]])
    wtt, wnn = t @ H @ t, n @ H @ n
    # number of separate maxima within 10% of the top (multiple birth sites)
    print(f"seed {s}: A_Z(IC)={AZ:.3f}  |w_tt|/|w_nn|={abs(wtt)/abs(wnn):.2f}  at {i}")
