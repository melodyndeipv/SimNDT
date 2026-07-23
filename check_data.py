import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.signal import hilbert as _hilbert

# ─── Simulation parameters (must match array_simulation.py) ──────────────────
VL         = 5850.0          # m/s  longitudinal wave speed in steel
VT         = 3220.0          # m/s  shear wave speed in steel
FREQ_MHZ   = 5.0             # MHz
FREQ_HZ    = FREQ_MHZ * 1e6
PointCycle = 15
N_CYCLES   = 5               # GaussianSine burst cycles
HEIGHT_MM  = 40.0            # mm  scenario depth
WIDTH_MM   = 40.0            # mm  scenario width
HOLE_X_MM  = WIDTH_MM / 2    # hole x  (mm from left edge → 0 mm in centred coords)
HOLE_Z_MM  = HEIGHT_MM / 2   # hole depth (mm)

# Derived grid parameters (same formulas as Simulation.job_parameters, Order=2)
dx = VT / (PointCycle * FREQ_HZ)        # m  — uses min wave speed
dt = 0.5 * 0.7071 * dx / VL            # s  — uses max wave speed
PULSE_DELAY_S = N_CYCLES / (2.0 * FREQ_HZ)  # GaussianSine burst centre

# Array geometry
ELEM_SIZE_MM = (VL / FREQ_HZ) * 1e3 / 2.0   # half-wavelength ~0.585 mm
N_ELEMENTS   = 32
PITCH_MM     = ELEM_SIZE_MM + 0.1
HALF_SPAN    = (N_ELEMENTS - 1) * PITCH_MM / 2.0

# ─── Load FMC data ────────────────────────────────────────────────────────────
fmc = np.load('fmc_data.npy')           # shape: (N_TX, N_RX, TimeSteps)
N_TX, N_RX, TimeSteps = fmc.shape

# Derive element positions from actual data shape so check_data always matches
# the simulation even if N_ELEMENTS was off by one due to floating-point.
elem_pos_mm = np.linspace(-HALF_SPAN, HALF_SPAN, N_TX)   # (N_TX,)

print(f"FMC shape  : {fmc.shape}  →  (TX={N_TX}, RX={N_RX}, TimeSteps={TimeSteps})")
print(f"dtype      : {fmc.dtype}   min={fmc.min():.3g}  max={fmc.max():.3g}")
print(f"dx = {dx*1e3:.4f} mm    dt = {dt*1e9:.4f} ns")
print(f"GaussianSine centre delay: {PULSE_DELAY_S*1e6:.3f} us")
print(f"Array span : {2*HALF_SPAN:.2f} mm  ({N_TX} elements, pitch={PITCH_MM:.3f} mm)")

# ─── TFM reconstruction ───────────────────────────────────────────────────────
# Pre-compute analytic (Hilbert) signal along the time axis so the TFM sums
# envelope contributions rather than raw oscillatory values. Without this,
# phase cancellation produces ring artifacts instead of a focused spot.
fmc_a = _hilbert(fmc, axis=2).astype(np.complex64)   # (N_TX, N_RX, TimeSteps)

N_PIX_X = 300
N_PIX_Z = 300
x_mm = np.linspace(-WIDTH_MM / 2 * 1.4, WIDTH_MM / 2 * 1.4, N_PIX_X)  # (N_PIX_X,)
z_mm = np.linspace(0.5, HEIGHT_MM*1.4,              N_PIX_Z)   # (N_PIX_Z,) depth

tfm = np.zeros((N_PIX_Z, N_PIX_X), dtype=np.complex64)

print(f"\nRunning TFM  ({N_TX}×{N_RX} = {N_TX*N_RX} A-scans, "
      f"grid {N_PIX_Z}×{N_PIX_X}) ...")

for i in range(N_TX):
    for j in range(N_RX):
        # ---- travel-time grid (N_PIX_Z, N_PIX_X) ----------------------------
        d_tx = np.sqrt((x_mm[np.newaxis, :] - elem_pos_mm[i])**2
                       + z_mm[:, np.newaxis]**2) * 1e-3          # m
        d_rx = np.sqrt((x_mm[np.newaxis, :] - elem_pos_mm[j])**2
                       + z_mm[:, np.newaxis]**2) * 1e-3          # m
        t_grid = (d_tx + d_rx) / VL + PULSE_DELAY_S              # s

        # ---- linear interpolation into A-scan --------------------------------
        t_idx = t_grid / dt                                       # float index
        k     = np.int32(np.floor(t_idx))
        frac  = (t_idx - k).astype(np.float32)
        valid = (k >= 0) & (k < TimeSteps - 1)
        kc    = np.clip(k, 0, TimeSteps - 2)

        a    = fmc_a[i, j, :]    # analytic A-scan
        tfm += np.where(valid,
                        a[kc] * (1.0 - frac) + a[kc + 1] * frac,
                        np.float32(0.0))

    if (i + 1) % max(1, N_TX // 4) == 0:
        print(f"  TX {i+1}/{N_TX}")

# Envelope (|analytic TFM|) and dB normalisation
tfm_env = np.abs(tfm).astype(np.float32)
tfm_db  = 20.0 * np.log10(tfm_env / (tfm_env.max() + 1e-40) + 1e-12)

print("TFM complete — plotting ...")

# ─── Plot ─────────────────────────────────────────────────────────────────────
DYNAMIC_DB = 30
extent = [-WIDTH_MM / 2 * 1.5, WIDTH_MM / 2 * 1.5, HEIGHT_MM * 1.5, 0]   # (left, right, bottom, top)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# left: linear amplitude
im0 = axes[0].imshow(tfm_env, extent=extent, aspect='equal',
                     cmap='hot', origin='upper')
axes[0].set_title('TFM — linear amplitude')
axes[0].set_xlabel('x (mm)')
axes[0].set_ylabel('Depth (mm)')
plt.colorbar(im0, ax=axes[0])

# right: dB dynamic range
im1 = axes[1].imshow(np.clip(tfm_db, -DYNAMIC_DB, 0), extent=extent,
                     aspect='equal', cmap='hot', origin='upper',
                     vmin=-DYNAMIC_DB, vmax=0)
axes[1].set_title(f'TFM — {DYNAMIC_DB} dB dynamic range')
axes[1].set_xlabel('x (mm)')
axes[1].set_ylabel('Depth (mm)')
plt.colorbar(im1, ax=axes[1], label='dB')

# Mark expected hole position (convert from left-edge coords to centred coords)
hole_x_centred = HOLE_X_MM - WIDTH_MM / 2
for ax in axes:
    ax.plot(hole_x_centred, HOLE_Z_MM, 'c+', markersize=14, markeredgewidth=2,
            label=f'Hole ({hole_x_centred:.1f}, {HOLE_Z_MM:.1f}) mm')
    ax.legend(fontsize=8)

plt.suptitle(f'TFM reconstruction  —  {N_TX}-element array @ {FREQ_MHZ:.0f} MHz, steel VL={VL:.0f} m/s',
             fontsize=11)
plt.tight_layout()
plt.savefig('tfm_image.png', dpi=150)
print("Saved tfm_image.png")
plt.show()
