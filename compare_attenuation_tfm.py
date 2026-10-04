"""Compare TFM reconstructions with and without material attenuation.

Runs ``src/array_simulation.py`` twice (attenuation off / on), builds a TFM
image for each resulting FMC dataset, and annotates both images with the
defect location and a measured signal-to-noise ratio (SNR) so the SNR drop
caused by attenuation is directly visible.

Usage:
    python compare_attenuation_tfm.py [--db-per-mm 0.25] [--output-dir attenuation_compare]

Each run of array_simulation.py is launched with SIMNDT_SHOW_PLOTS=0 so it
does not block on interactive plot windows; this script does its own
plotting/annotation afterwards.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert as _hilbert

REPO_ROOT = Path(__file__).resolve().parent
ARRAY_SIM_SCRIPT = REPO_ROOT / "src" / "array_simulation.py"

# GaussianSine N_Cycles used by array_simulation.py's excitation signal.
N_CYCLES = 5


def run_array_simulation(output_npy, apply_attenuation, db_per_mm, extra_env=None):
    """Invoke array_simulation.py as a subprocess with the given attenuation
    settings and return the path to the metadata JSON sidecar it writes."""
    env = dict(**__import__("os").environ)
    env["SIMNDT_SHOW_PLOTS"] = "0"
    env["SIMNDT_FMC_OUTPUT"] = str(output_npy)
    env["SIMNDT_APPLY_ATTENUATION"] = "1" if apply_attenuation else "0"
    env["SIMNDT_ATTENUATION_DB_PER_MM"] = str(db_per_mm)
    if extra_env:
        env.update(extra_env)

    print(
        f"\n=== Running array_simulation.py "
        f"(attenuation={'ON' if apply_attenuation else 'OFF'}) -> {output_npy} ==="
    )
    result = subprocess.run(
        [sys.executable, str(ARRAY_SIM_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"array_simulation.py failed (exit code {result.returncode}) "
            f"for output {output_npy}"
        )

    metadata_path = Path(str(Path(output_npy).with_suffix(""))).with_name(
        Path(output_npy).stem + "_metadata.json"
    )
    return metadata_path


def reconstruct_tfm(fmc, metadata):
    """Delay-and-sum TFM reconstruction (same algorithm as check_data.py).

    Returns (x_mm, z_mm, tfm_env) where tfm_env is the (N_PIX_Z, N_PIX_X)
    envelope amplitude image.
    """
    VL = metadata["VL_m_s"]
    dt = metadata["dt_s"]
    width_mm = metadata["width_mm"]
    height_mm = metadata["height_mm"]
    half_span_mm = metadata["half_span_mm"]
    n_tx, n_rx, time_steps = fmc.shape

    elem_pos_mm = np.linspace(-half_span_mm, half_span_mm, n_tx)

    # Analytic (Hilbert) signal so the TFM sums envelope contributions.
    fmc_a = _hilbert(fmc, axis=2).astype(np.complex64)

    n_pix_x, n_pix_z = 300, 300
    x_mm = np.linspace(-width_mm / 2 * 1.4, width_mm / 2 * 1.4, n_pix_x)
    z_mm = np.linspace(0.5, height_mm * 1.4, n_pix_z)

    # Centre-of-burst delay: GaussianSine N_Cycles=5 @ array_simulation's FREQ_MHZ.
    pulse_delay_s = N_CYCLES / (2.0 * metadata["freq_mhz"] * 1e6)

    tfm = np.zeros((n_pix_z, n_pix_x), dtype=np.complex64)
    for i in range(n_tx):
        d_tx = (
            np.sqrt(
                (x_mm[np.newaxis, :] - elem_pos_mm[i]) ** 2 + z_mm[:, np.newaxis] ** 2
            )
            * 1e-3
        )
        for j in range(n_rx):
            d_rx = (
                np.sqrt(
                    (x_mm[np.newaxis, :] - elem_pos_mm[j]) ** 2
                    + z_mm[:, np.newaxis] ** 2
                )
                * 1e-3
            )
            t_grid = (d_tx + d_rx) / VL + 1.7 * pulse_delay_s
            t_idx = t_grid / dt
            k = np.int32(np.floor(t_idx))
            frac = (t_idx - k).astype(np.float32)
            valid = (k >= 0) & (k < time_steps - 1)
            kc = np.clip(k, 0, time_steps - 2)

            a = fmc_a[i, j, :]
            tfm += np.where(
                valid, a[kc] * (1.0 - frac) + a[kc + 1] * frac, np.float32(0.0)
            )

    tfm_env = np.abs(tfm).astype(np.float32)
    return x_mm, z_mm, tfm_env


def compute_snr_db(
    x_mm,
    z_mm,
    tfm_env,
    defect_x_mm,
    defect_z_mm,
    defect_d_mm,
    wavelength_mm,
    dead_zone_mm,
    probe_x_mm=0.0,
):
    """SNR (dB) = 20*log10(peak amplitude inside the circular defect spot /
    RMS amplitude in a fixed rectangular window just past the array's
    pulse dead-zone).

    The noise window is centred under the probe (``probe_x_mm``), not on
    the defect, so it stays valid as the defect is moved around: it starts
    right after the near-field "dead zone" caused by the transmitted
    pulse's ring-down (``dead_zone_mm`` = one-way distance covered by the
    pulse duration) and extends ``5 * wavelength_mm`` further in x and z.
    This avoids the main-bang clutter near the surface, the defect's own
    sidelobes/mode-converted coda, and (for typical defect/backwall depths)
    the backwall echo.

    Returns ``(snr_db, signal_peak, noise_rms, noise_box)`` where
    ``noise_box = (x_lo, x_hi, z_lo, z_hi)`` in mm, so the caller can draw
    the noise window's bounding box on the TFM image.
    """
    xx, zz = np.meshgrid(x_mm, z_mm)
    dist_mm = np.sqrt((xx - defect_x_mm) ** 2 + (zz - defect_z_mm) ** 2)
    defect_radius_mm = defect_d_mm / 2.0
    signal_mask = dist_mm <= defect_radius_mm

    noise_half_width_mm = 5.0 * wavelength_mm
    noise_height_mm = 5.0 * wavelength_mm - 1.0
    x_lo, x_hi = probe_x_mm - noise_half_width_mm, probe_x_mm + noise_half_width_mm
    z_lo = dead_zone_mm - 3.0
    z_hi = z_lo + noise_height_mm

    noise_mask = (
        (xx >= x_lo) & (xx <= x_hi) & (zz >= z_lo) & (zz <= z_hi)
    )

    signal_peak = tfm_env[signal_mask].max()
    noise_rms = np.sqrt(np.mean(tfm_env[noise_mask] ** 2))

    snr_db = 20.0 * np.log10(signal_peak / (noise_rms + 1e-40))
    noise_box = (float(x_lo), float(x_hi), float(z_lo), float(z_hi))
    return float(snr_db), float(signal_peak), float(noise_rms), noise_box


def plot_tfm(
    ax,
    x_mm,
    z_mm,
    tfm_env,
    metadata,
    snr_db,
    title,
    ref_amplitude,
    noise_box=None,
    dynamic_db=40,
):
    """Plot a TFM image in dB, referenced to a shared absolute amplitude
    (``ref_amplitude``) rather than each image's own peak, so that equal
    color intensity means equal absolute amplitude across images."""
    tfm_db = 20.0 * np.log10(tfm_env / (ref_amplitude + 1e-40) + 1e-12)
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    im = ax.imshow(
        np.clip(tfm_db, -dynamic_db, 0),
        extent=extent,
        aspect="equal",
        cmap="hot",
        origin="upper",
        vmin=-dynamic_db,
        vmax=0,
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Depth (mm)")

    if noise_box is not None:
        x_lo, x_hi, z_lo, z_hi = noise_box
        box = plt.Rectangle(
            (x_lo, z_lo),
            x_hi - x_lo,
            z_hi - z_lo,
            fill=False,
            edgecolor="lime",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(box)

    for defect in metadata["defects"]:
        ax.plot(
            defect["x_mm_centred"],
            defect["depth_mm"],
            "c+",
            markersize=14,
            markeredgewidth=2,
        )

    ax.text(
        0.03,
        0.06,
        f"SNR = {snr_db:.1f} dB",
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
    )
    ax.set_title(title, fontsize=10)
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-per-mm", type=float, default=0.25)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "attenuation_compare"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    no_atten_npy = output_dir / "fmc_no_attenuation.npy"
    atten_npy = output_dir / "fmc_attenuation.npy"

    no_atten_meta_path = run_array_simulation(
        no_atten_npy, apply_attenuation=False, db_per_mm=args.db_per_mm
    )
    atten_meta_path = run_array_simulation(
        atten_npy, apply_attenuation=True, db_per_mm=args.db_per_mm
    )

    results = {}
    for label, npy_path, meta_path in [
        ("no_attenuation", no_atten_npy, no_atten_meta_path),
        ("attenuation", atten_npy, atten_meta_path),
    ]:
        fmc = np.load(npy_path)
        with open(meta_path, encoding="utf-8") as fh:
            metadata = json.load(fh)

        print(f"\n--- TFM reconstruction: {label} ---")
        x_mm, z_mm, tfm_env = reconstruct_tfm(fmc, metadata)

        defect = metadata["defects"][0]
        wavelength_mm = metadata["VL_m_s"] / (metadata["freq_mhz"] * 1e6) * 1000.0
        # One-way distance covered during the transmitted pulse's ring-down
        # (N_CYCLES / freq duration) -- the array's near-field "dead zone".
        dead_zone_mm = metadata["VL_m_s"] * (N_CYCLES / (metadata["freq_mhz"] * 1e6)) * 1000.0
        snr_db, peak, noise, noise_box = compute_snr_db(
            x_mm,
            z_mm,
            tfm_env,
            defect["x_mm_centred"],
            defect["depth_mm"],
            defect["diameter_mm"],
            wavelength_mm,
            dead_zone_mm,
        )
        print(f"  Peak amplitude (defect spot)        : {peak:.4g}")
        print(f"  Noise RMS (dead-zone window)        : {noise:.4g}")
        print(f"  Dead zone depth (mm)                : {dead_zone_mm:.2f}")
        print(f"  Noise window (x_lo,x_hi,z_lo,z_hi)  : {noise_box}")
        print(f"  SNR                                 : {snr_db:.2f} dB")

        results[label] = dict(
            x_mm=x_mm,
            z_mm=z_mm,
            tfm_env=tfm_env,
            metadata=metadata,
            snr_db=snr_db,
            noise_box=noise_box,
        )

    # ── Individual annotated images ──────────────────────────────────────────
    # Shared absolute amplitude reference (0 dB) so equal color intensity
    # means equal absolute amplitude across both images, not self-normalized.
    ref_amplitude = results["no_attenuation"]["tfm_env"].max()
    titles = {
        "no_attenuation": "TFM — No Attenuation",
        "attenuation": f"TFM — Attenuation = {args.db_per_mm:.2f} dB/mm",
    }
    for label, data in results.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        im = plot_tfm(
            ax,
            data["x_mm"],
            data["z_mm"],
            data["tfm_env"],
            data["metadata"],
            data["snr_db"],
            titles[label],
            ref_amplitude,
            noise_box=data["noise_box"],
        )
        plt.colorbar(im, ax=ax, label="dB")
        plt.tight_layout()
        out_png = output_dir / f"tfm_{label}.png"
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved {out_png}")

    # ── Side-by-side comparison ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, label in zip(axes, ["no_attenuation", "attenuation"]):
        data = results[label]
        im = plot_tfm(
            ax,
            data["x_mm"],
            data["z_mm"],
            data["tfm_env"],
            data["metadata"],
            data["snr_db"],
            titles[label],
            ref_amplitude,
            noise_box=data["noise_box"],
        )
    plt.colorbar(im, ax=axes, label="dB", shrink=0.8)
    comparison_png = output_dir / "tfm_comparison.png"
    plt.savefig(comparison_png, dpi=150)
    plt.close(fig)
    print(f"Saved {comparison_png}")

    delta_snr = results["no_attenuation"]["snr_db"] - results["attenuation"]["snr_db"]
    print(f"\nSNR drop due to attenuation: {delta_snr:.2f} dB")
    if delta_snr > 0:
        print("Confirmed: attenuated defect SNR is lower than non-attenuated SNR.")
    else:
        print(
            "WARNING: attenuated SNR was not lower than non-attenuated SNR "
            "(check attenuation coefficient / defect depth)."
        )


if __name__ == "__main__":
    main()
