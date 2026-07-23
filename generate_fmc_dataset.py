"""Generate randomized FMC acquisitions for later TFM reconstruction.

Each completed sample is saved as ``fmc_00000.npy`` in the output directory.
``metadata.csv`` stores the physical circular-defect parameters needed to make
labels after reconstruction.
"""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

VL_M_S = 5850.0
FREQ_HZ = 5.0e6
WIDTH_MM = 50.0
HEIGHT_MM = 30.0
N_ELEMENTS = 32
ELEMENT_SIZE_MM = (VL_M_S / FREQ_HZ) * 1.0e3 / 2.0
PITCH_MM = ELEMENT_SIZE_MM + 0.1
HALF_APERTURE_MM = (N_ELEMENTS - 1) * PITCH_MM / 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate raw 32x32 FMC NPY files with randomized circular defects."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("fmc_dataset"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--min-diameter-mm", type=float, default=1.5)
    parser.add_argument("--max-diameter-mm", type=float, default=5.0)
    parser.add_argument("--min-depth-mm", type=float, default=7.0)
    parser.add_argument("--max-depth-mm", type=float, default=23.0)
    parser.add_argument("--lateral-limit-mm", type=float, default=8.0)
    return parser.parse_args()


def validate_args(args):
    if args.count < 0 or args.start_index < 0:
        raise ValueError("count and start-index must be non-negative")
    if not 0 < args.min_diameter_mm <= args.max_diameter_mm:
        raise ValueError("diameter bounds must be positive and ordered")
    if not 0 < args.min_depth_mm <= args.max_depth_mm < HEIGHT_MM:
        raise ValueError("depth bounds must be inside the sample and ordered")
    if not 0 < args.lateral_limit_mm <= HALF_APERTURE_MM:
        raise ValueError("lateral-limit-mm must lie within the array half-aperture")


def sample_defect(rng, args):
    diameter_mm = rng.uniform(args.min_diameter_mm, args.max_diameter_mm)
    radius_mm = diameter_mm / 2.0
    depth_min = max(args.min_depth_mm, radius_mm + 1.0)
    depth_max = min(args.max_depth_mm, HEIGHT_MM - radius_mm - 1.0)
    if depth_min > depth_max:
        raise ValueError("depth bounds leave no room for the requested defect size")

    x_centered_mm = rng.uniform(-args.lateral_limit_mm, args.lateral_limit_mm)
    depth_mm = rng.uniform(depth_min, depth_max)
    return {
        "x_mm": WIDTH_MM / 2.0 + x_centered_mm,
        "depth_mm": depth_mm,
        "diameter_mm": diameter_mm,
        "x_centered_mm": x_centered_mm,
    }


def append_metadata(metadata_path, row):
    write_header = not metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="ascii") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_sample(repo_root, simulator_path, output_path, defect):
    temporary_output = output_path.with_suffix(".tmp.npy")
    environment = os.environ.copy()
    environment.update(
        {
            "SIMNDT_REQUIRE_GPU": "1",
            "SIMNDT_SHOW_PLOTS": "0",
            "SIMNDT_FMC_OUTPUT": str(temporary_output.resolve()),
            "SIMNDT_HOLE_X_MM": f"{defect['x_mm']:.8f}",
            "SIMNDT_HOLE_Y_MM": f"{defect['depth_mm']:.8f}",
            "SIMNDT_HOLE_D_MM": f"{defect['diameter_mm']:.8f}",
        }
    )

    try:
        completed = subprocess.run(
            [sys.executable, str(simulator_path)],
            cwd=repo_root,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Simulation failed for {output_path.name}:\n{error.stdout}\n{error.stderr}"
        ) from error

    if not temporary_output.exists():
        raise RuntimeError(f"Simulation finished without creating {temporary_output}")
    temporary_output.replace(output_path)
    return completed.stdout


def main():
    args = parse_args()
    validate_args(args)

    repo_root = Path(__file__).resolve().parent
    simulator_path = repo_root / "src" / "array_simulation.py"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"

    print(f"Output directory: {output_dir}")
    print(f"Generating {args.count} FMC files starting at index {args.start_index}")
    print(
        "Defects: diameter %.1f-%.1f mm; centered x +/-%.1f mm; depth %.1f-%.1f mm"
        % (
            args.min_diameter_mm,
            args.max_diameter_mm,
            args.lateral_limit_mm,
            args.min_depth_mm,
            args.max_depth_mm,
        )
    )

    completed_count = 0
    for sample_index in range(args.start_index, args.start_index + args.count):
        output_path = output_dir / f"fmc_{sample_index:05d}.npy"
        defect = sample_defect(np.random.default_rng(args.seed + sample_index), args)
        if output_path.exists():
            print(f"[{sample_index:05d}] exists, skipping")
            continue

        print(
            "[%05d] x=%+.2f mm, depth=%.2f mm, diameter=%.2f mm"
            % (
                sample_index,
                defect["x_centered_mm"],
                defect["depth_mm"],
                defect["diameter_mm"],
            ),
            flush=True,
        )
        run_sample(repo_root, simulator_path, output_path, defect)
        append_metadata(
            metadata_path,
            {
                "sample_id": sample_index,
                "fmc_file": output_path.name,
                "x_mm_from_left": f"{defect['x_mm']:.8f}",
                "x_mm_centered": f"{defect['x_centered_mm']:.8f}",
                "depth_mm": f"{defect['depth_mm']:.8f}",
                "diameter_mm": f"{defect['diameter_mm']:.8f}",
            },
        )
        completed_count += 1

    print(f"Completed {completed_count} new FMC files.")


if __name__ == "__main__":
    main()
