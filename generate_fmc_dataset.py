"""Generate randomized FMC acquisitions without launching another script."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from SimNDT.core.boundary import Boundary
from SimNDT.core.constants import BC
from SimNDT.core.geometryObjects import Circle
from SimNDT.core.inspectionMethods import FMC, Source
from SimNDT.core.material import Material
from SimNDT.core.scenario import Scenario
from SimNDT.core.signal import Signals
from SimNDT.core.simPack import SimPack
from SimNDT.core.simulation import Simulation
from SimNDT.core.transducer import Transducer
from SimNDT.engine.efit2d import EFIT2D

VL_M_S = 5850.0
VT_M_S = 3220.0
RHO_KG_M3 = 7800.0
FREQ_HZ = 2.0e6
WIDTH_MM = 50.0
HEIGHT_MM = 30.0
N_ELEMENTS = 32
PIXEL_MM = 10.0
POINT_CYCLE = 15
N_CYCLES = 5
USE_GPU = True
ELEMENT_SIZE_MM = (VL_M_S / FREQ_HZ) * 1.0e3 / 2.0
PITCH_MM = ELEMENT_SIZE_MM + 0.1
HALF_APERTURE_MM = (N_ELEMENTS - 1) * PITCH_MM / 2.0
DEFECT_EDGE_MARGIN_MM = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate raw 32x32 FMC NPY files with randomized circular defects."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("fmc30mm_2MHz_dataset"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--defect-count", type=int, default=1)
    parser.add_argument("--min-diameter-mm", type=float, default=1.5)
    parser.add_argument("--max-diameter-mm", type=float, default=5.0)
    parser.add_argument("--min-depth-mm", type=float, default=HEIGHT_MM * 0.2)
    parser.add_argument("--max-depth-mm", type=float, default=HEIGHT_MM * 0.8)
    parser.add_argument("--lateral-limit-mm", type=float, default=None)
    args = parser.parse_args()
    if args.lateral_limit_mm is None:
        args.lateral_limit_mm = min(
            HALF_APERTURE_MM,
            WIDTH_MM / 2.0 - args.max_diameter_mm / 2.0 - DEFECT_EDGE_MARGIN_MM,
        )
    return args


def validate_args(args):
    if args.count < 0 or args.start_index < 0:
        raise ValueError("count and start-index must be non-negative")
    if args.defect_count < 0:
        raise ValueError("defect-count must be non-negative")
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
    return {
        "x_mm": WIDTH_MM / 2.0 + x_centered_mm,
        "depth_mm": rng.uniform(depth_min, depth_max),
        "diameter_mm": diameter_mm,
        "x_centered_mm": x_centered_mm,
    }


def append_metadata(metadata_path, row):
    existing_rows = []
    existing_fields = []
    if metadata_path.exists():
        with metadata_path.open(newline="", encoding="ascii") as metadata_file:
            existing_rows = list(csv.DictReader(metadata_file))
            existing_fields = list(existing_rows[0].keys()) if existing_rows else []
    fieldnames = existing_fields + [key for key in row if key not in existing_fields]
    rows = [dict.fromkeys(fieldnames, "") for _ in existing_rows]
    for target, source in zip(rows, existing_rows):
        target.update(source)
    rows.append(dict.fromkeys(fieldnames, ""))
    rows[-1].update(row)
    with metadata_path.open("w", newline="", encoding="ascii") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_simulation(defects):
    c11 = RHO_KG_M3 * VL_M_S**2
    c44 = RHO_KG_M3 * VT_M_S**2
    c12 = RHO_KG_M3 * (VL_M_S**2 - 2 * VT_M_S**2)
    steel = Material("steel", RHO_KG_M3, c11, c12, c11, c44, 1)
    air = Material("air", 1.2, 1e-20, 1e-20, 1e-20, 1e-20, 0)
    scenario = Scenario(Width=WIDTH_MM, Height=HEIGHT_MM, Pixel_mm=PIXEL_MM, Label=1)
    boundaries = [
        Boundary(name, BC=BC.AbsorbingLayer, size=0)
        for name in ("Top", "Bottom", "Left", "Right")
    ]
    scenario.createBoundaries(boundaries)
    for defect in defects:
        scenario.addObject(
            Circle(
                x0=defect["x_mm"],
                y0=defect["depth_mm"],
                r=defect["diameter_mm"] / 2.0,
                Label=0,
            )
        )

    transducer = Transducer(
        name="array_element",
        Size=ELEMENT_SIZE_MM,
        CenterOffset=0,
        BorderOffset=0,
        Location="Top",
        PointSource=False,
    )
    signal = Signals(
        Name="GaussianSine",
        Amplitud=1.0,
        Frequency=FREQ_HZ,
        N_Cycles=N_CYCLES,
    )
    sim_time_s = 2.0 * HEIGHT_MM * 1e-3 / VL_M_S * 1.2
    simulation = Simulation(
        TimeScale=1,
        MaxFreq=FREQ_HZ,
        PointCycle=POINT_CYCLE,
        SimTime=sim_time_s,
        Order=2,
    )
    simulation.job_parameters([air, steel], transducer)

    platform = "CPU"
    if USE_GPU:
        import pyopencl as cl

        selected = None
        for opencl_platform in cl.get_platforms():
            for device in opencl_platform.get_devices():
                device_type = cl.device_type.to_string(device.type)
                if "GPU" in device_type:
                    selected = (opencl_platform.name, device_type)
                    if "NVIDIA" in opencl_platform.name.upper():
                        break
            if selected and "NVIDIA" in selected[0].upper():
                break
        if selected is None:
            raise RuntimeError("USE_GPU=True, but no OpenCL GPU was found")
        simulation.setPlatform(selected[0])
        simulation.setDevice(selected[1])
        platform = "OpenCL"

    scan_vector = np.linspace(-HALF_APERTURE_MM, HALF_APERTURE_MM, N_ELEMENTS)
    inspection = FMC(
        ini=-HALF_APERTURE_MM,
        end=HALF_APERTURE_MM,
        step=PITCH_MM,
        Location="Top",
    )
    inspection.ScanVector = scan_vector
    simulation.create_numericalModel(scenario)
    inspection.setInspection(scenario, transducer, simulation)
    source = Source()
    source.Longitudinal = True
    source.Shear = False
    source.Pressure = True
    source.Displacement = False
    simpack = SimPack(
        scenario=scenario,
        materials=[air, steel],
        boundary=boundaries,
        inspection=inspection,
        source=source,
        transducers=[transducer],
        signal=signal,
        simulation=simulation,
    )
    return simpack, simulation, scan_vector, platform


def run_sample(output_path, defects):
    temporary_output = output_path.with_suffix(".tmp.npy")
    simpack, simulation, scan_vector, platform = create_simulation(defects)
    n_tx = len(scan_vector)
    fmc_matrix = np.zeros((n_tx, n_tx, simulation.TimeSteps), dtype=np.float32)
    grid_per_mm = PIXEL_MM * simulation.Rgrid
    nodes_per_element = max(1, int(np.round(ELEMENT_SIZE_MM * grid_per_mm)))
    tx_row = int(np.round(simulation.TapGrid[0]))
    rx_row = tx_row + 1
    y_center = (
        simulation.NRI - simulation.TapGrid[2] - simulation.TapGrid[3]
    ) / 2.0 + simulation.TapGrid[2]
    execution_name = "run" if platform == "OpenCL" else "runSerial"

    for tx_index, tx_position in enumerate(scan_vector):
        inspection = FMC(
            ini=-HALF_APERTURE_MM,
            end=HALF_APERTURE_MM,
            step=PITCH_MM,
            Location="Top",
        )
        tx_center = y_center + tx_position * grid_per_mm
        tx_start = int(np.round(tx_center - nodes_per_element / 2.0))
        tx_nodes = np.arange(
            tx_start,
            tx_start + nodes_per_element,
            dtype=np.float32,
        )
        inspection.XL = np.full((nodes_per_element, 2), tx_row, dtype=np.float32)
        inspection.YL = np.column_stack((tx_nodes, tx_nodes)).astype(np.float32)
        receiver_nodes = []
        receiver_positions = []
        for rx_position in scan_vector:
            rx_center = y_center + rx_position * grid_per_mm
            rx_start = int(np.round(rx_center - nodes_per_element / 2.0))
            rx_nodes = np.arange(
                rx_start,
                rx_start + nodes_per_element,
                dtype=np.float32,
            )
            receiver_nodes.append(
                np.column_stack(
                    (np.full(nodes_per_element, rx_row, dtype=np.float32), rx_nodes)
                )
            )
            receiver_positions.append([rx_row, rx_center])
        inspection.IR = np.asarray(receiver_positions, dtype=np.float32)
        inspection.ElementNodes = np.asarray(receiver_nodes, dtype=np.float32)
        simpack.Inspection = inspection

        engine = EFIT2D(simpack, Platform=platform)
        execute = getattr(engine, execution_name)
        for _ in range(simulation.TimeSteps):
            execute()
            engine.n += 1
        if platform == "OpenCL":
            engine.saveOutput()
        fmc_matrix[tx_index] = engine.receiver_signals.T

    np.save(temporary_output, fmc_matrix)
    temporary_output.replace(output_path)


def main():
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    print(f"Output directory: {output_dir}")
    print(f"Generating {args.count} FMC files starting at index {args.start_index}")

    completed_count = 0
    for sample_index in range(args.start_index, args.start_index + args.count):
        output_path = output_dir / f"fmc_{sample_index:05d}.npy"
        if output_path.exists():
            print(f"[{sample_index:05d}] exists, skipping")
            continue
        rng = np.random.default_rng(args.seed + sample_index)
        defects = [sample_defect(rng, args) for _ in range(args.defect_count)]
        print(f"[{sample_index:05d}] generating", flush=True)
        run_sample(output_path, defects)
        metadata = {"sample_id": sample_index, "fmc_file": output_path.name}
        for defect_index, defect in enumerate(defects, start=1):
            metadata.update(
                {
                    f"hole_{defect_index}_x_mm_from_left": f"{defect['x_mm']:.8f}",
                    f"hole_{defect_index}_x_mm_centered": f"{defect['x_centered_mm']:.8f}",
                    f"hole_{defect_index}_depth_mm": f"{defect['depth_mm']:.8f}",
                    f"hole_{defect_index}_diameter_mm": f"{defect['diameter_mm']:.8f}",
                }
            )
        append_metadata(metadata_path, metadata)
        completed_count += 1
    print(f"Completed {completed_count} new FMC files.")


if __name__ == "__main__":
    main()
