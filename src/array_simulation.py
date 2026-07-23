"""
5 MHz 32-element linear array FMC (Full Matrix Capture) simulation in steel.

Inspection mode: Full Aperture Receive (1 TX at center × 32 RX = 32 channels per time step).
Material:        Steel  (VL=5850 m/s, VT=3220 m/s, rho=7800 kg/m3)
Signal:          Raised-Cosine pulse @ 5 MHz
Array geometry:  32 elements x 0.585 mm pitch = 18.72 mm aperture
Output:          FMC data array: (TX, RX, TimeSteps) = (N_TX, N_RX, TimeSteps)
"""

import sys
import os
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # use Tk instead of Qt to avoid PySide2/NumPy2 crash
import matplotlib.pyplot as plt
from math import pi

SHOW_PLOTS = os.environ.get("SIMNDT_SHOW_PLOTS", "1") == "1"
FMC_OUTPUT_PATH = os.environ.get("SIMNDT_FMC_OUTPUT", "fmc_data.npy")
REQUIRE_GPU = os.environ.get("SIMNDT_REQUIRE_GPU", "0") == "1"

# ── make sure src/ is on the path ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SimNDT.core.scenario import Scenario
from SimNDT.core.material import Material
from SimNDT.core.boundary import Boundary
from SimNDT.core.constants import BC
from SimNDT.core.transducer import Transducer
from SimNDT.core.signal import Signals
from SimNDT.core.simulation import Simulation
from SimNDT.core.inspectionMethods import FMC
from SimNDT.core.simPack import SimPack
from SimNDT.core.geometryObjects import Circle

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: CPU vs GPU execution
# Auto-detects the first OpenCL GPU; set USE_GPU = False to force CPU.
# ─────────────────────────────────────────────────────────────────────────────
USE_GPU = True  # set to False to force CPU (Cython serial)

GPU_PLATFORM_NAME = None
GPU_DEVICE_TYPE = None

if USE_GPU:
    try:
        import pyopencl as _cl

        _found = None
        for _plat in _cl.get_platforms():
            for _dev in _plat.get_devices():
                _dtype = _cl.device_type.to_string(_dev.type)
                if "GPU" in _dtype:
                    _candidate = (_plat.name, _dtype, _dev.name)
                    if _found is None:
                        _found = _candidate  # take the first GPU found
                    if "NVIDIA" in _plat.name.upper():
                        _found = _candidate  # prefer NVIDIA if present
                        break
        if _found:
            GPU_PLATFORM_NAME, GPU_DEVICE_TYPE, _gpu_name = _found
            print(
                f"OpenCL GPU detected : {_gpu_name!r} on platform {GPU_PLATFORM_NAME!r}"
            )
            print(f"  device type string: {GPU_DEVICE_TYPE!r}")
        else:
            if REQUIRE_GPU:
                raise RuntimeError("SIMNDT_REQUIRE_GPU=1, but no OpenCL GPU was found")
            print("No OpenCL GPU found  — falling back to CPU serial")
            USE_GPU = False
    except ImportError:
        if REQUIRE_GPU:
            raise RuntimeError("SIMNDT_REQUIRE_GPU=1, but pyopencl is not installed")
        print("pyopencl not installed — falling back to CPU serial")
        USE_GPU = False

PLATFORM = "OpenCL" if USE_GPU else "CPU"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MATERIAL  –  Steel
# ─────────────────────────────────────────────────────────────────────────────
rho = 7800.0  # kg/m³
VL = 5850.0  # longitudinal wave speed  (m/s)
VT = 3220.0  # shear wave speed         (m/s)

c11 = rho * VL**2  # ~  2.659e11 Pa
c44 = rho * VT**2  # ~  8.084e10 Pa
c12 = rho * (VL**2 - 2 * VT**2)  # ~  1.042e11 Pa
c22 = c11

steel = Material(name="steel", rho=rho, c11=c11, c12=c12, c22=c22, c44=c44, label=1)
air = Material(
    name="air",
    rho=1.2,  # rho < 2.0 → engine treats as vacuum
    c11=1e-20,
    c12=1e-20,
    c22=1e-20,
    c44=1e-20,
    label=0,
)

materials = [air, steel]  # label 0 = air/void, label 1 = steel

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SCENARIO  –  40 mm wide × 30 mm deep, filled with steel
# ─────────────────────────────────────────────────────────────────────────────
WIDTH_MM = 50  # mm
HEIGHT_MM = 30  # mm
PIXEL_MM = 10  # pixels per mm  (geometric resolution of the model image)

scenario = Scenario(Width=WIDTH_MM, Height=HEIGHT_MM, Pixel_mm=PIXEL_MM, Label=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  ABSORBING BOUNDARIES
# Boundary size is in mm (multiplied by Pixel_mm internally).
# 5 mm absorbing layer on all sides except the bottom (keep as reflective backwall).
# ─────────────────────────────────────────────────────────────────────────────
ABS_SIZE = 0  # absorbing layer thickness in mm

boundaries = [
    Boundary(name="Top", BC=BC.AbsorbingLayer, size=ABS_SIZE),
    Boundary(
        name="Bottom", BC=BC.AbsorbingLayer, size=ABS_SIZE
    ),  # 0 = reflective backwall
    Boundary(name="Left", BC=BC.AbsorbingLayer, size=ABS_SIZE),
    Boundary(name="Right", BC=BC.AbsorbingLayer, size=ABS_SIZE),
]
scenario.createBoundaries(boundaries)

# ─────────────────────────────────────────────────────────────────────────────
# 3b. DEFECT  –  1 mm diameter side-drilled hole at mid-depth, centre
# ─────────────────────────────────────────────────────────────────────────────
# Coordinates are in mm from the top-left corner of the scenario image.
# x0 = WIDTH_MM/2  → horizontal centre
# y0 = HEIGHT_MM/2 → mid-depth (5 mm)
# Label=0 → void (background/water material) inside steel = strong reflector
HOLE_X_MM = float(os.environ.get("SIMNDT_HOLE_X_MM", WIDTH_MM / 2))
HOLE_Y_MM = float(os.environ.get("SIMNDT_HOLE_Y_MM", HEIGHT_MM / 2))
HOLE_D_MM = float(os.environ.get("SIMNDT_HOLE_D_MM", 5.0))

hole = Circle(x0=HOLE_X_MM, y0=HOLE_Y_MM, r=HOLE_D_MM / 2, Label=0)
scenario.addObject(hole)

print(
    f"Hole added : {HOLE_D_MM:.1f} mm diameter at "
    f"x={HOLE_X_MM:.1f} mm, depth={HOLE_Y_MM:.1f} mm"
)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ARRAY TRANSDUCER  –  32 elements, 0.6 mm pitch → 19.2 mm aperture
# ─────────────────────────────────────────────────────────────────────────────
FREQ_MHZ = 5.0  # MHz
FREQ_HZ = FREQ_MHZ * 1e6  # Hz  (5e6)
WAVELENGTH_MM = (VL / FREQ_HZ) * 1e3  # wavelength in mm  (~1.17 mm)
ELEM_SIZE = WAVELENGTH_MM / 2.0  # half-wavelength element in mm  (~0.585 mm)
N_ELEMENTS = 32
PITCH = ELEM_SIZE + 0.1  # pitch = element size, no gap (mm)
HALF_SPAN = (N_ELEMENTS - 1) * PITCH / 2.0  # half aperture in mm  (~9.3 mm)

transducer = Transducer(
    name="array_element",
    Size=ELEM_SIZE,  # single element width (mm)
    CenterOffset=0,
    BorderOffset=0,
    Location="Top",
    PointSource=False,
)

print(f"Element size  : {ELEM_SIZE:.2f} mm")
print(f"Total aperture: {N_ELEMENTS*PITCH:.2f} mm  ({N_ELEMENTS} x {PITCH:.3f} mm)")
print(f"Scan span     : {-HALF_SPAN:.2f} mm to {+HALF_SPAN:.2f} mm")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  EXCITATION SIGNAL  –  Raised-Cosine @ 5 MHz
# ─────────────────────────────────────────────────────────────────────────────
signal = Signals(
    Name="GaussianSine", Amplitud=1.0, Frequency=FREQ_MHZ * 1e6, N_Cycles=5  # Hz
)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# SimTime: round-trip in HEIGHT_MM of steel + 200 % margin
ROUND_TRIP_US = 2.0 * HEIGHT_MM * 1e-3 / VL * 1e6  # µs
SIM_TIME_US = ROUND_TRIP_US * 1.2

simulation = Simulation(
    TimeScale=1,
    MaxFreq=FREQ_HZ,  # Hz  – Simulation.job_parameters divides V[m/s] / (PointCycle * MaxFreq[Hz])
    PointCycle=15,  # grid points per wavelength
    SimTime=SIM_TIME_US * 1e-6,  # seconds
    Order=2,
)

simulation.job_parameters(materials, transducer)

if USE_GPU:
    simulation.setPlatform(GPU_PLATFORM_NAME)
    simulation.setDevice(GPU_DEVICE_TYPE)

print(f"\nNumerical grid step  dx = {simulation.dx*1e3:.4f} mm")
print(f"Time step            dt = {simulation.dt*1e9:.4f} ns")
print(
    f"Simulation time       T = {SIM_TIME_US:.2f} µs  ({int(simulation.TimeSteps)} steps)"
)

scenario.createBoundaries(boundaries)
simulation.create_numericalModel(scenario)

print(f"Numerical model size : {simulation.MRI} × {simulation.NRI} grid points")

# ─────────────────────────────────────────────────────────────────────────────
# 6b. CHECK TRANSDUCER PLACEMENT RELATIVE TO BOUNDARIES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== TRANSDUCER PLACEMENT ANALYSIS ===")
print(f"Sample dimensions (mm):")
print(
    f"  Width:  ±{WIDTH_MM/2:.1f} mm  (from {-WIDTH_MM/2:.1f} to {WIDTH_MM/2:.1f} mm)"
)
print(f"  Height: 0 to {HEIGHT_MM:.1f} mm (surface to depth)")

print(f"\nTransducer element positions (mm):")
print(f"  Leftmost element:  {-HALF_SPAN:.3f} mm")
print(f"  Rightmost element: {+HALF_SPAN:.3f} mm")
print(f"  Scan span width:   {2*HALF_SPAN:.3f} mm")
print(f"  Sample width:      {WIDTH_MM:.3f} mm")

print(f"\nBoundary configuration (grid points):")
print(f"  TapGrid[0] (top):   {simulation.TapGrid[0]}")
print(f"  TapGrid[1] (bot):   {simulation.TapGrid[1]}")
print(f"  TapGrid[2] (left):  {simulation.TapGrid[2]}")
print(f"  TapGrid[3] (right): {simulation.TapGrid[3]}")

print(f"\nGrid resolution:")
print(f"  Rgrid (mm/grid):    {simulation.Rgrid:.6f} mm/point")
print(f"  dx (m):             {simulation.dx:.6e} m")

print(f"\nTransducer placement check:")
if abs(HALF_SPAN) < WIDTH_MM / 2:
    print(f"  [OK] Elements are INSIDE the sample (not on boundary)")
    margin_left = WIDTH_MM / 2 - HALF_SPAN
    margin_right = WIDTH_MM / 2 - HALF_SPAN
    print(f"    Left margin:  {margin_left:.3f} mm from edge")
    print(f"    Right margin: {margin_right:.3f} mm from edge")
else:
    print(f"  [WARNING] Elements may be ON or OUTSIDE the sample boundary!")

# Check if elements are on the top surface (y=0)
print(f"\nVertical placement:")
print(f"  Transducer location: Top surface (y=0)")
print(f"  Excitation row (grid): TapGrid[0] = {int(np.round(simulation.TapGrid[0]))}")
print(f"    This is the physical top boundary (y=0)")
print(
    f"  Receive row (grid):    TapGrid[0]+1 = {int(np.round(simulation.TapGrid[0]))+1}"
)
print(f"    This captures propagated Txx one grid point inside the sample")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FMC INSPECTION  –  Full Aperture Receive (single TX center, all RX)
# ─────────────────────────────────────────────────────────────────────────────
inspection = FMC(
    ini=-HALF_SPAN,  # start offset from centre (mm)
    end=HALF_SPAN,  # end offset from centre (mm)
    step=PITCH,  # step = one pitch (mm)
    Location="Top",
)
# np.arange floating-point rounding can add one spurious step → clamp to exactly N_ELEMENTS
inspection.ScanVector = np.linspace(-HALF_SPAN, HALF_SPAN, N_ELEMENTS)

print(f"\nInspection mode    : {inspection.Name}")
print(f"Transmitter        : 1 element at center")
print(f"Receivers          : {len(inspection.ScanVector)} elements (full aperture)")

# Set up inspection geometry (populates XL, YL, IR arrays)
inspection.setInspection(scenario, transducer, simulation)

# Debug: check actual sizes
print(f"  XL shape (Transmitter): {inspection.XL.shape}")
print(f"  YL shape: {inspection.YL.shape}")
print(f"  IR shape (All Receivers): {inspection.IR.shape}")
N_RX = inspection.IR.shape[0]
print(f"  FMC receive aperture: {N_RX} channels")
print(f"  receiver_signals shape will be: (TimeSteps, {N_RX})")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PACK EVERYTHING into SimPack
# ─────────────────────────────────────────────────────────────────────────────
from SimNDT.core.inspectionMethods import Source

source = Source()
source.Longitudinal = True
source.Shear = False
source.Pressure = True
source.Displacement = False

simpack = SimPack(
    scenario=scenario,
    materials=materials,
    boundary=boundaries,
    inspection=inspection,
    source=source,
    transducers=[transducer],  # must be a list
    signal=signal,
    simulation=simulation,
)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  QUICK VISUALISATION  –  scenario + array footprint
# ─────────────────────────────────────────────────────────────────────────────
if SHOW_PLOTS:
    fig, ax = plt.subplots(figsize=(8, 6))

    # show the scenario image (steel block)
    # y=0 is at top (surface), depth increases downward
    extent = [-WIDTH_MM / 2, WIDTH_MM / 2, HEIGHT_MM, 0]
    ax.imshow(
        scenario.Iabs,
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=2,
        aspect="equal",
        origin="lower",
    )

    # Draw sample boundary (white dashed lines)
    ax.axvline(
        x=-WIDTH_MM / 2,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="Sample edges",
    )
    ax.axvline(x=+WIDTH_MM / 2, color="white", linestyle="--", linewidth=1.5)
    ax.axhline(y=0, color="white", linestyle="--", linewidth=1.5)

    # overlay the array elements on the top surface (y=0)
    for pos in inspection.ScanVector:
        x_left = pos - ELEM_SIZE / 2
        ax.add_patch(
            plt.Rectangle(
                (x_left, 0),
                ELEM_SIZE,
                0.5,
                color="red",
                alpha=0.9,
                linewidth=1,
                label="TX/RX elements" if pos == inspection.ScanVector[0] else "",
            )
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("depth (mm)")
    ax.set_title(
        f"5 MHz {N_ELEMENTS}-element array on steel\n"
        f"Element pitch = {PITCH:.2f} mm, "
        f"dx = {simulation.dx*1e3:.3f} mm, "
        f"dt = {simulation.dt*1e9:.3f} ns"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("array_scenario.png", dpi=150)
    plt.show()

print("\nSimPack ready. scenario, inspection, simulation, signal all configured.")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FULL 32×32 FMC ACQUISITION  –  Loop through all transmitter positions
# ─────────────────────────────────────────────────────────────────────────────
from SimNDT.engine.efit2d import EFIT2D

N_TX = len(inspection.ScanVector)
fmc_matrix = np.zeros((N_TX, N_TX, simulation.TimeSteps), dtype=np.float32)

print(f"\n=== FULL {N_TX}×{N_TX} FMC MATRIX ACQUISITION ===")
print(
    f"Running {N_TX} transmitter positions × {N_TX} receiver channels = {N_TX*N_TX} total channels\n"
)

if USE_GPU:
    print("(OpenCL GPU kernel execution - faster)")
    exec_func_name = "run"
else:
    print("(CPU Cython serial execution)")
    exec_func_name = "runSerial"

for tx_idx in range(N_TX):
    tx_position = inspection.ScanVector[tx_idx]

    # Create inspection with TX at this position
    inspection_tx = FMC(
        ini=-HALF_SPAN,
        end=HALF_SPAN,
        step=PITCH,
        Location="Top",
    )

    # Manually configure this transmitter's geometry
    MRI, NRI = simulation.MRI, simulation.NRI
    TapGrid = simulation.TapGrid
    Rgrid = simulation.Rgrid

    # Excite at y=0; sample returning stress at the first propagated row.
    tx_row = int(np.round(TapGrid[0]))
    rx_row = tx_row + 1
    y_center = (NRI - TapGrid[2] - TapGrid[3]) / 2.0 + TapGrid[
        2
    ]  # horizontal centre col
    grid_per_mm = PIXEL_MM * Rgrid
    nodes_per_element = max(1, int(np.round(ELEM_SIZE * grid_per_mm)))
    y_tx = y_center + tx_position * grid_per_mm

    # Excite every top-surface grid node covered by the transmit element.
    tx_start = int(np.round(y_tx - nodes_per_element / 2.0))
    tx_nodes = np.arange(tx_start, tx_start + nodes_per_element, dtype=np.float32)
    inspection_tx.XL = np.full((nodes_per_element, 2), tx_row, dtype=np.float32)
    inspection_tx.YL = np.column_stack((tx_nodes, tx_nodes)).astype(np.float32)

    # Record every top-surface node for each receiver element; the engine
    # averages each node span into one FMC receive channel.
    element_nodes = []
    IR_list = []
    for rx_offset in inspection.ScanVector:
        y_rx = y_center + rx_offset * grid_per_mm
        rx_start = int(np.round(y_rx - nodes_per_element / 2.0))
        rx_nodes = np.arange(rx_start, rx_start + nodes_per_element, dtype=np.float32)
        element_nodes.append(
            np.column_stack(
                (np.full(nodes_per_element, rx_row, dtype=np.float32), rx_nodes)
            )
        )
        IR_list.append([rx_row, y_rx])
    inspection_tx.IR = np.array(IR_list, dtype=np.float32)
    inspection_tx.ElementNodes = np.array(element_nodes, dtype=np.float32)

    # Update SimPack with new inspection
    simpack.Inspection = inspection_tx

    # Create engine for this transmitter
    engine = EFIT2D(simpack, Platform=PLATFORM)

    # Get execution function
    exec_func = getattr(engine, exec_func_name)

    # Run simulation for this transmitter
    for step in range(simulation.TimeSteps):
        exec_func()
        engine.n += 1

    if USE_GPU:
        engine.saveOutput()  # Copy GPU results back to CPU

    # Store receiver signals for this transmitter
    fmc_matrix[tx_idx, :, :] = engine.receiver_signals.T
    print(
        f"TX {tx_idx + 1}/{N_TX}: position={tx_position:+.2f} mm, "
        f"samples={engine.receiver_signals.shape[0]}, "
        f"receivers={engine.receiver_signals.shape[1]}",
        flush=True,
    )

print("\n=== FMC Acquisition Complete ===")

print(f"\nFull Matrix Capture (FMC) Complete:")
print(f"  Shape: {fmc_matrix.shape}")
print(f"  Transmitters: {fmc_matrix.shape[0]}")
print(f"  Receivers: {fmc_matrix.shape[1]}")
print(f"  TimeSteps: {fmc_matrix.shape[2]}")
print(f"  Total channels: {N_TX * N_TX} = {N_TX} TX × {N_TX} RX")

# Save in (TX, RX, TimeSteps) format
np.save(FMC_OUTPUT_PATH, fmc_matrix)
print(f"\nFMC data saved to: {FMC_OUTPUT_PATH}")
print(f"  Load with: fmc = np.load('fmc_data.npy')")
print(f"  Access signal: fmc[tx_idx, rx_idx, time_step]")
