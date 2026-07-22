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
matplotlib.use('TkAgg')  # use Tk instead of Qt to avoid PySide2/NumPy2 crash
import matplotlib.pyplot as plt
from math import pi

# ── make sure src/ is on the path ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SimNDT.core.scenario       import Scenario
from SimNDT.core.material       import Material
from SimNDT.core.boundary       import Boundary
from SimNDT.core.constants      import BC
from SimNDT.core.transducer     import Transducer
from SimNDT.core.signal         import Signals
from SimNDT.core.simulation     import Simulation
from SimNDT.core.inspectionMethods import FMC
from SimNDT.core.simPack        import SimPack

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: CPU vs GPU execution
# ─────────────────────────────────────────────────────────────────────────────
USE_GPU = False    # Set to True for OpenCL GPU execution, False for CPU
PLATFORM = "OpenCL" if USE_GPU else "CPU"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MATERIAL  –  Steel
# ─────────────────────────────────────────────────────────────────────────────
rho  = 7800.0          # kg/m³
VL   = 5850.0          # longitudinal wave speed  (m/s)
VT   = 3220.0          # shear wave speed         (m/s)

c11  = rho * VL**2                  # ~  2.659e11 Pa
c44  = rho * VT**2                  # ~  8.084e10 Pa
c12  = rho * (VL**2 - 2*VT**2)     # ~  1.042e11 Pa
c22  = c11

steel = Material(name="steel", rho=rho, c11=c11, c12=c12, c22=c22, c44=c44, label=1)
water = Material(name="water", rho=1000.0,
                 c11=1000*1480**2, c12=1000*1480**2,
                 c22=1000*1480**2, c44=1e-30, label=0)

materials = [water, steel]   # label 0 = background (not used here), label 1 = steel

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SCENARIO  –  40 mm wide × 30 mm deep, filled with steel
# ─────────────────────────────────────────────────────────────────────────────
WIDTH_MM  = 20      # mm
HEIGHT_MM = 10      # mm
PIXEL_MM  = 10      # pixels per mm  (geometric resolution of the model image)

scenario = Scenario(Width=WIDTH_MM, Height=HEIGHT_MM, Pixel_mm=PIXEL_MM, Label=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  ABSORBING BOUNDARIES  (10-pixel layer on every side)
# ─────────────────────────────────────────────────────────────────────────────
ABS_SIZE = 5   # absorbing layer thickness in pixels (= 1 mm at 10 px/mm)

boundaries = [
    Boundary(name="Top",    BC=BC.AbsorbingLayer, size=ABS_SIZE),
    Boundary(name="Bottom", BC=BC.AbsorbingLayer, size=0),
    Boundary(name="Left",   BC=BC.AbsorbingLayer, size=ABS_SIZE),
    Boundary(name="Right",  BC=BC.AbsorbingLayer, size=ABS_SIZE),
]
scenario.createBoundaries(boundaries)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ARRAY TRANSDUCER  –  32 elements, 0.6 mm pitch → 19.2 mm aperture
# ─────────────────────────────────────────────────────────────────────────────
FREQ_MHZ    = 5.0        # MHz
FREQ_HZ     = FREQ_MHZ * 1e6           # Hz  (5e6)
WAVELENGTH_MM = (VL / FREQ_HZ) * 1e3  # wavelength in mm  (~1.17 mm)
ELEM_SIZE   = WAVELENGTH_MM / 2.0      # half-wavelength element in mm  (~0.585 mm)
N_ELEMENTS  = 4
PITCH       = ELEM_SIZE + 0.1               # pitch = element size, no gap (mm)
HALF_SPAN   = (N_ELEMENTS - 1) * PITCH / 2.0   # half aperture in mm  (~9.3 mm)

transducer = Transducer(
    name         = "array_element",
    Size         = ELEM_SIZE,           # single element width (mm)
    CenterOffset = 0,
    BorderOffset = 0,
    Location     = "Top",
    PointSource  = False,
)

print(f"Element size  : {ELEM_SIZE:.2f} mm")
print(f"Total aperture: {N_ELEMENTS*PITCH:.2f} mm  ({N_ELEMENTS} × {PITCH:.3f} mm)")
print(f"Scan span     : {-HALF_SPAN:.2f} mm  →  {+HALF_SPAN:.2f} mm")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  EXCITATION SIGNAL  –  Raised-Cosine @ 5 MHz
# ─────────────────────────────────────────────────────────────────────────────
signal = Signals(Name="RaisedCosine", Amplitud=1.0,
                 Frequency=FREQ_MHZ * 1e6,   # Hz
                 N_Cycles=2)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# SimTime: round-trip in HEIGHT_MM of steel + 20 % margin
ROUND_TRIP_US = 2.0 * HEIGHT_MM * 1e-3 / VL * 1e6   # µs
SIM_TIME_US   = ROUND_TRIP_US * 1.2

simulation = Simulation(
    TimeScale  = 1,
    MaxFreq    = FREQ_HZ,     # Hz  – Simulation.job_parameters divides V[m/s] / (PointCycle * MaxFreq[Hz])
    PointCycle = 15,          # grid points per wavelength
    SimTime    = SIM_TIME_US * 1e-6,   # seconds
    Order      = 2,
    Device     = "CPU",       # change to "GPU" if OpenCL GPU device available
)

simulation.job_parameters(materials, transducer)

print(f"\nNumerical grid step  dx = {simulation.dx*1e3:.4f} mm")
print(f"Time step            dt = {simulation.dt*1e9:.4f} ns")
print(f"Simulation time       T = {SIM_TIME_US:.2f} µs  ({int(simulation.TimeSteps)} steps)")

scenario.createBoundaries(boundaries)
simulation.create_numericalModel(scenario)

print(f"Numerical model size : {simulation.MRI} × {simulation.NRI} grid points")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FMC INSPECTION  –  Full Aperture Receive (single TX center, all RX)
# ─────────────────────────────────────────────────────────────────────────────
inspection = FMC(
    ini      = -HALF_SPAN,          # start offset from centre (mm)
    end      =  HALF_SPAN,          # end offset from centre (mm)
    step     =  PITCH,              # step = one pitch (mm)
    Location = "Top",
)

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
print(f"  → FMC receive aperture: {N_RX} channels")
print(f"  → receiver_signals shape will be: (TimeSteps, {N_RX})")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PACK EVERYTHING into SimPack
# ─────────────────────────────────────────────────────────────────────────────
from SimNDT.core.inspectionMethods import Source
source = Source()
source.Longitudinal = True
source.Shear        = False
source.Pressure     = True
source.Displacement = False

simpack = SimPack(
    scenario   = scenario,
    materials  = materials,
    boundary   = boundaries,
    inspection = inspection,
    source     = source,
    transducers= [transducer],   # must be a list
    signal     = signal,
    simulation = simulation,
)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  QUICK VISUALISATION  –  scenario + array footprint
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# show the scenario image (steel block)
# Flip extent so y=0 is at top, y=30 is below
extent = [-WIDTH_MM/2, WIDTH_MM/2, 0, HEIGHT_MM]
ax.imshow(scenario.Iabs, extent=extent, cmap="gray", vmin=0, vmax=2, aspect="equal", origin="upper")

# overlay the array elements on the top surface (y=0)
for pos in inspection.ScanVector:
    x_left  = pos - ELEM_SIZE/2
    x_right = pos + ELEM_SIZE/2
    ax.add_patch(plt.Rectangle((x_left, 0), ELEM_SIZE, 0.5,
                               color="red", alpha=0.9, linewidth=1))

ax.set_xlabel("x (mm)")
ax.set_ylabel("depth (mm)")
ax.set_title(f"5 MHz {N_ELEMENTS}-element array on steel\n"
             f"Element pitch = {PITCH*1e3:.2f} mm, "
             f"dx = {simulation.dx*1e3:.3f} mm, "
             f"dt = {simulation.dt*1e9:.3f} ns")
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
print(f"Running {N_TX} transmitter positions × {N_TX} receiver channels = {N_TX*N_TX} total channels\n")

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
        ini      = -HALF_SPAN,
        end      =  HALF_SPAN,
        step     =  PITCH,
        Location = "Top",
    )
    
    # Manually configure this transmitter's geometry
    MRI, NRI = simulation.MRI, simulation.NRI
    TapGrid = simulation.TapGrid
    Rgrid = simulation.Rgrid
    
    y_tx = (NRI - TapGrid[2] - TapGrid[3]) / 2.0 + TapGrid[2]
    x_center = np.around((MRI) / 2.0)
    x_tx = x_center + tx_position * Rgrid  # TX offset from center
    
    # Single transmitter at position tx_idx
    inspection_tx.XL = np.array([[x_tx, x_tx]], dtype=np.float32)
    inspection_tx.YL = np.array([[y_tx, y_tx]], dtype=np.float32)
    
    # All receivers
    IR_list = []
    for rx_offset in inspection.ScanVector:
        x_rx = x_center + rx_offset * Rgrid
        IR_list.append([x_rx, y_tx])
    inspection_tx.IR = np.array(IR_list, dtype=np.float32)
    
    # Update SimPack with new inspection
    simpack.Inspection = inspection_tx
    
    # Create engine for this transmitter
    engine = EFIT2D(simpack, Platform=PLATFORM)
    
    # Get execution function
    exec_func = getattr(engine, exec_func_name)
    
    print(f"  TX[{tx_idx:2d}] at {tx_position:+7.2f} mm: ", end="", flush=True)
    
    # Run simulation for this transmitter
    for step in range(simulation.TimeSteps):
        exec_func()
        engine.n += 1
    
    if USE_GPU:
        engine.saveOutput()  # Copy GPU results back to CPU
    
    # Store receiver signals for this transmitter
    fmc_matrix[tx_idx, :, :] = engine.receiver_signals.T
    print(f"✓ ({engine.receiver_signals.shape[0]} time steps, {engine.receiver_signals.shape[1]} RX)")

print("\n=== FMC Acquisition Complete ===")

print(f"\nFull Matrix Capture (FMC) Complete:")
print(f"  Shape: {fmc_matrix.shape}")
print(f"  Transmitters: {fmc_matrix.shape[0]}")
print(f"  Receivers: {fmc_matrix.shape[1]}")
print(f"  TimeSteps: {fmc_matrix.shape[2]}")
print(f"  Total channels: {N_TX * N_TX} = {N_TX} TX × {N_TX} RX")

# Save in (TX, RX, TimeSteps) format
np.save("fmc_data.npy", fmc_matrix)
print(f"\nFMC data saved to: fmc_data.npy")
print(f"  Load with: fmc = np.load('fmc_data.npy')")
print(f"  Access signal: fmc[tx_idx, rx_idx, time_step]")