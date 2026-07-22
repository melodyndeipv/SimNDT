"""
5 MHz 32-element linear array simulation in steel (metal).

Inspection mode: PulseEcho LinearScan (each element fires & receives).
Material:        Steel  (VL=5850 m/s, VT=3220 m/s, rho=7800 kg/m3)
Signal:          Raised-Cosine pulse @ 5 MHz
Array geometry:  32 elements x 0.6 mm pitch = 19.2 mm aperture
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
from SimNDT.core.inspectionMethods import LinearScan
from SimNDT.core.simPack        import SimPack

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
WIDTH_MM  = 40      # mm
HEIGHT_MM = 30      # mm
PIXEL_MM  = 10      # pixels per mm  (geometric resolution of the model image)

scenario = Scenario(Width=WIDTH_MM, Height=HEIGHT_MM, Pixel_mm=PIXEL_MM, Label=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  ABSORBING BOUNDARIES  (10-pixel layer on every side)
# ─────────────────────────────────────────────────────────────────────────────
ABS_SIZE = 10   # absorbing layer thickness in pixels (= 1 mm at 10 px/mm)

boundaries = [
    Boundary(name="Top",    BC=BC.AbsorbingLayer, size=ABS_SIZE),
    Boundary(name="Bottom", BC=BC.AbsorbingLayer, size=ABS_SIZE),
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
N_ELEMENTS  = 32
PITCH       = ELEM_SIZE                # pitch = element size, no gap (mm)
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
# 7.  LINEAR SCAN INSPECTION  –  sweep all 32 element positions
# ─────────────────────────────────────────────────────────────────────────────
inspection = LinearScan(
    ini      = -HALF_SPAN,          # start offset from centre (mm)
    end      =  HALF_SPAN,          # end offset from centre (mm)
    step     =  PITCH,              # step = one pitch (mm)
    Location = "Top",
    Method   = "PulseEcho",
    Theta    = [270.0*pi/180.0, 270.0*pi/180.0],   # normal incidence, top
)

print(f"\nLinear scan positions : {len(inspection.ScanVector)} steps")
print(f"  from {inspection.ScanVector[0]:.3f} mm  to  {inspection.ScanVector[-1]:.3f} mm")

# Set up inspection geometry (populates XL, YL, etc.)
inspection.setInspection(scenario, transducer, simulation)

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
extent = [0, WIDTH_MM, HEIGHT_MM, 0]
ax.imshow(scenario.Iabs, extent=extent, cmap="gray", vmin=0, vmax=2, aspect="equal")

# overlay the array elements on the top surface
for pos in inspection.ScanVector:
    x_left  = WIDTH_MM/2 + pos - ELEM_SIZE/2
    x_right = WIDTH_MM/2 + pos + ELEM_SIZE/2
    ax.add_patch(plt.Rectangle((x_left, 0), ELEM_SIZE, 1.0,
                               color="red", alpha=0.8))

ax.set_xlabel("x (mm)")
ax.set_ylabel("depth (mm)")
ax.set_title(f"5 MHz {N_ELEMENTS}-element array on steel\n"
             f"Element pitch = {PITCH*1e3:.2f} mm, "
             f"dx = {simulation.dx*1e3:.3f} mm, "
             f"dt = {simulation.dt*1e9:.3f} ns")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("array_scenario.png", dpi=150)
plt.show()

print("\nSimPack ready. scenario, inspection, simulation, signal all configured.")

## Run simulation
from SimNDT.engine.efit2d import EFIT2D

engine = EFIT2D(simpack, Platform="OpenCL")   # GPU with OpenCL

print(f"Running {simulation.TimeSteps} time steps on GPU...")
print("(OpenCL kernel execution - this is much faster than CPU)")

# For GPU: use engine.run() which executes the full simulation
# The loop is handled internally by the OpenCL kernel
for step in range(simulation.TimeSteps):
    engine.run()
    engine.n += 1
    if engine.n % 500 == 0:
        print(f"  step {engine.n}/{simulation.TimeSteps}")

print("Simulation complete.")

np.save("receiver_signals.npy", engine.receiver_signals)
print(f"Receiver signals saved: {engine.receiver_signals.shape}")