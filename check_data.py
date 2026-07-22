import numpy as np

# Load the FMC data
fmc_data = np.load('fmc_data.npy')

# Check shape and data
print(f"Shape: {fmc_data.shape}")
print(f"Data type: {fmc_data.dtype}")
print(f"Min: {fmc_data.min()}, Max: {fmc_data.max()}")

# Access specific data
time_step_0 = fmc_data[0, :]      # First time step, all receivers
receiver_0 = fmc_data[:, 0]       # All time steps, first receiver