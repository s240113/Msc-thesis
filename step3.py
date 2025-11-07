import pandas as pd
import numpy as np

# ------------------------------
# Step 3: Measurement Constraints & Partial Observability
# ------------------------------

# Load Step 2 dataset
feeder_name = "IEEE33"
df = pd.read_csv(f"synthetic_power_flow_{feeder_name}.csv")

# Parameters
sparsity_ratio = 0.5           # fraction of buses to keep measurements
noise_voltage_pct = 0.005      # 0.5% noise on voltage
noise_P_pct = 0.02             # 2% noise on active power
noise_Q_pct = 0.02             # 2% noise on reactive power
zero_injection_ratio = 0.1     # fraction of buses to set as zero-injection

# ------------------------------
# 1. Introduce measurement sparsity
# ------------------------------

# Identify all buses
all_buses = df['bus'].unique()
n_observed = int(len(all_buses) * sparsity_ratio)

# Randomly select observed buses
observed_buses = np.random.choice(all_buses, n_observed, replace=False)

# Mask unobserved bus measurements
df_sparse = df.copy()
df_sparse.loc[~df_sparse['bus'].isin(observed_buses), ['voltage_pu','P_MW','Q_MVar']] = np.nan

# ------------------------------
# 2. Introduce zero-injection nodes
# ------------------------------
n_zero_inj = int(len(all_buses) * zero_injection_ratio)
zero_inj_buses = np.random.choice(all_buses, n_zero_inj, replace=False)

# Set loads to zero
df_sparse.loc[df_sparse['bus'].isin(zero_inj_buses), ['P_MW','Q_MVar']] = 0.0

# ------------------------------
# 3. Add measurement noise
# ------------------------------

# Function to add Gaussian noise
def add_noise(series, pct):
    noise = np.random.normal(0, pct * series.abs(), size=len(series))
    return series + noise

# Apply noise only to observed buses
df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'voltage_pu'] = add_noise(
    df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'voltage_pu'], noise_voltage_pct)
df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'P_MW'] = add_noise(
    df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'P_MW'], noise_P_pct)
df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'Q_MVar'] = add_noise(
    df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'Q_MVar'], noise_Q_pct)

# ------------------------------
# 4. Optional: Grid Probing
# ------------------------------
def apply_grid_probing(df, buses_to_probe, magnitude=0.01):
    df_probe = df.copy()
    for bus in buses_to_probe:
        df_probe.loc[df_probe['bus'] == bus, 'P_MW'] *= (1 + magnitude * np.random.choice([-1,1]))
    return df_probe

# Example: probe 5 random observed buses
probe_buses = np.random.choice(observed_buses, min(5, len(observed_buses)), replace=False)
df_probed = apply_grid_probing(df_sparse, probe_buses)

# ------------------------------
# 5. Optional: Covariance-based synthetic measurements
# ------------------------------
def synthetic_measurements_covariance(df_full, df_sparse):
    # Compute covariance from fully observed data
    cov_matrix = df_full[['voltage_pu','P_MW','Q_MVar']].cov()
    # For simplicity, generate synthetic values for NaNs based on mean and covariance
    df_syn = df_sparse.copy()
    for col in ['voltage_pu','P_MW','Q_MVar']:
        nan_idx = df_syn[col].isna()
        mu = df_full[col].mean()
        sigma = df_full[col].std()
        df_syn.loc[nan_idx, col] = np.random.normal(mu, sigma, nan_idx.sum())
    return df_syn

df_synthetic = synthetic_measurements_covariance(df, df_sparse)

# ------------------------------
# 6. Save final datasets
# ------------------------------
df_sparse.to_csv(f"{feeder_name}_sparse_noisy.csv", index=False)
df_probed.to_csv(f"{feeder_name}_probed.csv", index=False)
df_synthetic.to_csv(f"{feeder_name}_synthetic.csv", index=False)

print("Step 3 datasets generated:")
print("- Sparse & noisy:", f"{feeder_name}_sparse_noisy.csv")
print("- Grid probed:", f"{feeder_name}_probed.csv")
print("- Covariance-based synthetic:", f"{feeder_name}_synthetic.csv")
