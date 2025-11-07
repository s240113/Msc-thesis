import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np

# ------------------------------
# PARAMETERS
# ------------------------------
n_timesteps = 24                  # hourly for one day
load_variation = (0.6, 1.4)       # scaling factor for realistic load fluctuations
power_factor = 0.95               # constant PF for Q calculation
sparsity_ratio = 0.5              # fraction of buses measured
noise_voltage_pct = 0.005         # 0.5% voltage noise
noise_P_pct = 0.02                # 2% active power noise
noise_Q_pct = 0.02                # 2% reactive power noise
zero_injection_ratio = 0.1        # fraction of buses as zero-injection
grid_probe_magnitude = 0.01       # magnitude for probing perturbations
n_probe_buses = 5                 # number of buses to probe

# ------------------------------
# FUNCTIONS
# ------------------------------

def add_noise(series, pct):
    noise = np.random.normal(0, pct * series.abs(), size=len(series))
    return series + noise

def apply_grid_probing(df, buses_to_probe, magnitude):
    df_probe = df.copy()
    for bus in buses_to_probe:
        idx = df_probe['bus'] == bus
        df_probe.loc[idx, 'P_MW'] *= (1 + magnitude * np.random.choice([-1,1]))
    return df_probe

def synthetic_measurements_covariance(df_full, df_sparse):
    df_syn = df_sparse.copy()
    for col in ['voltage_pu','P_MW','Q_MVar']:
        nan_idx = df_syn[col].isna()
        mu = df_full[col].mean()
        sigma = df_full[col].std()
        df_syn.loc[nan_idx, col] = np.random.normal(mu, sigma, nan_idx.sum())
    return df_syn

def generate_scenarios(net, n_timesteps):
    results = {"timestep":[],"bus":[],"voltage_pu":[],"angle_deg":[],"P_MW":[],"Q_MVar":[]}
    for t in range(n_timesteps):
        # Scale loads randomly
        for load_idx in net.load.index:
            scale = np.random.uniform(load_variation[0], load_variation[1])
            net.load.at[load_idx,"p_mw"] = net.load.at[load_idx,"p_mw"] * scale
            net.load.at[load_idx,"q_mvar"] = net.load.at[load_idx,"p_mw"] * np.tan(np.arccos(power_factor))
        pp.runpp(net)
        for bus_idx in net.bus.index:
            results["timestep"].append(t)
            results["bus"].append(bus_idx)
            results["voltage_pu"].append(net.res_bus.vm_pu.at[bus_idx])
            results["angle_deg"].append(net.res_bus.va_degree.at[bus_idx])
            connected_loads = net.load[net.load.bus==bus_idx]
            P = connected_loads.p_mw.sum()
            Q = connected_loads.q_mvar.sum()
            results["P_MW"].append(P)
            results["Q_MVar"].append(Q)
    return pd.DataFrame(results)

# ------------------------------
# MAIN PIPELINE FOR IEEE 33
# ------------------------------

print("Generating synthetic scenarios for IEEE 33-bus feeder...")

# Load feeder
net = pn.example_simple()  # Replace with IEEE33 feeder if you have a dedicated version

# 1. Generate full AC power flow dataset
df_full = generate_scenarios(net, n_timesteps)
df_full['measurement_type'] = 'full'

# 2. Introduce sparse measurements and zero-injection nodes
all_buses = df_full['bus'].unique()
n_observed = int(len(all_buses) * sparsity_ratio)
observed_buses = np.random.choice(all_buses, n_observed, replace=False)

n_zero_inj = int(len(all_buses) * zero_injection_ratio)
zero_inj_buses = np.random.choice(all_buses, n_zero_inj, replace=False)

df_sparse = df_full.copy()
df_sparse.loc[~df_sparse['bus'].isin(observed_buses), ['voltage_pu','P_MW','Q_MVar']] = np.nan
df_sparse.loc[df_sparse['bus'].isin(zero_inj_buses), ['P_MW','Q_MVar']] = 0.0
df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'voltage_pu'] = add_noise(
    df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'voltage_pu'], noise_voltage_pct)
df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'P_MW'] = add_noise(
    df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'P_MW'], noise_P_pct)
df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'Q_MVar'] = add_noise(
    df_sparse.loc[df_sparse['bus'].isin(observed_buses), 'Q_MVar'], noise_Q_pct)
df_sparse['measurement_type'] = 'sparse_noisy'

# 3. Grid probing
probe_buses = np.random.choice(observed_buses, min(n_probe_buses, len(observed_buses)), replace=False)
df_probed = apply_grid_probing(df_sparse, probe_buses, grid_probe_magnitude)
df_probed['measurement_type'] = 'probed'

# 4. Covariance-based synthetic measurements
df_synthetic = synthetic_measurements_covariance(df_full, df_sparse)
df_synthetic['measurement_type'] = 'synthetic'

# 5. Combine all datasets
df_combined = pd.concat([df_full, df_sparse, df_probed, df_synthetic], ignore_index=True)

# 6. Save combined dataset
df_combined.to_csv("IEEE33_combined_dataset.csv", index=False)
print("Combined dataset saved as IEEE33_combined_dataset.csv")
