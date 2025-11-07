import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd

# ------------------------------
# Step 2: Synthetic Power Flow Scenario Generation
# ------------------------------

# 1. Load the feeder
feeder_name = "IEEE33"
if feeder_name == "IEEE13":
    # pandapower does not provide a built-in 13-bus example by this name.
    # Provide a custom feeder JSON or choose an available test case.
    raise NotImplementedError("IEEE13 network not available via pandapower.networks; provide a custom feeder or use an available test case.")
elif feeder_name == "IEEE33":
    # Use the available 33-bus Matpower-based case
    net = pn.case33bw()
elif feeder_name == "IEEE69":
    # pandapower does not provide a built-in 69-bus example by this name.
    raise NotImplementedError("IEEE69 network not available via pandapower.networks; provide a custom feeder or use an available test case.")
else:
    # No helper provided: require the user to supply a feeder loader or JSON
    raise NotImplementedError("No built-in network for this feeder name. Provide a custom feeder loader or use an available pandapower test case.")

# 2. Define simulation parameters
n_timesteps = 24  # e.g., hourly for one day
load_variation = 0.6, 1.4  # scaling factor range for realistic fluctuations
power_factor = 0.95  # assumed constant for Q calculation

# 3. Prepare storage for simulation results
results = {
    "bus": [],
    "timestep": [],
    "voltage_pu": [],
    "angle_deg": [],
    "P_MW": [],
    "Q_MVar": []
}

# 4. Generate time-varying load profiles
for t in range(n_timesteps):
    # Apply random load scaling to each load
    for load_idx in net.load.index:
        scale = np.random.uniform(load_variation[0], load_variation[1])
        net.load.at[load_idx, "p_mw"] = net.load.at[load_idx, "p_mw"] * scale
        net.load.at[load_idx, "q_mvar"] = net.load.at[load_idx, "p_mw"] * np.tan(np.arccos(power_factor))
    
    # 5. Run AC power flow
    pp.runpp(net)
    
    # 6. Store bus results
    for bus_idx in net.bus.index:
        results["bus"].append(bus_idx)
        results["timestep"].append(t)
        results["voltage_pu"].append(net.res_bus.vm_pu.at[bus_idx])
        results["angle_deg"].append(net.res_bus.va_degree.at[bus_idx])
        # Active/Reactive injection: sum loads connected to this bus
        connected_loads = net.load[net.load.bus == bus_idx]
        P = connected_loads.p_mw.sum()
        Q = connected_loads.q_mvar.sum()
        results["P_MW"].append(P)
        results["Q_MVar"].append(Q)

# 7. Convert results to DataFrame
df_results = pd.DataFrame(results)

# 8. Save dataset for later use in AI model
df_results.to_csv(f"synthetic_power_flow_{feeder_name}.csv", index=False)


print("Synthetic power flow scenario generation complete. Dataset saved.")
