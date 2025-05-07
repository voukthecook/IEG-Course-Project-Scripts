import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load input data only once
df_elec = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/electricity_demand.csv', sep=';', index_col=0)
df_elec.index = pd.to_datetime(df_elec.index)
df_onshorewind = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)
df_solar = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index)

# Store results
years = range(1979, 2018)
gen_list = ["solar", "onshorewind",]
capacity_results = {gen: [] for gen in gen_list}
mean_output = {gen: [] for gen in gen_list}
std_output = {gen: [] for gen in gen_list}

country = 'ESP'

# Annuity function
def annuity(n, r):
    return r/(1. - 1./(1.+r)**n) if r > 0 else 1/n

for year in years:
    print(f"\nRunning year {year}...")
    network = pypsa.Network()

    snapshots = pd.date_range(f'{year}-01-01 00:00', f'{year}-12-31 23:00', freq='h').tz_localize(None)  # Ensure timezone-naive
    network.set_snapshots(snapshots)

    network.add("Bus", "electricity bus")

    # Add load
    load_series = df_elec[country].reindex(snapshots, fill_value=0)  # Reindex to match snapshots
    network.add("Load", "load", bus="electricity bus", p_set=load_series.values)

    # Add carriers
    network.add("Carrier", "onshorewind")
    network.add("Carrier", "solar")
  

    # Add wind generator
    CF_wind = df_onshorewind[country].reindex(snapshots, fill_value=0)  # Reindex to match snapshots
    network.add("Generator", "onshorewind", bus="electricity bus",
                p_nom_extendable=True, carrier="onshorewind",
                capital_cost=annuity(30,0.07)*910000*1.033,
                marginal_cost=0,
                p_max_pu=CF_wind.values)

    # Add solar generator
    CF_solar = df_solar[country].reindex(snapshots, fill_value=0)  # Reindex to match snapshots
    network.add("Generator", "solar", bus="electricity bus",
                p_nom_extendable=True, carrier="solar",
                capital_cost=annuity(25,0.07)*425000*1.03,
                marginal_cost=0,
                p_max_pu=CF_solar.values)

 

    # Solve
    try:
        network.optimize(solver_name='gurobi')
    except:
        print(f"Solver failed for {year}")
        continue

    # Save capacity and dispatch stats
    for gen in gen_list:
        if gen in network.generators.index:  # Check if the generator exists
            capacity_results[gen].append(network.generators.p_nom_opt[gen])
            p_series = network.generators_t.p[gen]
            mean_output[gen].append(p_series.mean())
            std_output[gen].append(p_series.std())
        else:
            capacity_results[gen].append(0)
            mean_output[gen].append(0)
            std_output[gen].append(0)

# Plotting results
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for gen in gen_list:
    axs[0].plot(years, capacity_results[gen], label=gen)
    axs[1].plot(years, std_output[gen], label=gen)

axs[0].set_ylabel("Optimal capacity (MW)")
axs[1].set_ylabel("Variability (Std Dev, MW)")
axs[1].set_xlabel("Year")
axs[0].legend()
axs[1].legend()
axs[0].set_title("Optimal capacity vs. weather year")
axs[1].set_title("Dispatch variability vs. weather year")

plt.tight_layout()
plt.show()

utilization = network.generators_t.p.sum() / (network.generators.p_nom_opt * 8760)
utilization.plot(kind='bar', title='Average Utilization Rate (Capacity Factor)', figsize=(10,5))
plt.ylabel('Utilization (0-1)')
plt.tight_layout()
plt.show()
