import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa

# Create a new network for Australia (focusing on National Electricity Market region)
network = pypsa.Network()

# Set up snapshots for the year 2020
hours_in_2020 = pd.date_range('2020-01-01 00:00', '2020-12-31 23:00', freq='h')
network.set_snapshots(hours_in_2020)

# Add a single bus for the simplified model
network.add("Bus", "electricity bus")

print('The network has been created with the following snapshots:')
print(network.snapshots)

# Define annuity function (same as benchmark)
def annuity(n, r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate r"""
    if r > 0:
        return r/(1. - 1./(1+r)**n)
    else:
        return 1/n

# Create synthetic electricity demand data for Australia (NEM region)
# Based on typical Australian demand patterns with seasonal variation
# Australia's NEM annual consumption is approximately 180 TWh
annual_demand = 180e6  # 180 TWh in MWh

# Create base load with seasonal pattern (summer peak due to air conditioning)
# Australia's summer is December-February
day_of_year = hours_in_2020.dayofyear
# Shift phase to get peak in January
seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * ((day_of_year - 15) % 365) / 365)

# Daily pattern - peaks in morning and evening
hour_of_day = hours_in_2020.hour
daily_pattern = 1 + 0.2 * np.sin(np.pi * (hour_of_day - 8) / 12) + 0.3 * np.sin(np.pi * (hour_of_day - 18) / 6)

# Weekend effect
weekday = hours_in_2020.dayofweek
weekend_factor = np.where(weekday >= 5, 0.9, 1.0)

# Combine all factors
demand = annual_demand / 8760 * seasonal_factor * daily_pattern * weekend_factor
# Add noise
demand = demand * (0.95 + 0.1 * np.random.rand(len(demand)))

# Add load to the network
network.add("Load", "load", bus="electricity bus", p_set=demand)

print('The load time series has been added to the network:')
print(network.loads_t.p_set)

# Add carriers with respective CO2 emissions
network.add("Carrier", "onshorewind")
network.add("Carrier", "offshorewind")
network.add("Carrier", "solar")
network.add("Carrier", "coal", co2_emissions=0.9)  # in t_CO2/MWh_th
network.add("Carrier", "gas", co2_emissions=0.37)  # in t_CO2/MWh_th
network.add("Carrier", "hydro")
network.add("Carrier", "biomass", co2_emissions=0.23)  # in t_CO2/MWh_th

# Create capacity factors for renewable generators
# Australia has excellent solar and good wind resources

# Onshore wind capacity factors - moderate to high
cf_onshore = np.zeros(len(hours_in_2020))
for i, hour in enumerate(hours_in_2020):
    # Base value with seasonal variation (higher in winter months)
    month = hour.month
    if 5 <= month <= 8:  # Winter in Southern Hemisphere
        base = 0.45  # Higher in winter
    else:
        base = 0.35  # Lower in summer
    
    # Daily variation (slightly higher at night)
    hour_val = hour.hour
    if 18 <= hour_val or hour_val <= 6:
        time_factor = 1.1
    else:
        time_factor = 0.9
    
    # Add randomness
    random_factor = 0.8 + 0.4 * np.random.random()
    
    cf_onshore[i] = base * time_factor * random_factor

# Clip to valid range
cf_onshore = np.clip(cf_onshore, 0, 1)

# Solar capacity factors - excellent in Australia
cf_solar = np.zeros(len(hours_in_2020))
for i, hour in enumerate(hours_in_2020):
    # Base value with seasonal variation (higher in summer)
    month = hour.month
    if 11 <= month or month <= 2:  # Summer in Southern Hemisphere
        base = 0.3  # Higher in summer
    else:
        base = 0.2  # Lower in winter
    
    # Daytime only
    hour_val = hour.hour
    if 6 <= hour_val <= 18:
        # Bell curve for daylight hours
        time_factor = np.sin(np.pi * (hour_val - 6) / 12)
    else:
        time_factor = 0
    
    # Add randomness (less for solar - more predictable daily pattern)
    random_factor = 0.9 + 0.2 * np.random.random()
    
    cf_solar[i] = base * time_factor * random_factor

# Clip to valid range
cf_solar = np.clip(cf_solar, 0, 1)

# Offshore wind - higher and more consistent than onshore
cf_offshore = cf_onshore * 1.2
cf_offshore = np.clip(cf_offshore, 0, 1)

# Hydro - seasonal pattern with higher availability in wet season
cf_hydro = np.zeros(len(hours_in_2020))
for i, hour in enumerate(hours_in_2020):
    # Higher in Australian wet season (November-April)
    month = hour.month
    if month <= 4 or month >= 11:
        base = 0.5
    else:
        base = 0.3
    
    # Add randomness
    random_factor = 0.9 + 0.2 * np.random.random()
    
    cf_hydro[i] = base * random_factor

# Clip to valid range
cf_hydro = np.clip(cf_hydro, 0, 0.8)  # Lower max CF for hydro due to water constraints

# Cost data based on Australian CSIRO GenCost 2021-22 report
# https://www.csiro.au/-/media/EF/Files/GenCost2021-22.pdf
# All costs in AUD, converted to EUR at 0.6 EUR/AUD

# Onshore Wind
capital_cost_onshore = annuity(25, 0.07) * 1800000 * 0.6  # 1800 AUD/kW, converted to EUR
network.add("Generator",
            "onshorewind",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="onshorewind",
            p_nom_max=40000,  # 40 GW maximum potential
            capital_cost=capital_cost_onshore,
            marginal_cost=0,
            p_max_pu=cf_onshore)

# Offshore Wind
capital_cost_offshore = annuity(25, 0.07) * 3400000 * 0.6  # 3400 AUD/kW
network.add("Generator",
            "offshorewind",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="offshorewind",
            p_nom_max=10000,  # 10 GW maximum potential (limited continental shelf)
            capital_cost=capital_cost_offshore,
            marginal_cost=0,
            p_max_pu=cf_offshore)

# Solar PV
capital_cost_solar = annuity(25, 0.07) * 1060000 * 0.6  # 1060 AUD/kW
network.add("Generator",
            "solar",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="solar",
            p_nom_max=100000,  # 100 GW maximum potential
            capital_cost=capital_cost_solar,
            marginal_cost=0,
            p_max_pu=cf_solar)

# Coal Power Plants
fuel_cost_coal = 20 * 0.6  # AUD/MWh_th, converted to EUR
efficiency_coal = 0.37
capital_cost_coal = annuity(40, 0.07) * 3600000 * 0.6  # 3600 AUD/kW
network.add("Generator",
            "coal",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="coal",
            capital_cost=capital_cost_coal,
            marginal_cost=fuel_cost_coal/efficiency_coal,
            efficiency=efficiency_coal)

# CCGT (Combined Cycle Gas Turbine)
fuel_cost_gas = 30 * 0.6  # AUD/MWh_th
efficiency_ccgt = 0.55
capital_cost_ccgt = annuity(25, 0.07) * 1500000 * 0.6  # 1500 AUD/kW
network.add("Generator",
            "ccgt",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="gas",
            capital_cost=capital_cost_ccgt,
            marginal_cost=fuel_cost_gas/efficiency_ccgt,
            efficiency=efficiency_ccgt)

# OCGT (Open Cycle Gas Turbine)
efficiency_ocgt = 0.35
capital_cost_ocgt = annuity(25, 0.07) * 1000000 * 0.6  # 1000 AUD/kW
network.add("Generator",
            "ocgt",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="gas",
            capital_cost=capital_cost_ocgt,
            marginal_cost=fuel_cost_gas/efficiency_ocgt,
            efficiency=efficiency_ocgt)

# Hydro (mostly fixed capacity in Australia)
capital_cost_hydro = 0  # Existing capacity
network.add("Generator",
            "hydro",
            bus="electricity bus",
            p_nom=8000,  # 8 GW existing capacity
            p_nom_extendable=False,
            carrier="hydro",
            capital_cost=capital_cost_hydro,
            marginal_cost=0,
            p_max_pu=cf_hydro)

# Biomass
fuel_cost_biomass = 50 * 0.6  # AUD/MWh_th
efficiency_biomass = 0.35
capital_cost_biomass = annuity(25, 0.07) * 2500000 * 0.6  # 2500 AUD/kW
network.add("Generator",
            "biomass",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="biomass",
            p_nom_max=5000,  # Limited by resource availability
            capital_cost=capital_cost_biomass,
            marginal_cost=fuel_cost_biomass/efficiency_biomass,
            efficiency=efficiency_biomass)

# Battery Storage
capital_cost_battery_power = annuity(15, 0.07) * 600000 * 0.6  # 600 AUD/kW
capital_cost_battery_energy = annuity(15, 0.07) * 200000 * 0.6  # 200 AUD/kWh
network.add("StorageUnit",
            "battery",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="battery",
            capital_cost=capital_cost_battery_power + capital_cost_battery_energy * 4,  # 4 hour storage
            efficiency_store=0.9,
            efficiency_dispatch=0.9,
            cyclic_state_of_charge=True,
            max_hours=4)  # 4 hours of storage

# Add pumped hydro storage
capital_cost_phs = annuity(60, 0.07) * 2000000 * 0.6  # 2000 AUD/kW
network.add("StorageUnit",
            "pumped_hydro",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="pumped_hydro",
            capital_cost=capital_cost_phs,
            efficiency_store=0.75,
            efficiency_dispatch=0.75,
            cyclic_state_of_charge=True,
            max_hours=12)  # 12 hours of storage capacity

# Check that generator capacity factors were loaded correctly
print('The generator capacity factor time series have been added to the network:')
print(network.generators_t.p_max_pu)

# Add CO2 constraint
co2_limit = 50e6  # 50 million tonnes CO2
network.add("GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit)

# Optimize the system
print("Starting optimization...")
network.optimize(solver_name='gurobi')  
print("Optimization complete.")

# Print results
print("Total system cost:", network.objective/1e9, "billion EUR")
print("Cost per MWh:", network.objective/network.loads_t.p.sum().sum(), "EUR/MWh")

# Print optimal capacities
print("\nOptimal generation capacities (MW):")
for gen in network.generators.index:
    if network.generators.loc[gen, "p_nom_extendable"]:
        capacity = network.generators.loc[gen, "p_nom_opt"]
        print(f"{gen}: {capacity:.2f}")
    else:
        capacity = network.generators.loc[gen, "p_nom"]
        print(f"{gen}: {capacity:.2f} (fixed)")

# Print storage capacities
print("\nStorage capacities (MW):")
for storage in network.storage_units.index:
    capacity = network.storage_units.loc[storage, "p_nom_opt"]
    hours = network.storage_units.loc[storage, "max_hours"]
    energy_capacity = capacity * hours
    print(f"{storage}: {capacity:.2f} MW, {energy_capacity:.2f} MWh")

# Calculate annual generation by technology
annual_generation = {}
for gen in network.generators.index:
    annual_generation[gen] = network.generators_t.p[gen].sum() / 1e6  # TWh

# Add storage generation
for storage in network.storage_units.index:
    annual_generation[storage] = network.storage_units_t.p_dispatch[storage].sum() / 1e6  # TWh

total_demand = network.loads_t.p.sum().sum() / 1e6  # TWh
print(f"\nTotal annual demand: {total_demand:.2f} TWh")

print("\nAnnual generation by technology (TWh):")
for tech, gen in annual_generation.items():
    percent = gen / total_demand * 100
    print(f"{tech}: {gen:.2f} TWh ({percent:.1f}%)")

# Calculate capacity factors
print("\nCapacity factors:")
for gen in network.generators.index:
    if network.generators.loc[gen, "p_nom_extendable"]:
        capacity = network.generators.loc[gen, "p_nom_opt"]
    else:
        capacity = network.generators.loc[gen, "p_nom"]
    
    if capacity > 0:
        cf = network.generators_t.p[gen].mean() / capacity
        print(f"{gen}: {cf:.2f}")

# Plot dispatch for a summer week (January in Australia)
summer_week = slice("2020-01-15", "2020-01-21")

plt.figure(figsize=(14, 8))
# Start with storage dispatch (can be negative)
storage_data = {}
for storage in network.storage_units.index:
    storage_data[storage] = network.storage_units_t.p_dispatch[storage][summer_week] - \
                            network.storage_units_t.p_store[storage][summer_week]

# Get generation data
gen_data = {}
for gen in network.generators.index:
    gen_data[gen] = network.generators_t.p[gen][summer_week]

# Create dataframe for plotting
plot_data = pd.DataFrame(index=network.snapshots[summer_week])
for gen in gen_data:
    plot_data[gen] = gen_data[gen]
for storage in storage_data:
    plot_data[storage] = storage_data[storage]

# Plot stacked area for generation
# First, we need to create a version with only positive values for stacking
pos_data = plot_data.copy()
for col in pos_data.columns:
    pos_data.loc[pos_data[col] < 0, col] = 0

# Colors for each technology
colors = {
    'onshorewind': '#4ECDC4',  # Teal
    'offshorewind': '#1A535C',  # Dark teal
    'solar': '#FFE66D',  # Yellow
    'coal': '#696969',  # Gray
    'ccgt': '#FF6B6B',  # Red
    'ocgt': '#F38181',  # Light red
    'hydro': '#1E88E5',  # Blue
    'biomass': '#43A047',  # Green
    'battery': '#7E57C2',  # Purple
    'pumped_hydro': '#26C6DA'  # Cyan
}

# Plot positive generation as stacked area
plt.stackplot(pos_data.index, 
              [pos_data[col] for col in pos_data.columns],
              labels=pos_data.columns,
              colors=[colors.get(col, '#000000') for col in pos_data.columns])

# Plot negative values (storage charging) as lines
for col in plot_data.columns:
    neg_data = plot_data[col].copy()
    neg_data[neg_data >= 0] = 0
    if (neg_data < 0).any():
        plt.plot(neg_data.index, neg_data, color=colors.get(col, '#000000'), linestyle='--')

# Plot demand as line
plt.plot(network.loads_t.p["load"][summer_week].index, 
         network.loads_t.p["load"][summer_week].values,
         color='black', linewidth=2, label='Demand')

plt.title('Australia NEM - Electricity Dispatch in Summer Week (January 15-21)')
plt.xlabel('Date')
plt.ylabel('Power (MW)')
plt.grid(True, alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('australia_summer_dispatch.png')
plt.close()

# Plot dispatch for a winter week (July in Australia)
winter_week = slice("2020-07-15", "2020-07-21")

plt.figure(figsize=(14, 8))
# Create dataframe for plotting
plot_data = pd.DataFrame(index=network.snapshots[winter_week])
for gen in network.generators.index:
    plot_data[gen] = network.generators_t.p[gen][winter_week]
for storage in network.storage_units.index:
    plot_data[storage] = network.storage_units_t.p_dispatch[storage][winter_week] - \
                         network.storage_units_t.p_store[storage][winter_week]

# First, we need to create a version with only positive values for stacking
pos_data = plot_data.copy()
for col in pos_data.columns:
    pos_data.loc[pos_data[col] < 0, col] = 0

# Plot positive generation as stacked area
plt.stackplot(pos_data.index, 
              [pos_data[col] for col in pos_data.columns],
              labels=pos_data.columns,
              colors=[colors.get(col, '#000000') for col in pos_data.columns])

# Plot negative values (storage charging) as lines
for col in plot_data.columns:
    neg_data = plot_data[col].copy()
    neg_data[neg_data >= 0] = 0
    if (neg_data < 0).any():
        plt.plot(neg_data.index, neg_data, color=colors.get(col, '#000000'), linestyle='--')

# Plot demand as line
plt.plot(network.loads_t.p["load"][winter_week].index, 
         network.loads_t.p["load"][winter_week].values,
         color='black', linewidth=2, label='Demand')

plt.title('Australia NEM - Electricity Dispatch in Winter Week (July 15-21)')
plt.xlabel('Date')
plt.ylabel('Power (MW)')
plt.grid(True, alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('australia_winter_dispatch.png')
plt.close()

# Plot electricity mix (pie chart)
plt.figure(figsize=(10, 8))
gen_mix = {}
for gen in network.generators.index:
    gen_mix[gen] = network.generators_t.p[gen].sum()

# Add storage generation (only dispatch)
for storage in network.storage_units.index:
    gen_mix[storage] = network.storage_units_t.p_dispatch[storage].sum()

# Create pie chart
gen_labels = list(gen_mix.keys())
gen_values = [gen_mix[tech]/1e6 for tech in gen_labels]  # Convert to TWh
gen_colors = [colors.get(tech, '#000000') for tech in gen_labels]

plt.pie(gen_values, labels=gen_labels, colors=gen_colors, 
        autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Annual Electricity Generation Mix (TWh)')
plt.tight_layout()
plt.savefig('australia_electricity_mix.png')
plt.close()

# Create duration curves for each generator
plt.figure(figsize=(12, 8))
for gen in network.generators.index:
    duration_curve = network.generators_t.p[gen].sort_values(ascending=False).values
    plt.plot(range(len(duration_curve)), duration_curve, label=gen, color=colors.get(gen, '#000000'))

plt.title('Generator Duration Curves')
plt.xlabel('Hours')
plt.ylabel('Power (MW)')
plt.grid(True, alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('australia_duration_curves.png')
plt.close()

# Plot monthly capacity factors
plt.figure(figsize=(12, 8))
monthly_cf = {}
for gen in network.generators.index:
    if network.generators.loc[gen, "p_nom_extendable"]:
        capacity = network.generators.loc[gen, "p_nom_opt"]
    else:
        capacity = network.generators.loc[gen, "p_nom"]
    
    if capacity > 0:
        monthly_cf[gen] = []
        for month in range(1, 13):
            month_data = network.generators_t.p[gen][network.snapshots.month == month]
            if not month_data.empty:
                cf = month_data.mean() / capacity
                monthly_cf[gen].append(cf)

# Create a DataFrame for plotting
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
cf_df = pd.DataFrame(monthly_cf, index=months)

# Plot
for gen in cf_df.columns:
    plt.plot(cf_df.index, cf_df[gen], marker='o', label=gen, color=colors.get(gen, '#000000'))

plt.title('Monthly Average Capacity Factors')
plt.xlabel('Month')
plt.ylabel('Capacity Factor')
plt.grid(True, alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('australia_monthly_capacity_factors.png')
plt.close()

print("\nAll plots have been saved to files.")