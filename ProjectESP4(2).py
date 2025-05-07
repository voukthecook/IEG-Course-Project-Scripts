import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import seaborn as sns

# Function to create and solve a model with a specific CO2 constraint and renewable target
def solve_with_co2_constraint(co2_limit=None, include_storage=True, min_renewable_share=None):
    # We start by creating the network as in the original code
    network = pypsa.Network()
    hours_in_2015 = pd.date_range('2015-01-01 00:00Z',
                                '2015-12-31 23:00Z',
                                freq='h')

    network.set_snapshots(hours_in_2015.values)
    network.add("Bus", "electricity bus")

    # Load electricity demand data (adjust path as necessary)
    df_elec = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/electricity_demand.csv', sep=';', index_col=0) # in MWh
    df_elec.index = pd.to_datetime(df_elec.index)
    country = 'ESP'

    # add load to the bus
    network.add("Load",
                "load",
                bus="electricity bus",
                p_set=df_elec[country].values)

    # Define annuity function
    def annuity(n, r):
        if r > 0:
            return r/(1. - 1./(1.+r)**n)
        else:
            return 1/n

    # Add carriers
    network.add("Carrier", "gas", co2_emissions=0.19)
    network.add("Carrier", "biomass", co2_emissions=0.0)
    network.add("Carrier", "onshorewind")
    network.add("Carrier", "solar")
    network.add("Carrier", "battery")
    network.add("Carrier", "hydrogen")
    network.add("Carrier", "hydro")

    # add onshore wind generator
    df_onshorewind = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/onshore_wind_1979-2017.csv', sep=';', index_col=0)
    df_onshorewind.index = pd.to_datetime(df_onshorewind.index)
    CF_wind = df_onshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    capital_cost_onshorewind = annuity(30,0.07)*910000*(1+0.033)
    network.add("Generator",
                "onshorewind",
                bus="electricity bus",
                p_nom_extendable=True,
                carrier="onshorewind",
                capital_cost = capital_cost_onshorewind,
                marginal_cost = 0,
                p_max_pu = CF_wind.values)

    # add solar PV generator
    df_solar = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/pv_optimal.csv', sep=';', index_col=0)
    df_solar.index = pd.to_datetime(df_solar.index)
    CF_solar = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    capital_cost_solar = annuity(25,0.07)*425000*(1+0.03)
    network.add("Generator",
                "solar",
                bus="electricity bus",
                p_nom_extendable=True,
                carrier="solar",
                capital_cost = capital_cost_solar,
                marginal_cost = 0,
                p_max_pu = CF_solar.values)

    # add OCGT generator
    capital_cost_OCGT = annuity(25,0.07)*560000*(1+0.033)
    fuel_cost = 25
    efficiency = 0.39
    marginal_cost_OCGT = fuel_cost/efficiency
    network.add("Generator",
                "OCGT",
                bus="electricity bus",
                p_nom_extendable=True,
                carrier="gas",
                capital_cost = capital_cost_OCGT,
                marginal_cost = marginal_cost_OCGT)

    # add Biomass generator
    capital_cost_biomass = annuity(25,0.07)*2500000*(1+0.033)
    biomass_fuel_cost = 30
    biomass_efficiency = 0.35
    marginal_cost_biomass = biomass_fuel_cost/biomass_efficiency
    biomass_p_nom_max = 5000

    network.add("Generator",
                "biomass",
                bus="electricity bus",
                p_nom_extendable=True,
                carrier="biomass",
                p_nom_max=biomass_p_nom_max,
                capital_cost=capital_cost_biomass,
                marginal_cost=marginal_cost_biomass)

    # add hydro generator
    capital_cost_hydro = annuity(30, 0.07) * 2000000
    hydro_p_nom_max = 1000

    network.add("Generator",
                "hydro",
                bus="electricity bus",
                p_nom_extendable=True,
                carrier="hydro",
                capital_cost=capital_cost_hydro,
                marginal_cost=0,
                p_nom_max=hydro_p_nom_max)
    
    # Add storage technologies
    if include_storage:
        # Add battery storage
        battery_capital_cost = annuity(25, 0.07)*400000*(1+0.02)  # €/MW for power capacity
        battery_capital_cost_e = annuity(25, 0.07)*400000*(1+0.02) # €/MWh for energy capacity
        
        network.add("Bus", "battery bus")
        network.add("Store", 
                   "battery_store",
                   bus="battery bus",
                   e_nom_extendable=True,
                   e_cyclic=True,
                   capital_cost=battery_capital_cost_e)
        
        network.add("Link",
                   "battery_charger",
                   bus0="electricity bus",
                   bus1="battery bus",
                   p_nom_extendable=True,
                   efficiency=0.95,
                   capital_cost=battery_capital_cost)
        
        network.add("Link",
                   "battery_discharger",
                   bus0="battery bus",
                   bus1="electricity bus",
                   p_nom_extendable=True,
                   efficiency=0.95,
                   capital_cost=0)  # Cost is already in charger
        
        # Add hydrogen storage system with reduced costs to encourage usage
        h2_capital_cost_electrolyzer = annuity(25, 0.07)*500000*(1+0.05)  # Reduced from 600000
        h2_capital_cost_fuel_cell = annuity(10, 0.07)*1000000*(1+0.05) # Reduced from 1300000
        h2_capital_cost_storage = annuity(25, 0.07)*50000*(1+0.011)  # Reduced from 57000
        
        network.add("Bus", "hydrogen bus")
        network.add("Store", 
                   "hydrogen_store",
                   bus="hydrogen bus",
                   e_nom_extendable=True,
                   e_cyclic=True,
                   capital_cost=h2_capital_cost_storage)
        
        network.add("Link",
                   "hydrogen_electrolyzer",
                   bus0="electricity bus",
                   bus1="hydrogen bus",
                   p_nom_extendable=True,
                   efficiency=0.8,  # electrolyzer efficiency
                   capital_cost=h2_capital_cost_electrolyzer)
        
        network.add("Link",
                   "hydrogen_fuel_cell",
                   bus0="hydrogen bus",
                   bus1="electricity bus",
                   p_nom_extendable=True,
                   efficiency=0.58,  # fuel cell efficiency
                   capital_cost=h2_capital_cost_fuel_cell)

    # Validate co2_limit
    if co2_limit is not None and not isinstance(co2_limit, (int, float)):
        raise ValueError(f"Invalid co2_limit value: {co2_limit}. It must be a numeric value or None.")

    # Add CO2 constraint if specified
    if co2_limit is not None:
        network.add("GlobalConstraint",
                   "co2_limit",
                   type="primary_energy",
                   carrier_attribute="co2_emissions",
                   sense="<=",
                   constant=co2_limit)
    
    # Add minimum renewable share constraint if specified
    if min_renewable_share is not None:
        # Create extra variables to track generation by carrier
        renewable_carriers = ["onshorewind", "solar", "hydro"]
        for carrier in renewable_carriers:
            network.add("Carrier", f"{carrier}", nice_name=f"{carrier}")
        
        # Calculate total demand
        total_demand = network.loads_t.p_set.sum().sum()
        
        # Add renewable generation constraint - this is a linear constraint
        # We constrain: sum(renewable_generation) >= min_renewable_share * total_demand
        renewable_gens = network.generators[network.generators.carrier.isin(renewable_carriers)].index
        weightdict = {gen: 1.0 for gen in renewable_gens}
        
        network.add(
            "GlobalConstraint",
            "min_renewable_share",
            type="generator_group",
            generator_weightdict=weightdict,
            sense=">=",
            constant=min_renewable_share * total_demand,
        )

    # Optimize
    network.optimize(solver_name='gurobi')

    # Extract results
    results = {
        'objective': network.objective,
        'total_load': network.loads_t.p.sum().sum(),
        'capacities': {
            'onshorewind': network.generators.p_nom_opt['onshorewind'],
            'solar': network.generators.p_nom_opt['solar'],
            'biomass': network.generators.p_nom_opt['biomass'],
            'OCGT': network.generators.p_nom_opt['OCGT'],
            'hydro': network.generators.p_nom_opt['hydro']
        },
        'generation': {
            'onshorewind': network.generators_t.p['onshorewind'].sum(),
            'solar': network.generators_t.p['solar'].sum(),
            'biomass': network.generators_t.p['biomass'].sum(),
            'OCGT': network.generators_t.p['OCGT'].sum(),
            'hydro': network.generators_t.p['hydro'].sum()
        }
    }
    
    # Add storage results if included
    if include_storage:
        # Battery storage
        results['capacities']['battery_power'] = network.links.p_nom_opt['battery_charger']
        results['capacities']['battery_energy'] = network.stores.e_nom_opt['battery_store']
        results['operation'] = {
            'battery_charge': network.links_t.p0['battery_charger'].sum(),
            'battery_discharge': network.links_t.p1['battery_discharger'].sum()
        }
        
        # Hydrogen storage
        results['capacities']['h2_electrolyzer'] = network.links.p_nom_opt['hydrogen_electrolyzer']
        results['capacities']['h2_fuel_cell'] = network.links.p_nom_opt['hydrogen_fuel_cell']
        results['capacities']['h2_storage'] = network.stores.e_nom_opt['hydrogen_store']
        results['operation']['h2_production'] = network.links_t.p0['hydrogen_electrolyzer'].sum()
        results['operation']['h2_consumption'] = network.links_t.p1['hydrogen_fuel_cell'].sum()
    
    # Calculate CO2 emissions
    results['emissions'] = {
        'gas': network.generators_t.p['OCGT'].sum() * network.carriers.loc['gas', 'co2_emissions']/efficiency,
        'biomass': network.generators_t.p['biomass'].sum() * network.carriers.loc['biomass', 'co2_emissions']/biomass_efficiency,
    }
    results['total_emissions'] = sum(results['emissions'].values())
    
    # Calculate actual renewable share
    renewable_gen = sum([results['generation'].get(carrier, 0) for carrier in ['onshorewind', 'solar', 'hydro']])
    results['renewable_share'] = renewable_gen / results['total_load'] * 100
    
    return results, network

# Define a range of CO2 constraints (in tons CO2)
co2_constraints = [
    None,              # No constraint (cost optimization only)
    60000000,          # 1990 level (~60 million tons)
    30000000,          # 50% reduction from 1990
    15000000,          # 75% reduction from 1990
    6000000,           # 90% reduction from 1990
    1000000            # Near-zero emissions
]

# Define a range of renewable penetration targets
renewable_targets = [
    None,      # No explicit target
    0.70,      # 70% renewable 
    0.80,      # 80% renewable
    0.90,      # 90% renewable
    0.95,      # 95% renewable
    0.98       # 98% renewable
]

# Select combinations to test
scenarios = [
    (co2_constraints[3], renewable_targets[0]),  # 75% CO2 reduction, no RE target
    (co2_constraints[3], renewable_targets[2]),  # 75% CO2 reduction, 80% RE target
    (co2_constraints[3], renewable_targets[3]),  # 75% CO2 reduction, 90% RE target
    (co2_constraints[3], renewable_targets[4]),  # 75% CO2 reduction, 95% RE target
    (co2_constraints[4], renewable_targets[0]),  # 90% CO2 reduction, no RE target
    (co2_constraints[4], renewable_targets[3]),  # 90% CO2 reduction, 90% RE target
    (co2_constraints[4], renewable_targets[4]),  # 90% CO2 reduction, 95% RE target
    (co2_constraints[5], renewable_targets[0]),  # Near-zero CO2, no RE target
    (co2_constraints[5], renewable_targets[4]),  # Near-zero CO2, 95% RE target
    (co2_constraints[5], renewable_targets[5]),  # Near-zero CO2, 98% RE target
]

# Run scenarios
results_list = []

for co2_limit, re_target in scenarios:
    co2_label = "No limit" if co2_limit is None else f"{co2_limit/1000000:.1f}M tons"
    re_label = "No target" if re_target is None else f"{re_target*100:.0f}%"
    scenario_label = f"CO2: {co2_label}, RE: {re_label}"
    
    print(f"Solving scenario: {scenario_label}")
    
    # Solve with storage and the specified constraints
    result, network = solve_with_co2_constraint(co2_limit, include_storage=True, min_renewable_share=re_target)
    
    # Save scenario information
    result['co2_constraint'] = co2_limit
    result['co2_label'] = co2_label
    result['re_target'] = re_target
    result['re_label'] = re_label
    result['scenario_label'] = scenario_label
    
    results_list.append(result)
    
    print(f"  Total emissions: {result['total_emissions']/1000000:.2f}M tons CO2")
    print(f"  System cost: {result['objective']/1000000:.2f}M €")
    print(f"  Renewable share: {result['renewable_share']:.2f}%")
    print(f"  Battery capacity: {result['capacities']['battery_energy']/1000:.2f} GWh")
    print(f"  H2 storage capacity: {result['capacities']['h2_storage']/1000:.2f} GWh")
    print()

# Create DataFrame for easier analysis
results_df = pd.DataFrame([
    {
        'Scenario': r['scenario_label'],
        'System Cost (M€)': r['objective']/1000000,
        'Renewable Share (%)': r['renewable_share'],
        'CO2 Emissions (Mt)': r['total_emissions']/1000000,
        'Battery Power (GW)': r['capacities']['battery_power']/1000,
        'Battery Energy (GWh)': r['capacities']['battery_energy']/1000,
        'H2 Electrolyzer (GW)': r['capacities']['h2_electrolyzer']/1000,
        'H2 Fuel Cell (GW)': r['capacities']['h2_fuel_cell']/1000,
        'H2 Storage (GWh)': r['capacities']['h2_storage']/1000,
        'Wind Capacity (GW)': r['capacities']['onshorewind']/1000,
        'Solar Capacity (GW)': r['capacities']['solar']/1000,
        'Gas Capacity (GW)': r['capacities']['OCGT']/1000
    } for r in results_list
])

# Display results table
print("Summary of Results:")
print("=" * 100)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Plot system components across scenarios
plt.figure(figsize=(15, 10))

# Select key metrics for plotting
scenarios = results_df['Scenario'].tolist()
x_pos = np.arange(len(scenarios))

# Prepare data for storage technologies
batt_power = results_df['Battery Power (GW)'].values
batt_energy = results_df['Battery Energy (GWh)'].values
h2_electrolyzer = results_df['H2 Electrolyzer (GW)'].values
h2_fuel_cell = results_df['H2 Fuel Cell (GW)'].values
h2_storage = results_df['H2 Storage (GWh)'].values

# Plot storage capacities
ax1 = plt.subplot(211)
ax1.bar(x_pos - 0.3, batt_power, width=0.15, color='blue', label='Battery Power (GW)')
ax1.bar(x_pos - 0.15, batt_energy, width=0.15, color='skyblue', label='Battery Energy (GWh)')
ax1.bar(x_pos + 0.0, h2_electrolyzer, width=0.15, color='green', label='H2 Electrolyzer (GW)')
ax1.bar(x_pos + 0.15, h2_fuel_cell, width=0.15, color='lightgreen', label='H2 Fuel Cell (GW)')

ax2 = ax1.twinx()
ax2.bar(x_pos + 0.3, h2_storage, width=0.15, color='orange', label='H2 Storage (GWh)')

ax1.set_ylabel('Power Capacity (GW) / Battery Energy (GWh)', fontsize=12)
ax2.set_ylabel('H2 Storage Capacity (GWh)', fontsize=12, color='orange')
ax1.set_title('Storage Capacity Across Different Scenarios', fontsize=14)
ax1.set_xticks([])

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Plot system costs and renewable shares
ax3 = plt.subplot(212)
costs = results_df['System Cost (M€)'].values
re_shares = results_df['Renewable Share (%)'].values

ax3.bar(x_pos, costs, color='purple', alpha=0.7, label='System Cost (M€)')

ax4 = ax3.twinx()
ax4.plot(x_pos, re_shares, 'go-', linewidth=3, markersize=8, label='Renewable Share (%)')

ax3.set_xlabel('Scenario', fontsize=14)
ax3.set_ylabel('System Cost (M€)', fontsize=12)
ax4.set_ylabel('Renewable Share (%)', fontsize=12, color='green')
ax3.set_title('System Cost and Renewable Penetration', fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(scenarios, rotation=45, ha='right')

# Combine legends
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')

plt.tight_layout()
plt.savefig('renewable_target_results.png', dpi=300)
plt.show()

# Function to calculate storage metrics
def calculate_storage_metrics(network):
    # Battery metrics
    battery_capacity = network.stores.e_nom_opt['battery_store']
    battery_charge_sum = network.links_t.p0['battery_charger'].sum()
    battery_cycles = battery_charge_sum / battery_capacity if battery_capacity > 0 else 0
    
    # H2 metrics
    h2_capacity = network.stores.e_nom_opt['hydrogen_store']
    h2_charge_sum = network.links_t.p0['hydrogen_electrolyzer'].sum()
    h2_cycles = h2_charge_sum / h2_capacity if h2_capacity > 0 else 0
    
    return {
        'battery_cycles': battery_cycles,
        'h2_cycles': h2_cycles,
        'battery_capacity': battery_capacity,
        'h2_capacity': h2_capacity
    }

# Choose one high renewable scenario for detailed analysis
high_re_scenario_index = -1  # Index of the desired scenario (last one in the list)
if not isinstance(scenarios[high_re_scenario_index], tuple) or len(scenarios[high_re_scenario_index]) != 2:
    raise ValueError(f"Invalid scenario at index {high_re_scenario_index}: {scenarios[high_re_scenario_index]}. It must be a tuple with two elements.")
high_re_scenario = scenarios[high_re_scenario_index]

# Validate high_re_scenario
if high_re_scenario[0] is not None and not isinstance(high_re_scenario[0], (int, float)):
    raise ValueError(f"Invalid CO2 limit in high_re_scenario: {high_re_scenario[0]}. It must be numeric or None.")
if high_re_scenario[1] is not None and not isinstance(high_re_scenario[1], (int, float)):
    raise ValueError(f"Invalid renewable target in high_re_scenario: {high_re_scenario[1]}. It must be numeric or None.")

# Run the model for this scenario
print(f"Running detailed analysis for scenario: CO2 limit = {high_re_scenario[0]}, Renewable target = {high_re_scenario[1]}")
result, network = solve_with_co2_constraint(high_re_scenario[0], include_storage=True, min_renewable_share=high_re_scenario[1])

# Calculate storage metrics
storage_metrics = calculate_storage_metrics(network)
print("\nStorage Utilization Metrics:")
print(f"Battery cycles per year: {storage_metrics['battery_cycles']:.2f}")
print(f"H2 storage cycles per year: {storage_metrics['h2_cycles']:.2f}")

# Analyze seasonal storage patterns - monthly average state of charge
full_year = network.stores_t.e
battery_monthly = full_year['battery_store'].resample('M').mean() / network.stores.e_nom_opt['battery_store'] * 100
h2_monthly = full_year['hydrogen_store'].resample('M').mean() / network.stores.e_nom_opt['hydrogen_store'] * 100

plt.figure(figsize=(14, 6))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.plot(months, battery_monthly.values, 'bo-', linewidth=3, label='Battery Average SoC')
plt.plot(months, h2_monthly.values, 'go-', linewidth=3, label='Hydrogen Average SoC')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average State of Charge (%)', fontsize=14)
plt.title(f'Seasonal Storage Patterns with {high_re_scenario[1]*100:.0f}% RE Target', fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('high_re_seasonal_storage.png', dpi=300)
plt.show()

# Analyze hydrogen storage operation for a week in winter and summer
sample_weeks = [
    ('Winter', slice('2015-01-15', '2015-01-21')),
    ('Summer', slice('2015-07-15', '2015-07-21'))
]

for season, time_slice in sample_weeks:
    plt.figure(figsize=(16, 12))
    
    # Demand data
    demand = network.loads_t.p['load'][time_slice]
    timestamps = pd.to_datetime(demand.index)
    
    # Generation data
    solar_gen = network.generators_t.p['solar'][time_slice]
    wind_gen = network.generators_t.p['onshorewind'][time_slice]
    
    # H2 storage operation
    h2_charge = -network.links_t.p0['hydrogen_electrolyzer'][time_slice]  # Negative for clarity
    h2_discharge = network.links_t.p1['hydrogen_fuel_cell'][time_slice]
    h2_soc = network.stores_t.e['hydrogen_store'][time_slice] / network.stores.e_nom_opt['hydrogen_store'] * 100  # SOC in %
    
    # Panel 1: Generation mix and demand
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(timestamps, demand, 'k', linewidth=2, label='Demand')
    ax1.stackplot(timestamps, solar_gen, wind_gen, 
                 labels=['Solar', 'Wind'],
                 colors=['gold', 'skyblue'])
    ax1.set_ylabel('Power (MW)', fontsize=12)
    ax1.set_title(f'{season} Week - Renewable Generation with {high_re_scenario[1]*100:.0f}% RE Target', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Panel 2: Hydrogen operation
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.bar(timestamps, h2_charge, width=0.01, color='purple', label='H2 Production')
    ax2.bar(timestamps, h2_discharge, width=0.01, color='orange', label='H2 Consumption')
    
    ax2b = ax2.twinx()
    ax2b.plot(timestamps, h2_soc, 'g-', linewidth=2, label='H2 SoC (%)')
    ax2.set_ylabel('Power (MW)', fontsize=12)
    ax2b.set_ylabel('State of Charge (%)', fontsize=12, color='g')
    ax2.set_title('Hydrogen Storage Operation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labels2 + labels2b, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'high_re_{season}_h2_operation.png', dpi=300)
    plt.show()