import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import seaborn as sns

# Function to create and solve a model with a specific CO2 constraint
def solve_with_co2_constraint(co2_limit=None, include_storage=True):
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
        
        # Add hydrogen storage system (for seasonal storage)
        h2_capital_cost_electrolyzer = annuity(25, 0.07)*600000*(1+0.05)  # €/MW for electrolyzer
        h2_capital_cost_fuel_cell = annuity(10, 0.07)*1300000*(1+0.05) # €/MW for fuel cell
        h2_capital_cost_storage = annuity(25, 0.07)*57000*(1+0.011)  # €/MWh for storage
        
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

    # Add CO2 constraint if specified
    if co2_limit is not None:
        network.add("GlobalConstraint",
                   "co2_limit",
                   type="primary_energy",
                   carrier_attribute="co2_emissions",
                   sense="<=",
                   constant=co2_limit)

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

# Run the model for each constraint with and without storage
results_no_storage = []
results_with_storage = []

for constraint in co2_constraints:
    constraint_label = "No limit" if constraint is None else f"{constraint/1000000:.1f}M tons"
    print(f"Solving with CO2 constraint: {constraint_label}")
    
    # Solve without storage
    print("  Without storage...")
    result_no_storage, network_no_storage = solve_with_co2_constraint(constraint, include_storage=False)
    result_no_storage['constraint'] = constraint
    result_no_storage['constraint_label'] = constraint_label
    results_no_storage.append(result_no_storage)
    
    # Solve with storage
    print("  With storage...")
    result_with_storage, network_with_storage = solve_with_co2_constraint(constraint, include_storage=True)
    result_with_storage['constraint'] = constraint
    result_with_storage['constraint_label'] = constraint_label
    results_with_storage.append(result_with_storage)
    
    print(f"  Total emissions (no storage): {result_no_storage['total_emissions']/1000000:.2f}M tons CO2")
    print(f"  Total emissions (with storage): {result_with_storage['total_emissions']/1000000:.2f}M tons CO2")
    print(f"  System cost (no storage): {result_no_storage['objective']/1000000:.2f}M €")
    print(f"  System cost (with storage): {result_with_storage['objective']/1000000:.2f}M €")
    print(f"  Cost reduction with storage: {(1-result_with_storage['objective']/result_no_storage['objective'])*100:.2f}%")
    print()

# Create DataFrames for easier analysis
results_df_no_storage = pd.DataFrame(results_no_storage)
results_df_with_storage = pd.DataFrame(results_with_storage)

# Compare system costs
x_labels = results_df_no_storage['constraint_label'].tolist()
x_pos = np.arange(len(x_labels))

plt.figure(figsize=(14, 8))
cost_no_storage = [r['objective']/1000000 for r in results_no_storage]
cost_with_storage = [r['objective']/1000000 for r in results_with_storage]

plt.bar(x_pos - 0.2, cost_no_storage, 0.4, label='Without Storage', color='blue')
plt.bar(x_pos + 0.2, cost_with_storage, 0.4, label='With Storage', color='green')

# Add cost reduction percentage labels
for i in range(len(cost_no_storage)):
    reduction = (cost_no_storage[i] - cost_with_storage[i]) / cost_no_storage[i] * 100
    plt.text(x_pos[i], cost_with_storage[i] - 200, f"-{reduction:.1f}%", 
             ha='center', va='top', color='black', fontweight='bold')

plt.xlabel('CO2 Constraint', fontsize=14)
plt.ylabel('System Cost (Million €)', fontsize=14)
plt.title('Impact of Storage on System Costs', fontsize=16)
plt.xticks(x_pos, x_labels, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('storage_cost_comparison.png', dpi=300)
plt.show()

# Analyze storage capacity across different CO2 constraints
plt.figure(figsize=(14, 8))
battery_power = [r['capacities'].get('battery_power', 0)/1000 for r in results_with_storage]  # GW
battery_energy = [r['capacities'].get('battery_energy', 0)/1000 for r in results_with_storage]  # GWh
h2_power_in = [r['capacities'].get('h2_electrolyzer', 0)/1000 for r in results_with_storage]  # GW
h2_power_out = [r['capacities'].get('h2_fuel_cell', 0)/1000 for r in results_with_storage]  # GW
h2_energy = [r['capacities'].get('h2_storage', 0)/1000 for r in results_with_storage]  # GWh

ax1 = plt.subplot(111)
ax1.bar(x_pos - 0.3, battery_power, width=0.15, color='blue', label='Battery Power (GW)')
ax1.bar(x_pos - 0.15, battery_energy, width=0.15, color='skyblue', label='Battery Energy (GWh)')
ax1.bar(x_pos + 0.0, h2_power_in, width=0.15, color='green', label='H2 Electrolyzer (GW)')
ax1.bar(x_pos + 0.15, h2_power_out, width=0.15, color='lightgreen', label='H2 Fuel Cell (GW)')

ax2 = ax1.twinx()
ax2.bar(x_pos + 0.3, h2_energy, width=0.15, color='orange', label='H2 Storage (GWh)')

ax1.set_xlabel('CO2 Constraint', fontsize=14)
ax1.set_ylabel('Power Capacity (GW)', fontsize=14)
ax2.set_ylabel('H2 Storage Capacity (GWh)', fontsize=14, color='orange')
plt.title('Storage Capacity with Different CO2 Constraints', fontsize=16)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels, rotation=45)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('storage_capacity_comparison.png', dpi=300)
plt.show()

# Calculate renewable penetration
def calc_renewable_penetration(results_list):
    penetration = []
    for r in results_list:
        renewables = r['generation']['onshorewind'] + r['generation']['solar'] + r['generation']['hydro']
        total = sum(r['generation'].values())
        penetration.append(renewables / total * 100)
    return penetration

renewable_no_storage = calc_renewable_penetration(results_no_storage)
renewable_with_storage = calc_renewable_penetration(results_with_storage)

plt.figure(figsize=(12, 6))
plt.plot(x_pos, renewable_no_storage, 'bo-', label='Without Storage', linewidth=3)
plt.plot(x_pos, renewable_with_storage, 'go-', label='With Storage', linewidth=3)
plt.xlabel('CO2 Constraint', fontsize=14)
plt.ylabel('Renewable Energy Penetration (%)', fontsize=14)
plt.title('Impact of Storage on Renewable Energy Integration', fontsize=16)
plt.xticks(x_pos, x_labels, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('renewable_penetration.png', dpi=300)
plt.show()

# Analyze storage operation patterns for time balancing
# Let's pick a specific scenario for detailed analysis (e.g., 75% reduction, index 3)
scenario_index = 3  # 15Mt CO2 constraint
result, network = solve_with_co2_constraint(co2_constraints[scenario_index], include_storage=True)

# Create a DataFrame for a week in different seasons to analyze different balancing patterns
sample_weeks = [
    ('Winter', slice('2015-01-15', '2015-01-21')),
    ('Spring', slice('2015-04-15', '2015-04-21')),
    ('Summer', slice('2015-07-15', '2015-07-21')),
    ('Fall', slice('2015-10-15', '2015-10-21'))
]

for season, time_slice in sample_weeks:
    plt.figure(figsize=(16, 12))
    
    # Demand data
    demand = network.loads_t.p['load'][time_slice]
    timestamps = pd.to_datetime(demand.index)
    
    # Generation data
    solar_gen = network.generators_t.p['solar'][time_slice]
    wind_gen = network.generators_t.p['onshorewind'][time_slice]
    hydro_gen = network.generators_t.p['hydro'][time_slice]
    ocgt_gen = network.generators_t.p['OCGT'][time_slice] if 'OCGT' in network.generators_t.p else pd.Series(0, index=demand.index)
    biomass_gen = network.generators_t.p['biomass'][time_slice] if 'biomass' in network.generators_t.p else pd.Series(0, index=demand.index)
    
    # Storage operation
    battery_charge = -network.links_t.p0['battery_charger'][time_slice]  # Negative for visualization clarity
    battery_discharge = network.links_t.p1['battery_discharger'][time_slice]
    battery_soc = network.stores_t.e['battery_store'][time_slice] / network.stores.e_nom_opt['battery_store'] * 100  # SOC in %
    
    h2_charge = -network.links_t.p0['hydrogen_electrolyzer'][time_slice]  # Negative for clarity
    h2_discharge = network.links_t.p1['hydrogen_fuel_cell'][time_slice]
    h2_soc = network.stores_t.e['hydrogen_store'][time_slice] / network.stores.e_nom_opt['hydrogen_store'] * 100  # SOC in %
    
    # Panel 1: Generation mix and demand
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(timestamps, demand, 'k', linewidth=2, label='Demand')
    ax1.stackplot(timestamps, solar_gen, wind_gen, hydro_gen, biomass_gen, ocgt_gen,
                 labels=['Solar', 'Wind', 'Hydro', 'Biomass', 'Gas'],
                 colors=['gold', 'skyblue', 'purple', 'green', 'brown'])
    ax1.plot(timestamps, solar_gen + wind_gen + hydro_gen + biomass_gen + ocgt_gen + battery_discharge + h2_discharge,
            'r--', linewidth=1, label='Total Generation')
    ax1.set_ylabel('Power (MW)', fontsize=12)
    ax1.set_title(f'{season} Week Generation Mix and Storage Operation (CO2 limit: {x_labels[scenario_index]})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Panel 2: Battery operation
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.bar(timestamps, battery_charge, width=0.01, color='red', label='Battery Charging')
    ax2.bar(timestamps, battery_discharge, width=0.01, color='green', label='Battery Discharging')
    
    ax2b = ax2.twinx()
    ax2b.plot(timestamps, battery_soc, 'b-', linewidth=2, label='Battery SoC (%)')
    ax2.set_ylabel('Power (MW)', fontsize=12)
    ax2b.set_ylabel('State of Charge (%)', fontsize=12, color='b')
    ax2.set_title('Battery Storage Operation (Intraday Balancing)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends from both battery axes
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labels2 + labels2b, loc='upper left', fontsize=10)
    
    # Panel 3: Hydrogen operation
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.bar(timestamps, h2_charge, width=0.01, color='purple', label='H2 Production')
    ax3.bar(timestamps, h2_discharge, width=0.01, color='orange', label='H2 Consumption')
    
    ax3b = ax3.twinx()
    ax3b.plot(timestamps, h2_soc, 'g-', linewidth=2, label='H2 SoC (%)')
    ax3.set_ylabel('Power (MW)', fontsize=12)
    ax3b.set_ylabel('State of Charge (%)', fontsize=12, color='g')
    ax3.set_title('Hydrogen Storage Operation (Longer-term Balancing)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Combine legends from both hydrogen axes
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3b, labels3b = ax3b.get_legend_handles_labels()
    ax3.legend(lines3 + lines3b, labels3 + labels3b, loc='upper left', fontsize=10)
    
    # X-axis formatting
    ax3.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{season}_storage_operation.png', dpi=300, bbox_inches='tight')
    plt.show()

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
plt.title(f'Seasonal Storage Patterns (CO2 limit: {x_labels[scenario_index]})', fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('seasonal_storage_patterns.png', dpi=300)
plt.show()

# Calculate storage utilization/cycling
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

# Compare storage metrics across different CO2 constraints
storage_metrics = []
for i, constraint in enumerate(co2_constraints):
    _, network = solve_with_co2_constraint(constraint, include_storage=True)
    metrics = calculate_storage_metrics(network)
    metrics['constraint'] = x_labels[i]
    storage_metrics.append(metrics)

metrics_df = pd.DataFrame(storage_metrics)

# Plot storage cycles
plt.figure(figsize=(12, 6))
plt.bar(x_pos - 0.2, metrics_df['battery_cycles'], width=0.4, color='blue', label='Battery Cycles/Year')
plt.bar(x_pos + 0.2, metrics_df['h2_cycles'], width=0.4, color='green', label='H2 Storage Cycles/Year')
plt.xlabel('CO2 Constraint', fontsize=14)
plt.ylabel('Cycles per Year', fontsize=14)
plt.title('Storage Utilization Across CO2 Constraints', fontsize=16)
plt.xticks(x_pos, x_labels, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('storage_cycles.png', dpi=300)
plt.show()

# Create summary table comparing systems with and without storage
summary_data = []
for i, constraint in enumerate(co2_constraints):
    no_storage = results_no_storage[i]
    with_storage = results_with_storage[i]
    
    data = {
        'CO2 Constraint': x_labels[i],
        'System Cost w/o Storage (M€)': no_storage['objective']/1000000,
        'System Cost w/ Storage (M€)': with_storage['objective']/1000000,
        'Cost Reduction (%)': (1 - with_storage['objective']/no_storage['objective']) * 100,
        'RE Share w/o Storage (%)': calc_renewable_penetration([no_storage])[0],
        'RE Share w/ Storage (%)': calc_renewable_penetration([with_storage])[0],
        'Battery Power (GW)': with_storage['capacities'].get('battery_power', 0)/1000,
        'Battery Energy (GWh)': with_storage['capacities'].get('battery_energy', 0)/1000,
        'H2 Electrolyzer (GW)': with_storage['capacities'].get('h2_electrolyzer', 0)/1000,
        'H2 Fuel Cell (GW)': with_storage['capacities'].get('h2_fuel_cell', 0)/1000,
        'H2 Storage (GWh)': with_storage['capacities'].get('h2_storage', 0)/1000
    }
    summary_data.append(data)

summary_df = pd.DataFrame(summary_data)
print("\nSummary: Impact of Storage on Power System Configuration")
print("=" * 100)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Analysis of storage balancing strategies at different timescales
print("\nStorage Balancing Strategies Analysis:")
print("=" * 100)
print("1. Intraday Balancing (Battery Storage):")
print(f"   - Daily cycles: {metrics_df['battery_cycles'].mean()/365:.2f} cycles per day on average")
print(f"   - Primarily used for: Solar-peak shifting, demand-supply matching within 24-hour periods")
print("\n2. Weekly/Monthly Balancing (Combined Battery & H2):")
print(f"   - Weekly pattern: Battery handles weekday/weekend variations")
print(f"   - Monthly hydrogen SOC variation: {h2_monthly.std():.2f}% standard deviation")
print("\n3. Seasonal Balancing (Hydrogen Storage):")
print(f"   - Annual cycles: {metrics_df['h2_cycles'].mean():.2f} full cycles per year")
print(f"   - Highest SOC typically in: {'Spring/Summer'}")
print(f"   - Primarily used for: Long-term energy storage, seasonal load balancing")


# Analysis of storage balancing strategies at different timescales
print("\nStorage Balancing Strategies Analysis:")
print("=" * 100)
print("1. Intraday Balancing (Battery Storage):")
print(f"   - Daily cycles: {metrics_df['battery_cycles'].mean()/365:.2f} cycles per day on average")
print(f"   - Primarily used for: Solar-peak shifting, demand-supply matching within 24-hour periods")
print("\n2. Weekly/Monthly Balancing (Combined Battery & H2):")
print(f"   - Weekly pattern: Battery handles weekday/weekend variations")
print(f"   - Monthly hydrogen SOC variation: {h2_monthly.std():.2f}% standard deviation")
print("\n3. Seasonal Balancing (Hydrogen Storage):")
print(f"   - Annual cycles: {metrics_df['h2_cycles'].mean():.2f} full cycles per year")
print(f"   - Highest SOC typically in: {months[h2_monthly.argmax()]} (excess renewable generation)")
print(f"   - Lowest SOC typically in: {months[h2_monthly.argmin()]} (higher demand/lower renewable generation)")

# Analyze correlation between storage operation and renewable generation
# Let's examine this for the most restrictive CO2 scenario (near zero emissions)
_, network_zero_emission = solve_with_co2_constraint(co2_constraints[-1], include_storage=True)

# Create a daily dataframe with key variables
daily_data = pd.DataFrame({
    'solar': network_zero_emission.generators_t.p['solar'].resample('D').sum(),
    'wind': network_zero_emission.generators_t.p['onshorewind'].resample('D').sum(),
    'demand': network_zero_emission.loads_t.p['load'].resample('D').sum(),
    'battery_charge': network_zero_emission.links_t.p0['battery_charger'].resample('D').sum(),
    'h2_charge': network_zero_emission.links_t.p0['hydrogen_electrolyzer'].resample('D').sum(),
    'battery_discharge': network_zero_emission.links_t.p1['battery_discharger'].resample('D').sum(),
    'h2_discharge': network_zero_emission.links_t.p1['hydrogen_fuel_cell'].resample('D').sum()
})

# Calculate renewable surplus/deficit
daily_data['renewable_gen'] = daily_data['solar'] + daily_data['wind'] + network_zero_emission.generators_t.p['hydro'].resample('D').sum()
daily_data['net_load'] = daily_data['demand'] - daily_data['renewable_gen']

# Calculate correlation matrix
correlation = daily_data[['solar', 'wind', 'demand', 'net_load', 'battery_charge', 'h2_charge', 'battery_discharge', 'h2_discharge']].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation between Renewable Generation and Storage Operation', fontsize=16)
plt.tight_layout()
plt.savefig('storage_correlation_heatmap.png', dpi=300)
plt.show()

# Analyze average daily profiles by season
def plot_daily_profile_by_season(network, title):
    # Extract hourly data for a full year
    hourly_data = pd.DataFrame({
        'hour': pd.date_range('2015-01-01', periods=8760, freq='H').hour,
        'month': pd.date_range('2015-01-01', periods=8760, freq='H').month,
        'demand': network.loads_t.p['load'].values,
        'solar': network.generators_t.p['solar'].values,
        'wind': network.generators_t.p['onshorewind'].values,
        'battery_charge': -network.links_t.p0['battery_charger'].values,
        'battery_discharge': network.links_t.p1['battery_discharger'].values,
        'h2_charge': -network.links_t.p0['hydrogen_electrolyzer'].values,
        'h2_discharge': network.links_t.p1['hydrogen_fuel_cell'].values
    })
    
    # Define seasons
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    
    # Filter by season
    winter_data = hourly_data[hourly_data['month'].isin(winter_months)].groupby('hour').mean()
    spring_data = hourly_data[hourly_data['month'].isin(spring_months)].groupby('hour').mean()
    summer_data = hourly_data[hourly_data['month'].isin(summer_months)].groupby('hour').mean()
    fall_data = hourly_data[hourly_data['month'].isin(fall_months)].groupby('hour').mean()
    
    # Plot by season
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    axes = axes.flatten()
    
    # Hours for x-axis
    hours = np.arange(24)
    
    # Plot data for each season
    seasons = [(winter_data, 'Winter', 0), 
               (spring_data, 'Spring', 1), 
               (summer_data, 'Summer', 2), 
               (fall_data, 'Fall', 3)]
    
    for data, season, i in seasons:
        ax = axes[i]
        
        # Plot generation and demand
        ax.plot(hours, data['demand'], 'k-', linewidth=3, label='Demand')
        ax.plot(hours, data['solar'], 'orange', linewidth=2, label='Solar')
        ax.plot(hours, data['wind'], 'blue', linewidth=2, label='Wind')
        
        # Plot storage operation
        ax.plot(hours, data['battery_charge'], 'r--', linewidth=2, label='Battery Charging')
        ax.plot(hours, data['battery_discharge'], 'g--', linewidth=2, label='Battery Discharging')
        ax.plot(hours, data['h2_charge'], 'r:', linewidth=2, label='H2 Production')
        ax.plot(hours, data['h2_discharge'], 'g:', linewidth=2, label='H2 Consumption')
        
        ax.set_title(f'{season} - Average Daily Profile', fontsize=14)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Power (MW)', fontsize=12)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper left', fontsize=10)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('seasonal_daily_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot seasonal daily profiles for the most restrictive CO2 scenario
plot_daily_profile_by_season(network_zero_emission, 'Seasonal Daily Profiles with Storage - Near Zero CO2 Emissions')

# Analyze storage duration characteristics
def analyze_storage_duration(network):
    # Extract battery state of charge
    battery_soc = network.stores_t.e['battery_store']
    battery_capacity = network.stores.e_nom_opt['battery_store']
    
    # Extract hydrogen state of charge
    h2_soc = network.stores_t.e['hydrogen_store']
    h2_capacity = network.stores.e_nom_opt['hydrogen_store']
    
    # Calculate battery power capacity
    battery_power = network.links.p_nom_opt['battery_charger']
    
    # Calculate hydrogen power capacity (using electrolyzer)
    h2_power = network.links.p_nom_opt['hydrogen_electrolyzer']
    
    # Calculate storage duration in hours
    battery_duration = battery_capacity / battery_power if battery_power > 0 else 0
    h2_duration = h2_capacity / h2_power if h2_power > 0 else 0
    
    # Calculate storage utilization
    battery_utilization = (battery_soc.max() - battery_soc.min()) / battery_capacity if battery_capacity > 0 else 0
    h2_utilization = (h2_soc.max() - h2_soc.min()) / h2_capacity if h2_capacity > 0 else 0
    
    # Calculate longest continuous discharge period
    def find_longest_discharge(storage_data, threshold=0.01):
        discharge_periods = []
        current_period = 0
        
        # Calculate discharge rate (negative = discharge)
        discharge_rate = -storage_data.diff()
        
        for rate in discharge_rate:
            if rate > threshold:  # Discharging
                current_period += 1
            else:
                if current_period > 0:
                    discharge_periods.append(current_period)
                current_period = 0
                
        if current_period > 0:
            discharge_periods.append(current_period)
            
        return max(discharge_periods) if discharge_periods else 0
    
    battery_longest_discharge = find_longest_discharge(battery_soc)
    h2_longest_discharge = find_longest_discharge(h2_soc)
    
    return {
        'battery_duration': battery_duration,  # Hours
        'h2_duration': h2_duration,  # Hours
        'battery_utilization': battery_utilization * 100,  # Percentage
        'h2_utilization': h2_utilization * 100,  # Percentage
        'battery_longest_discharge': battery_longest_discharge,  # Hours
        'h2_longest_discharge': h2_longest_discharge  # Hours
    }

# Analyze storage duration characteristics for different CO2 constraints
duration_metrics = []
for i, constraint in enumerate(co2_constraints):
    _, network = solve_with_co2_constraint(constraint, include_storage=True)
    metrics = analyze_storage_duration(network)
    metrics['constraint'] = x_labels[i]
    duration_metrics.append(metrics)

duration_df = pd.DataFrame(duration_metrics)

# Plot storage duration characteristics
plt.figure(figsize=(14, 8))
bar_width = 0.35

plt.bar(x_pos - bar_width/2, duration_df['battery_duration'], bar_width, color='blue', label='Battery Duration (hours)')
plt.bar(x_pos + bar_width/2, duration_df['h2_duration']/24, bar_width, color='green', label='H2 Duration (days)')

plt.xlabel('CO2 Constraint', fontsize=14)
plt.ylabel('Storage Duration', fontsize=14)
plt.title('Storage Duration Characteristics Across CO2 Constraints', fontsize=16)
plt.xticks(x_pos, x_labels, rotation=45)

# Add secondary y-axis for a second metric if desired
ax2 = plt.twinx()
ax2.plot(x_pos, duration_df['battery_utilization'], 'bo-', label='Battery Utilization (%)')
ax2.plot(x_pos, duration_df['h2_utilization'], 'go-', label='H2 Utilization (%)')
ax2.set_ylabel('Storage Utilization (%)', fontsize=14)

# Combine legends
lines1, labels1 = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('storage_duration_characteristics.png', dpi=300)
plt.show()

# Create a comprehensive summary table
summary_table = {
    'CO2 Constraint': x_labels,
    'System Cost with Storage (M€)': [r['objective']/1000000 for r in results_with_storage],
    'Cost Reduction with Storage (%)': [(1 - results_with_storage[i]['objective']/results_no_storage[i]['objective']) * 100 for i in range(len(results_with_storage))],
    'Battery Power (GW)': [r['capacities'].get('battery_power', 0)/1000 for r in results_with_storage],
    'Battery Energy (GWh)': [r['capacities'].get('battery_energy', 0)/1000 for r in results_with_storage],
    'Battery Duration (hours)': duration_df['battery_duration'].values,
    'Battery Cycles/Year': metrics_df['battery_cycles'].values,
    'H2 Electrolyzer (GW)': [r['capacities'].get('h2_electrolyzer', 0)/1000 for r in results_with_storage],
    'H2 Fuel Cell (GW)': [r['capacities'].get('h2_fuel_cell', 0)/1000 for r in results_with_storage],
    'H2 Storage (GWh)': [r['capacities'].get('h2_storage', 0)/1000 for r in results_with_storage],
    'H2 Duration (days)': duration_df['h2_duration'].values/24,
    'H2 Cycles/Year': metrics_df['h2_cycles'].values,
    'Renewable Share (%)': calc_renewable_penetration(results_with_storage)
}

summary_table_df = pd.DataFrame(summary_table)
print("\nComprehensive Storage Analysis Summary:")
print("=" * 100)
print(summary_table_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("\nStorage Balancing Strategies:")
print("1. Intraday Balancing: Battery storage primarily handles daily solar/wind variations and peak shifting")
print("2. Weekly Balancing: Combined battery & hydrogen management for weekday/weekend patterns")
print("3. Seasonal Balancing: Hydrogen storage captures seasonal variations, storing excess renewable energy")
print("   for use during periods of low renewable generation (typically winter months)")