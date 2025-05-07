import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np

# Function to create and solve a model with a specific CO2 constraint
def solve_with_co2_constraint(co2_limit=None):
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
    
    # Calculate CO2 emissions
    results['emissions'] = {
        'gas': network.generators_t.p['OCGT'].sum() * network.carriers.loc['gas', 'co2_emissions']/efficiency,
        'biomass': network.generators_t.p['biomass'].sum() * network.carriers.loc['biomass', 'co2_emissions']/biomass_efficiency,
    }
    results['total_emissions'] = sum(results['emissions'].values())
    
    return results, network

# Define a range of CO2 constraints (in tons CO2)
# Spain's electricity sector emissions in 2019 were approximately 50 million tons
# In 1990 (Kyoto reference year) they were about 60 million tons
co2_constraints = [
    None,              # No constraint (cost optimization only)
    60000000,          # 1990 level (~60 million tons)
    50000000,          # 2019 level (~50 million tons)
    30000000,          # 50% reduction from 1990
    15000000,          # 75% reduction from 1990
    6000000,           # 90% reduction from 1990
    1000000,           # 98% reduction from 1990
    100000             # Near-zero emissions
]

# Run the model for each constraint and store results
results_list = []

for constraint in co2_constraints:
    constraint_label = "No limit" if constraint is None else f"{constraint/1000000:.1f}M tons"
    print(f"Solving with CO2 constraint: {constraint_label}")
    
    result, network = solve_with_co2_constraint(constraint)
    result['constraint'] = constraint
    result['constraint_label'] = constraint_label
    results_list.append(result)
    
    print(f"  Total emissions: {result['total_emissions']/1000000:.2f}M tons CO2")
    print(f"  Total cost: {result['objective']/1000000:.2f}M €")
    print(f"  Cost per MWh: {result['objective']/result['total_load']:.2f} €/MWh")
    print()

# Create DataFrame for easier analysis
results_df = pd.DataFrame(results_list)

# Calculate generation percentages
for tech in ['onshorewind', 'solar', 'biomass', 'OCGT', 'hydro']:
    results_df[f'{tech}_pct'] = results_df['generation'].apply(lambda x: x[tech]) / results_df['generation'].apply(lambda x: sum(x.values())) * 100

# Create stacked bar chart of generation mix
plt.figure(figsize=(14, 8))
bar_width = 0.7

# Extract constraint labels for x-axis
x_labels = results_df['constraint_label'].tolist()
x_pos = np.arange(len(x_labels))

# Get generation percentages
wind_pct = results_df['onshorewind_pct'].values
solar_pct = results_df['solar_pct'].values
biomass_pct = results_df['biomass_pct'].values
ocgt_pct = results_df['OCGT_pct'].values
hydro_pct = results_df['hydro_pct'].values

# Create stacked bars
plt.bar(x_pos, ocgt_pct, bar_width, label='Gas (OCGT)', color='brown')
plt.bar(x_pos, biomass_pct, bar_width, bottom=ocgt_pct, label='Biomass', color='green')
plt.bar(x_pos, hydro_pct, bar_width, bottom=ocgt_pct+biomass_pct, label='Hydro', color='purple')
plt.bar(x_pos, solar_pct, bar_width, bottom=ocgt_pct+biomass_pct+hydro_pct, label='Solar', color='orange')
plt.bar(x_pos, wind_pct, bar_width, bottom=ocgt_pct+biomass_pct+hydro_pct+solar_pct, label='Onshore Wind', color='blue')

# Add cost line on secondary axis
ax1 = plt.gca()
ax2 = ax1.twinx()
costs = [r['objective']/1000000 for r in results_list]
ax2.plot(x_pos, costs, 'ro-', linewidth=2, markersize=8, label='System Cost')

# Add labels and legend
ax1.set_xlabel('CO2 Constraint', fontsize=14)
ax1.set_ylabel('Generation Mix (%)', fontsize=14)
ax2.set_ylabel('System Cost (Million €)', fontsize=14, color='r')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels, rotation=45)
ax1.set_title('Impact of CO2 Constraints on Generation Mix and System Cost', fontsize=16)

# Add grid lines
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.tight_layout()
plt.savefig('generation_mix_vs_co2.png', dpi=300, bbox_inches='tight')
plt.show()

# Create capacity mix plot
plt.figure(figsize=(14, 8))

# Extract capacity values
wind_cap = [r['capacities']['onshorewind']/1000 for r in results_list]  # in GW
solar_cap = [r['capacities']['solar']/1000 for r in results_list]
biomass_cap = [r['capacities']['biomass']/1000 for r in results_list]
ocgt_cap = [r['capacities']['OCGT']/1000 for r in results_list]
hydro_cap = [r['capacities']['hydro']/1000 for r in results_list]

# Create stacked bars for capacity
plt.bar(x_pos, ocgt_cap, bar_width, label='Gas (OCGT)', color='brown')
plt.bar(x_pos, biomass_cap, bar_width, bottom=np.array(ocgt_cap), label='Biomass', color='green')
plt.bar(x_pos, hydro_cap, bar_width, bottom=np.array(ocgt_cap)+np.array(biomass_cap), label='Hydro', color='purple')
plt.bar(x_pos, solar_cap, bar_width, bottom=np.array(ocgt_cap)+np.array(biomass_cap)+np.array(hydro_cap), label='Solar', color='orange')
plt.bar(x_pos, wind_cap, bar_width, bottom=np.array(ocgt_cap)+np.array(biomass_cap)+np.array(hydro_cap)+np.array(solar_cap), label='Onshore Wind', color='blue')

# Calculate and plot total capacity
total_cap = np.array(wind_cap) + np.array(solar_cap) + np.array(biomass_cap) + np.array(ocgt_cap) + np.array(hydro_cap)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x_pos, total_cap, 'ro-', linewidth=2, markersize=8, label='Total Capacity')

# Add labels and legend
ax1.set_xlabel('CO2 Constraint', fontsize=14)
ax1.set_ylabel('Installed Capacity (GW)', fontsize=14)
ax2.set_ylabel('Total Capacity (GW)', fontsize=14, color='r')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels, rotation=45)
ax1.set_title('Impact of CO2 Constraints on Installed Capacity', fontsize=16)

# Add grid lines
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.tight_layout()
plt.savefig('capacity_mix_vs_co2.png', dpi=300, bbox_inches='tight')
plt.show()

# Adjust CO₂ labels if needed
co2_labels = ['Unlimited' if lbl == 'No limit' else lbl.replace('M', ' Mt') for lbl in x_labels]

summary_data = {
    'CO₂ Limit': co2_labels,
    'Emissions (Mt)': [r['total_emissions'] / 1e6 for r in results_list],
    'System Cost (M€)': [r['objective'] / 1e6 for r in results_list],
    'Cost/MWh (€)': [r['objective'] / r['total_load'] for r in results_list],
    'Wind (%)': results_df['onshorewind_pct'].values,
    'Solar (%)': results_df['solar_pct'].values,
    'Hydro (%)': results_df['hydro_pct'].values,
    'Biomass (%)': results_df['biomass_pct'].values,
    'Gas (%)': results_df['OCGT_pct'].values,
    'Wind (GW)': wind_cap,
    'Solar (GW)': solar_cap,
    'Hydro (GW)': hydro_cap,
    'Biomass (GW)': biomass_cap,
    'Gas (GW)': ocgt_cap,
    'Total Cap (GW)': total_cap
}

summary_df = pd.DataFrame(summary_data)

# Print it cleanly
print("\n Summary of Model Results")
print("=" * 100)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Add historical context
print("\nHistorical Context:")
print("- Spain's electricity sector emitted ~60 million tons CO2 in 1990")
print("- By 2019, emissions had decreased to ~50 million tons CO2")
print("- EU targets require 55% reduction from 1990 levels by 2030 (target: ~27 million tons)")
print("- Spain's climate neutrality target by 2050 requires near-zero emissions")

co2_limits = [r['constraint']/1000000 if r['constraint'] is not None else float('inf') for r in results_list]
system_costs = [r['objective']/1000000 for r in results_list]  # in million euros
cost_per_mwh = [r['objective']/r['total_load'] for r in results_list]  # in €/MWh
co2_emissions = [r['total_emissions']/1000000 for r in results_list]  # in million tons

# Create a figure with two subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Plot 1: System Cost vs CO2 Constraint
ax1.plot(x_pos, system_costs, 'bo-', linewidth=3, markersize=10)
ax1.set_ylabel('System Cost (Million €)', fontsize=14)
ax1.set_title('Impact of CO2 Constraints on System Cost', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)

# Annotate actual costs
for i, cost in enumerate(system_costs):
    ax1.annotate(f"{cost:.1f}M€", 
                 xy=(x_pos[i], cost),
                 xytext=(0, 10),
                 textcoords='offset points',
                 ha='center',
                 fontsize=11)

# Plot 2: Cost per MWh vs CO2 Constraint
ax2.plot(x_pos, cost_per_mwh, 'ro-', linewidth=3, markersize=10)
ax2.set_ylabel('Cost per MWh (€/MWh)', fontsize=14)
ax2.set_xlabel('CO2 Constraint', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

# Annotate cost per MWh
for i, cost in enumerate(cost_per_mwh):
    ax2.annotate(f"{cost:.1f} €/MWh", 
                 xy=(x_pos[i], cost),
                 xytext=(0, 10),
                 textcoords='offset points',
                 ha='center',
                 fontsize=11)

# Set x-ticks and labels
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels, rotation=45)

# Add a secondary x-axis showing the actual CO2 emissions
ax3 = ax2.twiny()
ax3.set_xlim(ax2.get_xlim())
# Set positions for the top ticks
ax3.set_xticks(x_pos)
# Set the actual emission values as labels
ax3.set_xticklabels([f"{emission:.1f}" for emission in co2_emissions], rotation=45)
ax3.set_xlabel('Actual CO2 Emissions (Million tons)', fontsize=14)

# Calculate percentage increase from baseline
baseline_cost = system_costs[0]
percentage_increase = [(cost/baseline_cost - 1) * 100 for cost in system_costs]

# Add text annotations for percentage increases
for i, pct in enumerate(percentage_increase):
    if i > 0:  # Skip the baseline
        ax1.annotate(f"+{pct:.1f}%", 
                     xy=(x_pos[i], system_costs[i]),
                     xytext=(0, -25),
                     textcoords='offset points',
                     ha='center',
                     fontsize=10,
                     color='darkblue')

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)  # Reduce space between subplots
plt.savefig('system_cost_vs_co2_constraint.png', dpi=300, bbox_inches='tight')
plt.show()

co2_constraints = [None, 60e6, 50e6, 30e6, 15e6, 6e6, 1e6, 0.1e6]
results = []

for constraint in co2_constraints:
    result, _ = solve_with_co2_constraint(constraint)  # Unpack the tuple
    results.append({
        "CO2_Limit": constraint,
        "System_Cost_MEUR": result["objective"] / 1e6,  # Convert to million €
        "Shadow_Price_CO2": result.get("shadow_price_co2", None),  # €/ton CO2 (if available)
        "Installed_Capacity_GW": sum(result["capacities"].values()) / 1e3,  # Total capacity in GW
        "CO2_Emissions_Mt": result["total_emissions"] / 1e6,  # Convert to million tons
        "Energy_Mix": result["generation"]
    })

# Convert to DataFrame
summary_df = pd.DataFrame(results)

plt.figure(figsize=(8, 6))
plt.plot(summary_df["CO2_Limit"] / 1e6, summary_df["Shadow_Price_CO2"], marker='o')
plt.xlabel("CO₂ Constraint (Mt)", fontsize=12)
plt.ylabel("Shadow Price of CO₂ (€ / ton)", fontsize=12)
plt.title("Shadow Prices of CO₂ Constraints", fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig("shadow_prices_co2_constraints.png", dpi=300)
plt.show()
