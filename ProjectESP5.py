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

    # Calculate baseline emissions if this is the first run with no constraint
    baseline_emissions = None
    if co2_limit is None:
        # Create a temporary copy to solve and find baseline emissions
        temp_network = network.copy()
        temp_network.optimize(
            solver_name="gurobi",
            solver_options={"method": 1, "crossover": 0, "threads": 4, "OutputFlag": 0}
        )
        
        # Calculate emissions from the unconstrained model
        baseline_emissions = temp_network.generators_t.p.sum().multiply(
            temp_network.generators.carrier.map(temp_network.carriers.co2_emissions)
        ).sum()
        
        print(f"Baseline CO2 emissions (no constraint): {baseline_emissions:.2f} tons")
    
    # Add CO2 constraint if specified
    # If no constraint is specified but we calculated baseline, use 99% of baseline
    # This ensures the constraint is binding but very loose
    if co2_limit is None and baseline_emissions is not None:
        co2_limit = baseline_emissions * 0.99
        print(f"Using 99% of baseline emissions as soft constraint: {co2_limit:.2f} tons")
    
    if co2_limit is not None:
        network.add("GlobalConstraint",
                   "co2_limit",
                   sense="<=",
                   constant=co2_limit,
                   type="primary_energy")

    # Solve the model with more detailed output
    print(f"Solving with CO2 constraint: {co2_limit}")
    network.optimize(
        solver_name="gurobi",
        solver_options={
            "method": 1, 
            "crossover": 0, 
            "threads": 4, 
            "OutputFlag": 1,  # Enable solver output for debugging
            "FeasibilityTol": 1e-6  # Tighten feasibility tolerance
        }
    )
    
    print("Solve completed.")

    # Extract shadow price (dual value) of CO2 constraint if it exists
    shadow_price = None
    if co2_limit is not None and 'co2_limit' in network.global_constraints.index:
        shadow_price = network.global_constraints.at['co2_limit', 'mu']
        print(f"Raw shadow price: {shadow_price}")
        
        # Adjust for numerical issues - shadow prices should be positive for <= constraints
        # when they are binding (negative means either wrong extraction or non-binding)
        if shadow_price is not None and shadow_price <= 0:
            # Calculate actual emissions to check if constraint is binding
            actual_emissions = network.generators_t.p.sum().multiply(
                network.generators.carrier.map(network.carriers.co2_emissions)
            ).sum()
            
            # If emissions are very close to the limit, constraint is binding
            if abs(actual_emissions - co2_limit) < co2_limit * 0.005:  # Within 0.5%
                # Run a sensitivity test by solving with slightly tighter constraint
                tighter_limit = co2_limit * 0.95  # 5% tighter
                print(f"Testing with 5% tighter constraint: {tighter_limit}")
                
                # Create a new network and manually copy components
                network_tight = pypsa.Network()
                network_tight.set_snapshots(network.snapshots)
                network_tight.buses = network.buses.copy()
                network_tight.loads = network.loads.copy()
                network_tight.generators = network.generators.copy()
                network_tight.carriers = network.carriers.copy()
                network_tight.global_constraints = network.global_constraints.copy()
                
                # Update the CO2 constraint for the tighter limit
                network_tight.global_constraints.at['co2_limit', 'constant'] = tighter_limit
                
                # Solve the tighter network
                network_tight.optimize(
                    solver_name="gurobi",
                    solver_options={"method": 1, "crossover": 0, "threads": 4, "OutputFlag": 0}
                )
                
                # Calculate shadow price as difference in objective divided by difference in constraint
                shadow_price = (network_tight.objective - network.objective) / (co2_limit - tighter_limit)
                print(f"Calculated shadow price from sensitivity: {shadow_price}")
            else:
                print(f"Constraint not binding. Emissions: {actual_emissions}, Limit: {co2_limit}")
                shadow_price = 0

    # Calculate generation by carrier
    generation = network.generators_t.p.sum()
    total_generation = generation.sum()
    generation_by_carrier = network.generators.carrier.map(generation)
    
    # Calculate CO2 emissions
    emissions = network.generators_t.p.sum().multiply(
        network.generators.carrier.map(network.carriers.co2_emissions)
    ).sum()

    # Print detailed results for debugging
    print(f"System cost: {network.objective}")
    print(f"CO2 emissions: {emissions}")
    if co2_limit is not None:
        print(f"CO2 limit: {co2_limit}")
        print(f"Difference: {emissions - co2_limit} ({(emissions/co2_limit - 1)*100:.4f}%)")
    print(f"Shadow price: {shadow_price}")
    print("-" * 50)

    # Return all relevant results including shadow price
    return {
        "constraint": co2_limit,
        "objective": network.objective,
        "total_load": df_elec[country].sum(),
        "total_generation": total_generation,
        "total_emissions": emissions,
        "shadow_price": shadow_price,
        "generation": {
            "wind": generation["onshorewind"],
            "solar": generation["solar"],
            "OCGT": generation["OCGT"],
            "biomass": generation["biomass"],
            "hydro": generation["hydro"]
        },
        "capacity": {
            "wind": network.generators.at["onshorewind", "p_nom_opt"] if "p_nom_opt" in network.generators.columns else network.generators.at["onshorewind", "p_nom"],
            "solar": network.generators.at["solar", "p_nom_opt"] if "p_nom_opt" in network.generators.columns else network.generators.at["solar", "p_nom"],
            "OCGT": network.generators.at["OCGT", "p_nom_opt"] if "p_nom_opt" in network.generators.columns else network.generators.at["OCGT", "p_nom"],
            "biomass": network.generators.at["biomass", "p_nom_opt"] if "p_nom_opt" in network.generators.columns else network.generators.at["biomass", "p_nom"],
            "hydro": network.generators.at["hydro", "p_nom"]
        }
    }

# Run the model for different CO2 constraints
# First run without constraint to get baseline emissions
print("Running baseline model without CO2 constraint...")
baseline_results = solve_with_co2_constraint(None)
baseline_emissions = baseline_results["total_emissions"]

# Define constraints as percentages of baseline
co2_constraints = [
    None,  # No limit (will be set to 99% of baseline to ensure binding constraint)
    baseline_emissions * 0.9,  # 90% of baseline
    baseline_emissions * 0.7,  # 70% of baseline
    baseline_emissions * 0.5,  # 50% of baseline
    baseline_emissions * 0.3,  # 30% of baseline
    baseline_emissions * 0.1,  # 10% of baseline
    baseline_emissions * 0.05  # 5% of baseline
]

# Store results for each constraint
results_list = []

for constraint in co2_constraints:
    constraint_desc = "No limit" if constraint is None else f"{constraint/1000000:.2f}M tons"
    print(f"\nSolving with CO2 constraint: {constraint_desc}")
    results = solve_with_co2_constraint(constraint)
    results_list.append(results)
    print(f"Shadow price of CO2: {results['shadow_price']} €/ton")
    print(f"System cost: {results['objective']} €")
    print(f"CO2 emissions: {results['total_emissions']} tons")
    print("-" * 50)

# Process results for visualization
results_df = pd.DataFrame([
    {
        "constraint": r["constraint"],
        "system_cost": r["objective"],
        "cost_per_mwh": r["objective"] / r["total_load"],
        "co2_emissions": r["total_emissions"],
        "shadow_price": r["shadow_price"] if r["shadow_price"] is not None and r["shadow_price"] > 0 else 0,
        "wind_gen": r["generation"]["wind"],
        "solar_gen": r["generation"]["solar"],
        "hydro_gen": r["generation"]["hydro"],
        "biomass_gen": r["generation"]["biomass"],
        "OCGT_gen": r["generation"]["OCGT"],
        "wind_cap": r["capacity"]["wind"],
        "solar_cap": r["capacity"]["solar"],
        "hydro_cap": r["capacity"]["hydro"],
        "biomass_cap": r["capacity"]["biomass"],
        "OCGT_cap": r["capacity"]["OCGT"]
    }
    for r in results_list
])

# Calculate percentage of generation for each technology
total_gen = results_df[["wind_gen", "solar_gen", "hydro_gen", "biomass_gen", "OCGT_gen"]].sum(axis=1)
results_df["wind_pct"] = results_df["wind_gen"] / total_gen * 100
results_df["solar_pct"] = results_df["solar_gen"] / total_gen * 100
results_df["hydro_pct"] = results_df["hydro_gen"] / total_gen * 100
results_df["biomass_pct"] = results_df["biomass_gen"] / total_gen * 100
results_df["OCGT_pct"] = results_df["OCGT_gen"] / total_gen * 100

# Prepare labels for plotting
x_labels = ["99% baseline"] + [f"{c/baseline_emissions*100:.0f}% of baseline" for c in co2_constraints[1:]]
x_pos = np.arange(len(co2_constraints))

# Create figure for shadow price visualization
plt.figure(figsize=(12, 8))
shadow_prices = results_df["shadow_price"].values

# Plot shadow prices
plt.subplot(2, 1, 1)
plt.bar(x_pos, shadow_prices, color='darkred', alpha=0.7)
plt.title('Shadow Price of CO2 Emissions', fontsize=16)
plt.ylabel('Shadow Price (€/ton CO2)', fontsize=14)
plt.xticks(x_pos, x_labels, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate shadow prices
for i, price in enumerate(shadow_prices):
    if price > 0:
        plt.annotate(f"{price:.2f} €/ton", 
                     xy=(x_pos[i], price),
                     xytext=(0, 5),
                     textcoords='offset points',
                     ha='center',
                     fontsize=10)

# Plot system cost vs. shadow price
plt.subplot(2, 1, 2)
valid_indices = shadow_prices > 0
if any(valid_indices):
    plt.plot(shadow_prices[valid_indices], 
             results_df['system_cost'].values[valid_indices]/1e6, 
             'bo-', linewidth=2, markersize=8)
    plt.xlabel('Shadow Price (€/ton CO2)', fontsize=14)
    plt.ylabel('System Cost (Million €)', fontsize=14)
    plt.title('System Cost vs. Shadow Price', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate points
    for i in np.where(valid_indices)[0]:
        plt.annotate(f"{x_labels[i]}", 
                     xy=(shadow_prices[i], results_df['system_cost'].values[i]/1e6),
                     xytext=(5, 0),
                     textcoords='offset points',
                     fontsize=10)
else:
    plt.text(0.5, 0.5, "No valid shadow prices to plot", 
             horizontalalignment='center', fontsize=14)

plt.tight_layout()
plt.savefig('shadow_price_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table with shadow prices
system_costs = results_df['system_cost'].values/1000000  # in million euros
cost_per_mwh = results_df['cost_per_mwh'].values  # in €/MWh
co2_emissions = results_df['co2_emissions'].values/1000000  # in million tons
shadow_prices = results_df['shadow_price'].values

# Calculate technology capacities in GW
wind_cap = results_df['wind_cap'].values / 1000
solar_cap = results_df['solar_cap'].values / 1000
hydro_cap = results_df['hydro_cap'].values / 1000
biomass_cap = results_df['biomass_cap'].values / 1000
ocgt_cap = results_df['OCGT_cap'].values / 1000
total_cap = wind_cap + solar_cap + hydro_cap + biomass_cap + ocgt_cap

# Create a comprehensive summary
summary_data = {
    'CO₂ Limit': x_labels,
    'Emissions (Mt)': co2_emissions,
    'System Cost (M€)': system_costs,
    'Cost/MWh (€)': cost_per_mwh,
    'Shadow Price (€/t)': shadow_prices,
    'Wind (%)': results_df['wind_pct'].values,
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

# Create figure with three subplots for comprehensive view
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 1, 1]})

# Plot 1: Shadow Price vs CO2 Constraint
ax1.bar(x_pos, shadow_prices, color='darkred', alpha=0.7, width=0.6)
ax1.set_ylabel('Shadow Price (€/ton CO₂)', fontsize=14)
ax1.set_title('Shadow Price of CO₂ Constraint', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)

# Annotate shadow prices
for i, price in enumerate(shadow_prices):
    if price > 0:
        ax1.annotate(f"{price:.2f} €/ton", 
                    xy=(x_pos[i], price),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10)

# Plot 2: System Cost vs CO2 Constraint
ax2.plot(x_pos, system_costs, 'bo-', linewidth=3, markersize=10)
ax2.set_ylabel('System Cost (Million €)', fontsize=14)
ax2.set_title('Impact of CO₂ Constraints on System Cost', fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)

# Annotate actual costs
for i, cost in enumerate(system_costs):
    ax2.annotate(f"{cost:.1f}M€", 
                xy=(x_pos[i], cost),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=11)

# Plot 3: Cost per MWh vs CO2 Constraint
ax3.plot(x_pos, cost_per_mwh, 'ro-', linewidth=3, markersize=10)
ax3.set_ylabel('Cost per MWh (€/MWh)', fontsize=14)
ax3.set_xlabel('CO₂ Constraint', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7)

# Annotate cost per MWh
for i, cost in enumerate(cost_per_mwh):
    ax3.annotate(f"{cost:.1f} €/MWh", 
                xy=(x_pos[i], cost),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=11)

# Set x-ticks and labels
ax3.set_xticks(x_pos)
ax3.set_xticklabels(x_labels, rotation=45)

# Add a secondary x-axis showing the actual CO2 emissions
ax4 = ax3.twiny()
ax4.set_xlim(ax3.get_xlim())
# Set positions for the top ticks
ax4.set_xticks(x_pos)
# Set the actual emission values as labels
ax4.set_xticklabels([f"{emission:.1f}" for emission in co2_emissions], rotation=45)
ax4.set_xlabel('Actual CO₂ Emissions (Million tons)', fontsize=14)

# Calculate percentage increase from baseline
baseline_cost = system_costs[0]
percentage_increase = [(cost/baseline_cost - 1) * 100 for cost in system_costs]

# Add text annotations for percentage increases
for i, pct in enumerate(percentage_increase):
    if i > 0:  # Skip the baseline
        ax2.annotate(f"+{pct:.1f}%", 
                    xy=(x_pos[i], system_costs[i]),
                    xytext=(0, -25),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    color='darkblue')

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)  # Reduce space between subplots
plt.savefig('system_cost_vs_co2_constraint_with_shadow_prices.png', dpi=300, bbox_inches='tight')
plt.show()
