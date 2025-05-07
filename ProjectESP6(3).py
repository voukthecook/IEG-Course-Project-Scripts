import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import warnings
from shapely.errors import ShapelyDeprecationWarning
import os
from datetime import datetime

# Suppress ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# Create output directory for figures
output_dir = 'pypsa_output'
os.makedirs(output_dir, exist_ok=True)

# Create the network with three countries
network = pypsa.Network()

# Set snapshots for the year 2015
hours_in_2015 = pd.date_range('2015-01-01 00:00Z', '2015-12-31 23:00Z', freq='h')
network.set_snapshots(hours_in_2015.values)

# Define the countries to model
countries = ['ESP', 'FRA', 'PRT']  # Spain, France, Portugal

# 1. Add buses for each country
for country in countries:
    network.add("Bus", f"{country} electricity bus")

# 2. Load electricity demand data for all countries
print("Loading electricity demand data for all countries")
# Adjust path based on your environment or use a relative path
data_path = 'C:/Users/nasos/DTU/IEG(Env and source data)/'
df_elec = pd.read_csv(f'{data_path}electricity_demand.csv', sep=';', index_col=0)  # in MWh
df_elec.index = pd.to_datetime(df_elec.index)

# Add load for each country
for country in countries:
    print(f"Adding load data for {country}")
    network.add("Load", 
                f"{country} load",
                bus=f"{country} electricity bus",
                p_set=df_elec[country].values)

# 3. Calculate annuity factor for cost calculations
def annuity(n, r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate r."""
    if r > 0:
        return r/(1 - 1/(1 + r)**n)
    else:
        return 1/n

# 4. Load renewable capacity factors for all countries
print("Loading solar capacity factors for all countries")
solar_cf = pd.read_csv(f'{data_path}pv_optimal.csv', sep=';', index_col=0)
solar_cf.index = pd.to_datetime(solar_cf.index)

# Format snapshots for indexing
formatted_snapshots = [hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]

# Extract capacity factors for each country
CF_solar_ESP = solar_cf['ESP'][formatted_snapshots]
CF_solar_FRA = solar_cf['FRA'][formatted_snapshots]
CF_solar_PRT = solar_cf['PRT'][formatted_snapshots]

print("Loading wind capacity factors for all countries")
wind_cf = pd.read_csv(f'{data_path}onshore_wind_1979-2017.csv', sep=';', index_col=0)
wind_cf.index = pd.to_datetime(wind_cf.index)

CF_wind_ESP = wind_cf['ESP'][formatted_snapshots]
CF_wind_FRA = wind_cf['FRA'][formatted_snapshots]
CF_wind_PRT = wind_cf['PRT'][formatted_snapshots]

# 5. Add generators for each country with technology-specific parameters
# Technology cost and lifetime assumptions
tech_params = {
    "onshorewind": {
        "lifetime": 30,
        "capital_costs": {
            "ESP": 1000000,  # €/MW
            "FRA": 1100000,  # €/MW
            "PRT": 950000    # €/MW
        },
        "fixed_om": 25000    # €/MW/year
    },
    "solar": {
        "lifetime": 25,
        "capital_costs": {
            "ESP": 600000,   # €/MW
            "FRA": 700000,   # €/MW
            "PRT": 650000    # €/MW
        },
        "fixed_om": 20000    # €/MW/year
    },
    "gas": {
        "lifetime": 25,
        "capital_costs": {
            "ESP": 800000,   # €/MW
            "FRA": 850000,   # €/MW
            "PRT": 820000    # €/MW
        },
        "marginal_costs": {
            "ESP": 50,       # €/MWh
            "FRA": 55,       # €/MWh
            "PRT": 53        # €/MWh
        },
        "fixed_om": 20000    # €/MW/year
    },
    "nuclear": {
        "lifetime": 60,
        "capital_cost": 5000000,  # €/MW
        "marginal_cost": 20,       # €/MWh
        "fixed_om": 90000    # €/MW/year
    },
    "hydro": {
        "lifetime": 80,
        "capital_cost": 2000000,  # €/MW
        "marginal_cost": 5,        # €/MWh
        "fixed_om": 45000    # €/MW/year
    }
}

# Discount rate for all calculations
discount_rate = 0.07

# For Spain (ESP)
print("Adding generators for Spain (ESP)")
network.add("Generator",
            "ESP onshorewind",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=100000,
            marginal_cost=0,
            capital_cost=annuity(tech_params["onshorewind"]["lifetime"], discount_rate) * 
                       (tech_params["onshorewind"]["capital_costs"]["ESP"] + 
                        tech_params["onshorewind"]["fixed_om"] / discount_rate),
            p_max_pu=CF_wind_ESP.values)

network.add("Generator",
            "ESP solar",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=100000,
            marginal_cost=0,
            capital_cost=annuity(tech_params["solar"]["lifetime"], discount_rate) * 
                       (tech_params["solar"]["capital_costs"]["ESP"] + 
                        tech_params["solar"]["fixed_om"] / discount_rate),
            p_max_pu=CF_solar_ESP.values)

# Add gas as backup for Spain
network.add("Generator",
            "ESP gas",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            marginal_cost=tech_params["gas"]["marginal_costs"]["ESP"],
            capital_cost=annuity(tech_params["gas"]["lifetime"], discount_rate) * 
                       (tech_params["gas"]["capital_costs"]["ESP"] + 
                        tech_params["gas"]["fixed_om"] / discount_rate))

# For France (FRA)
print("Adding generators for France (FRA)")
# Add nuclear with fixed capacity and constant availability
network.add("Generator",
            "FRA nuclear",
            bus="FRA electricity bus",
            p_nom=63000,  # MW of nuclear capacity
            p_nom_extendable=False,
            marginal_cost=tech_params["nuclear"]["marginal_cost"],
            p_max_pu=0.85)  # Constant availability factor of 85%

network.add("Generator",
            "FRA onshorewind",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(tech_params["onshorewind"]["lifetime"], discount_rate) * 
                       (tech_params["onshorewind"]["capital_costs"]["FRA"] + 
                        tech_params["onshorewind"]["fixed_om"] / discount_rate),
            p_max_pu=CF_wind_FRA.values)

network.add("Generator",
            "FRA solar",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(tech_params["solar"]["lifetime"], discount_rate) * 
                       (tech_params["solar"]["capital_costs"]["FRA"] + 
                        tech_params["solar"]["fixed_om"] / discount_rate),
            p_max_pu=CF_solar_FRA.values)

# Add gas as backup for France
network.add("Generator",
            "FRA gas",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            marginal_cost=tech_params["gas"]["marginal_costs"]["FRA"],
            capital_cost=annuity(tech_params["gas"]["lifetime"], discount_rate) * 
                       (tech_params["gas"]["capital_costs"]["FRA"] + 
                        tech_params["gas"]["fixed_om"] / discount_rate))

# For Portugal (PRT)
print("Adding generators for Portugal (PRT)")
# Add hydro with fixed capacity and constant availability
network.add("Generator",
            "PRT hydro",
            bus="PRT electricity bus",
            p_nom=7000,  # MW of hydro capacity
            p_nom_extendable=False,
            marginal_cost=tech_params["hydro"]["marginal_cost"],
            p_max_pu=0.5)  # Simplified constant availability of 50%

network.add("Generator",
            "PRT onshorewind",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(tech_params["onshorewind"]["lifetime"], discount_rate) * 
                       (tech_params["onshorewind"]["capital_costs"]["PRT"] + 
                        tech_params["onshorewind"]["fixed_om"] / discount_rate),
            p_max_pu=CF_wind_PRT.values)

network.add("Generator",
            "PRT solar",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(tech_params["solar"]["lifetime"], discount_rate) * 
                       (tech_params["solar"]["capital_costs"]["PRT"] + 
                        tech_params["solar"]["fixed_om"] / discount_rate),
            p_max_pu=CF_solar_PRT.values)

# Add gas as backup for Portugal
network.add("Generator",
            "PRT gas",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            marginal_cost=tech_params["gas"]["marginal_costs"]["PRT"],
            capital_cost=annuity(tech_params["gas"]["lifetime"], discount_rate) * 
                       (tech_params["gas"]["capital_costs"]["PRT"] + 
                        tech_params["gas"]["fixed_om"] / discount_rate))

# 6. Add transmission links between countries
print("Adding transmission links between countries")
# Define transmission parameters
transmission_params = {
    "ESP-FRA": {
        "p_nom": 2800,      # MW - current capacity
        "capital_cost": 900000,  # €/MW
        "lifetime": 40
    },
    "ESP-PRT": {
        "p_nom": 4200,      # MW - current capacity
        "capital_cost": 750000,  # €/MW
        "lifetime": 40
    }
}

network.add("Link",
            "ESP-FRA",
            bus0="ESP electricity bus",
            bus1="FRA electricity bus",
            p_nom=transmission_params["ESP-FRA"]["p_nom"],
            p_nom_extendable=True,
            p_min_pu=-1,  # Allow bidirectional flow
            capital_cost=annuity(transmission_params["ESP-FRA"]["lifetime"], discount_rate) * 
                       transmission_params["ESP-FRA"]["capital_cost"])

network.add("Link",
            "ESP-PRT",
            bus0="ESP electricity bus",
            bus1="PRT electricity bus",
            p_nom=transmission_params["ESP-PRT"]["p_nom"],
            p_nom_extendable=True,
            p_min_pu=-1,  # Allow bidirectional flow
            capital_cost=annuity(transmission_params["ESP-PRT"]["lifetime"], discount_rate) * 
                       transmission_params["ESP-PRT"]["capital_cost"])

def run_optimization_and_analyze(network):
    """Run optimization and perform comprehensive analysis of results"""
    # 7. Run the optimization
    print("\nRunning optimization for interconnected ESP, FRA, and PRT system")
    try:
        # Try using Gurobi first
        start_time = datetime.now()
        network.optimize(solver_name='gurobi')
        end_time = datetime.now()
    except:
        # Fall back to GLPK if Gurobi is not available
        print("Gurobi not available, falling back to GLPK")
        try:
            start_time = datetime.now()
            network.optimize(solver_name='glpk')
            end_time = datetime.now()
        except:
            # Fall back to other available solvers
            print("GLPK not available, trying other available solvers")
            start_time = datetime.now()
            network.optimize()
            end_time = datetime.now()
    
    print(f"Optimization completed in {(end_time - start_time).total_seconds()} seconds")

    # 8. Analyze results - printing optimal capacities
    print("\nOptimal generation capacities (MW):")
    
    capacity_results = {}
    for country in countries:
        capacity_results[country] = {}
        
        print(f"\n{country}:")
        
        # Get all generators for this country
        country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        
        for gen in country_gens:
            gen_type = gen.replace(f"{country} ", "")
            
            if network.generators.loc[gen, "p_nom_extendable"]:
                capacity = network.generators.loc[gen, "p_nom_opt"]
                print(f"  {gen_type}: {capacity:.1f}")
                capacity_results[country][gen_type] = capacity
            else:
                capacity = network.generators.loc[gen, "p_nom"]
                print(f"  {gen_type}: {capacity:.1f} (fixed)")
                capacity_results[country][gen_type] = capacity

    print("\nOptimal transmission capacities (MW):")
    for link in network.links.index:
        print(f"{link}: {network.links.loc[link, 'p_nom_opt']:.1f}")

    # 9. Calculate energy mix and system costs
    total_demand = {country: df_elec[country].sum() for country in countries}
    total_generation = {}
    generation_by_carrier = {country: {} for country in countries}

    for country in countries:
        country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        gen_by_type = {}
        
        for gen in country_gens:
            gen_type = gen.replace(f"{country} ", "")
            gen_by_type[gen_type] = network.generators_t.p[gen].sum()
            
            # Add to the carrier-specific tracking
            if gen_type not in generation_by_carrier[country]:
                generation_by_carrier[country][gen_type] = 0
            generation_by_carrier[country][gen_type] += network.generators_t.p[gen].sum()
            
        total_generation[country] = gen_by_type

    print("\nEnergy Generation Mix (GWh):")
    for country in countries:
        print(f"\n{country} (total demand: {total_demand[country]/1000:.1f} GWh):")
        total_gen = sum(total_generation[country].values())
        for gen_type, energy in total_generation[country].items():
            percentage = (energy / total_demand[country]) * 100
            print(f"  {gen_type}: {energy/1000:.1f} GWh ({percentage:.1f}%)")

    # Calculate total system cost
    total_cost = network.objective / 8760  # Average hourly cost
    print(f"\nTotal system cost: {total_cost:.2f} €/h or {total_cost*8760/1e6:.2f} M€/year")

    # Calculate country-specific costs
    country_costs = {}
    for country in countries:
        country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        capital_costs = 0
        operating_costs = 0
        
        for gen in country_gens:
            gen_type = gen.replace(f"{country} ", "")
            
            # Capital costs
            if network.generators.loc[gen, "p_nom_extendable"]:
                p_nom = network.generators.loc[gen, "p_nom_opt"]
            else:
                p_nom = network.generators.loc[gen, "p_nom"]
                
            capital_cost_hourly = p_nom * network.generators.loc[gen, "capital_cost"] / 8760
            capital_costs += capital_cost_hourly
            
            # Operating costs
            marginal_cost = network.generators.loc[gen, "marginal_cost"]
            energy_generated = network.generators_t.p[gen].sum()
            operating_costs += marginal_cost * energy_generated / 8760
            
        country_costs[country] = {
            "capital_costs": capital_costs,
            "operating_costs": operating_costs,
            "total_costs": capital_costs + operating_costs
        }
    
    print("\nCountry-specific costs (€/h):")
    for country in countries:
        print(f"\n{country}:")
        print(f"  Capital costs: {country_costs[country]['capital_costs']:.2f} €/h")
        print(f"  Operating costs: {country_costs[country]['operating_costs']:.2f} €/h")
        print(f"  Total costs: {country_costs[country]['total_costs']:.2f} €/h")

    # 10. Create comprehensive visualizations
    create_visualizations(network, total_demand, generation_by_carrier, capacity_results)
    
    return network, capacity_results, total_generation, country_costs


def create_visualizations(network, total_demand, generation_by_carrier, capacity_results):
    """Create comprehensive visualizations for the model results"""
    
    # 10.1 Plot power generation for each country (first week)
    for country in countries:
        fig, ax = plt.subplots(figsize=(12, 6))
        generators_country = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        generation_country = network.generators_t.p[generators_country].loc[network.snapshots[:168]]  # First week
        generation_country.columns = [col.replace(f"{country} ", "") for col in generation_country.columns]
        
        # Add load for comparison
        load = network.loads_t.p_set[f"{country} load"].loc[network.snapshots[:168]]
        
        # Plot generation stacked
        generation_country.plot.area(ax=ax, stacked=True, alpha=0.7, linewidth=0)
        
        # Plot load as a line
        ax.plot(generation_country.index, load, 'k-', linewidth=2, label='Load')
        
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Power [MW]", fontsize=12)
        ax.set_title(f"Electricity Generation and Load in {country} - First Week", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{country.lower()}_generation_week1.png", dpi=300)
        plt.close()

    # 10.2 Plot transmission flows (first week)
    fig, ax = plt.subplots(figsize=(12, 6))
    links = network.links_t.p0[["ESP-FRA", "ESP-PRT"]].loc[network.snapshots[:168]]  # First week
    links.plot(ax=ax, linewidth=2)
    
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Power [MW]", fontsize=12)
    ax.set_title("Transmission Flows Between Countries (First Week)\nPositive: Export from Spain", fontsize=14)
    ax.legend(["ESP → FRA", "ESP → PRT"])
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/transmission_flows_week1.png", dpi=300)
    plt.close()

    # 10.3 Create annual energy mix bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for consistency
    carrier_colors = {
        "onshorewind": "#3498db",  # Blue
        "solar": "#f1c40f",       # Yellow
        "gas": "#e74c3c",         # Red
        "nuclear": "#9b59b6",     # Purple
        "hydro": "#2ecc71"        # Green
    }
    
    # Create data for stacked bar chart
    carriers = ["onshorewind", "solar", "gas", "nuclear", "hydro"]
    data = []
    
    for country in countries:
        country_data = []
        for carrier in carriers:
            if carrier in generation_by_carrier[country]:
                country_data.append(generation_by_carrier[country][carrier] / 1e6)  # Convert to TWh
            else:
                country_data.append(0)
        data.append(country_data)
    
    # Create stacked bar chart
    bar_width = 0.6
    r = range(len(countries))
    
    bottom = np.zeros(len(countries))
    for i, carrier in enumerate(carriers):
        carrier_data = [data[j][i] for j in range(len(countries))]
        ax.bar(r, carrier_data, bottom=bottom, color=carrier_colors.get(carrier, "gray"), 
               width=bar_width, label=carrier)
        bottom += carrier_data
    
    # Add demand line
    demand_values = [total_demand[country] / 1e6 for country in countries]  # Convert to TWh
    ax.plot(r, demand_values, 'ko-', linewidth=2, markersize=8, label='Annual Demand')
    
    # Add labels and formatting
    ax.set_xticks(r)
    ax.set_xticklabels(countries)
    ax.set_ylabel('Annual Energy [TWh]', fontsize=12)
    ax.set_title('Annual Electricity Generation Mix by Country', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add data labels
    for i, country in enumerate(countries):
        total_gen = sum(data[i])
        ax.text(i, total_gen + 0.5, f"{total_gen:.1f} TWh", 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_mix_bar_chart.png", dpi=300)
    plt.close()

    # 10.4 Create capacity comparison chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for capacity comparison
    capacity_data = {}
    for carrier in carriers:
        capacity_data[carrier] = []
        for country in countries:
            if carrier in capacity_results[country]:
                capacity_data[carrier].append(capacity_results[country][carrier])
            else:
                capacity_data[carrier].append(0)
    
    # Plot capacities as grouped bar chart
    x = np.arange(len(countries))
    width = 0.15
    multiplier = 0
    
    for carrier, capacity in capacity_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, capacity, width, label=carrier, 
                      color=carrier_colors.get(carrier, "gray"))
        
        # Add labels on top of bars
        for rect in rects:
            height = rect.get_height()
            if height > 0:  # Only add label if there's actual capacity
                ax.annotate(f'{height:.0f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        multiplier += 1
    
    # Add labels and formatting
    ax.set_ylabel('Capacity [MW]', fontsize=12)
    ax.set_title('Optimal Generation Capacity by Country and Technology', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(countries)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/capacity_comparison.png", dpi=300)
    plt.close()

    # 10.5 Create geographic flow visualization
    create_energy_flow_visualizations(network)


def create_energy_flow_visualizations(network):
    """Create comprehensive energy flow visualizations for the 3-country model."""
    plt.rc("figure", figsize=(16, 12))
    
    # Set geographical positions for visualization
    bus_positions = {
        "ESP electricity bus": (-3.7, 40.0),  # Spain (longitude, latitude)
        "FRA electricity bus": (2.3, 46.6),   # France
        "PRT electricity bus": (-8.0, 39.5)   # Portugal
    }
    
    # Set coordinates for all buses
    for bus in network.buses.index:
        network.buses.loc[bus, "x"] = bus_positions[bus][0]
        network.buses.loc[bus, "y"] = bus_positions[bus][1]
    
    # Color mapping for different energy carriers
    carrier_colors = {
        "onshorewind": "#3498db",  # Blue
        "solar": "#f1c40f",       # Yellow
        "gas": "#e74c3c",         # Red
        "nuclear": "#9b59b6",     # Purple
        "hydro": "#2ecc71"        # Green
    }
    
    # Extract average flow values for links
    link_flow = network.links_t.p0[["ESP-FRA", "ESP-PRT"]].mean()
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])
    
    # Define axes for each plot
    ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax3 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax4 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax5 = plt.subplot(gs[2, :])
    
    axes = [ax1, ax2, ax3, ax4]
    
    # Get generator information by country and carrier
    gen_by_country = {}
    for country in ["ESP", "FRA", "PRT"]:
        # Filter generators by country
        country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        
        # Get optimal capacities (for extendable) or fixed capacities
        capacities = []
        for gen in country_gens:
            if network.generators.loc[gen, "p_nom_extendable"]:
                capacities.append(network.generators.loc[gen, "p_nom_opt"])
            else:
                capacities.append(network.generators.loc[gen, "p_nom"])
                
        gen_by_country[f"{country} electricity bus"] = sum(capacities)
    
    # Get average generation by carrier for pie charts
    gen_by_country_carrier = {}
    for country in ["ESP", "FRA", "PRT"]:
        country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        gen_by_carrier = {}
        
        for gen in country_gens:
            carrier = gen.replace(f"{country} ", "")
            gen_by_carrier[carrier] = network.generators_t.p[gen].mean()
            
        gen_by_country_carrier[country] = gen_by_carrier
    
    # PLOT 1: Basic capacity visualization
    bus_sizes = pd.Series(gen_by_country)
    
    # Create a mapping of bus colors based on predominant generation
    bus_colors = {}
    for country in ["ESP", "FRA", "PRT"]:
        bus = f"{country} electricity bus"
        country_gens = gen_by_country_carrier[country]
        if country_gens:
            # Get carrier with highest generation
            predominant_carrier = max(country_gens.items(), key=lambda x: x[1])[0]
            bus_colors[bus] = carrier_colors.get(predominant_carrier, "gray")
        else:
            bus_colors[bus] = "gray"
    
    # Plot the network with capacities
    collections = network.plot(
        ax=ax1,
        bus_sizes=bus_sizes/100000,  # Scale down for visibility
        bus_colors=bus_colors,
        line_widths=0,  # No lines since we're using links
        margin=0.2,
        color_geomap=True
    )
    
    # Add links manually as arrows
    for link_name in network.links.index:
        bus0 = network.links.loc[link_name, "bus0"]
        bus1 = network.links.loc[link_name, "bus1"]
        
        # Get coordinates
        x0, y0 = network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"]
        x1, y1 = network.buses.loc[bus1, "x"], network.buses.loc[bus1, "y"]
        
        # Get optimal capacity
        width = network.links.loc[link_name, "p_nom_opt"] / 200000
        
        # Draw arrow
        ax1.annotate(
            "",
            xy=(x1, y1), xycoords='data',
            xytext=(x0, y0), textcoords='data',
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3",
                lw=width*5,  # Scale width for visibility
                color="black",
                alpha=0.7
            )
        )
        
        # Add capacity label
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        capacity = network.links.loc[link_name, "p_nom_opt"]
        ax1.text(
            mid_x, mid_y, 
            f"{capacity:.0f} MW",
            fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7),
            ha='center', va='center'
        )
    
    ax1.set_title("Optimal Capacity Visualization", fontsize=14)
    
    # Add legend for country generation mix
    legend_elements = []
    for carrier, color in carrier_colors.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                   markersize=10, label=carrier)
        )
    
    ax1.legend(handles=legend_elements, loc='lower left', title="Generation Types")
    
    # PLOT 2: Flow visualization
    # Sample a week of flows for visualization
    sample_week = 24 * 7  # First week
    link_flows_week = network.links_t.p0[["ESP-FRA", "ESP-PRT"]].iloc[:sample_week]
    
    # Plot network with average flows
    collections = network.plot(
        ax=ax2,
        bus_sizes=bus_sizes/100000,
        bus_colors=bus_colors,
        line_widths=0,
        margin=0.2,
        color_geomap=True
    )
    
    # Add links manually with flow direction and magnitude
    for link_name in network.links.index:
        bus0 = network.links.loc[link_name, "bus0"]
        bus1 = network.links.loc[link_name, "bus1"]
        
        # Get coordinates
        x0, y0 = network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"]
        x1, y1 = network.buses.loc[bus1, "x"], network.buses.loc[bus1, "y"]
        
        # Get average flow (positive: bus0→bus1, negative: bus1→bus0)
        avg_flow = link_flow[link_name]
        
        # Width based on absolute flow
        width = abs(avg_flow) / 20000
        
        # Color based on flow direction
        color = "green" if avg_flow > 0 else "red"
        
        # Direction based on flow
        if avg_flow < 0:
            # Swap coordinates if flow is negative (from bus1 to bus0)
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        # Draw arrow
        ax2.annotate(
            "",
            xy=(x1, y1), xycoords='data',
            xytext=(x0, y0), textcoords='data',
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3",
                lw=width*5,
                color=color,
                alpha=0.7
            )
        )
        
        # Add flow label
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        ax2.text(
            mid_x, mid_y,
            f"{abs(avg_flow):.0f} MW",
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7),
            ha='center', va='center'
        )
    
    ax2.set_title("Average Energy Flow Visualization", fontsize=14)
    
    # PLOT 3: Peak flow visualization (time with maximum total flow)
    # Find time with maximum total flow
    flow_sums = network.links_t.p0.abs().sum(axis=1)
    max_flow_time = flow_sums.idxmax()
    max_flow = network.links_t.p0.loc[max_flow_time]
    
    # Calculate generation at peak flow time
    gen_at_peak = {}
    for country in ["ESP", "FRA", "PRT"]:
        # Filter generators by country
        country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        gen_at_peak[f"{country} electricity bus"] = network.generators_t.p[country_gens].loc[max_flow_time].sum()
    
    # Plot network with peak flows
    collections = network.plot(
        ax=ax3,
        bus_sizes=pd.Series(gen_at_peak)/50000,
        bus_colors=bus_colors,
        line_widths=0,
        margin=0.2,
        color_geomap=True
    )
    
    # Add links manually with peak flow direction and magnitude
    for link_name in network.links.index:
        bus0 = network.links.loc[link_name, "bus0"]
        bus1 = network.links.loc[link_name, "bus1"]
        
        # Get coordinates
        x0, y0 = network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"]
        x1, y1 = network.buses.loc[bus1, "x"], network.buses.loc[bus1, "y"]
        
        # Get flow at peak time
        peak_flow = max_flow[link_name]
        
        # Width based on absolute flow
        width = abs(peak_flow) / 20000
        
        # Color based on flow direction
        color = "green" if peak_flow > 0 else "red"
        
# Direction based on flow
        if peak_flow < 0:
            # Swap coordinates if flow is negative (from bus1 to bus0)
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        # Draw arrow
        ax3.annotate(
            "",
            xy=(x1, y1), xycoords='data',
            xytext=(x0, y0), textcoords='data',
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3",
                lw=width*5,
                color=color,
                alpha=0.7
            )
        )
        
        # Add flow label
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        ax3.text(
            mid_x, mid_y,
            f"{abs(peak_flow):.0f} MW",
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7),
            ha='center', va='center'
        )
    
    peak_time_str = pd.to_datetime(max_flow_time).strftime('%Y-%m-%d %H:%M')
    ax3.set_title(f"Peak Flow Visualization ({peak_time_str})", fontsize=14)
    
    # PLOT 4: Pie charts for energy mix by country
    collections = network.plot(
        ax=ax4,
        bus_sizes=0,  # No bus symbols
        line_widths=0,
        margin=0.2,
        color_geomap=True
    )
    
    # Calculate the average generation by technology for each country
    for country in ["ESP", "FRA", "PRT"]:
        bus = f"{country} electricity bus"
        x, y = network.buses.loc[bus, "x"], network.buses.loc[bus, "y"]
        
        # Get generators for this country
        gen_data = gen_by_country_carrier[country]
        
        # Calculate total generation to filter out small contributions
        total_gen = sum(gen_data.values())
        
        # Define minimum percentage to show (for readability)
        min_pct = 0.05
        
        # Prepare data for pie chart
        labels = []
        values = []
        colors_pie = []
        
        # Add data points above minimum percentage
        for carrier, value in gen_data.items():
            if value / total_gen > min_pct:
                labels.append(carrier)
                values.append(value)
                colors_pie.append(carrier_colors.get(carrier, "gray"))
        
        # Create a small pie chart
        pie_size = 0.15  # Size of pie chart (adjust based on map size)
        
        if values:  # Only create pie if there are values
            wedges, texts, autotexts = ax4.pie(
                values,
                colors=colors_pie,
                autopct='%1.1f%%',
                pctdistance=0.8,
                radius=pie_size,
                center=(x, y),
                wedgeprops=dict(width=pie_size*0.5, edgecolor='w'),
                textprops=dict(size=8)
            )
            
            # Make texts smaller and move them outward
            for autotext in autotexts:
                autotext.set_size(7)
                
            # Add country label
            ax4.text(x, y-pie_size-0.01, country, ha='center', va='top', fontsize=12, 
                   weight='bold', bbox=dict(facecolor='white', alpha=0.7))
    
    ax4.set_title("Energy Generation Mix by Country", fontsize=14)
    
    # PLOT 5: Time series showing the weekly generation mix by technology
    # Sample 2 weeks of data
    start_idx = 24 * 7 * 4  # Start at 4th week for diversity
    end_idx = start_idx + 24 * 14  # Show 2 weeks
    
    # Extract generation by technology for the sample period
    tech_names = ["onshorewind", "solar", "gas", "nuclear", "hydro"]
    generation_by_tech = {tech: pd.Series(0, index=network.snapshots[start_idx:end_idx]) for tech in tech_names}
    
    # Sum generation by technology
    for gen in network.generators.index:
        for tech in tech_names:
            if tech in gen:
                generation_by_tech[tech] += network.generators_t.p[gen].iloc[start_idx:end_idx]
    
    # Create a stacked area chart
    ax5.stackplot(
        network.snapshots[start_idx:end_idx],
        [generation_by_tech[tech] for tech in tech_names],
        labels=tech_names,
        colors=[carrier_colors.get(tech, "gray") for tech in tech_names],
        alpha=0.8
    )
    
    # Format x-axis with date ticks
    hours = pd.to_datetime(network.snapshots[start_idx:end_idx])
    ax5.set_xlim(hours[0], hours[-1])
    
    # Set date formatter for x-ticks
    import matplotlib.dates as mdates
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax5.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    ax5.set_ylabel('Power [MW]')
    ax5.set_title('Generation Mix Over Time (2-Week Sample)', fontsize=14)
    ax5.grid(alpha=0.3)
    ax5.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_flow_visualizations.png", dpi=300)
    plt.close()


def analyze_renewable_statistics(network):
    """Create statistics and visualizations for renewable generation."""
    # Calculate renewable penetration
    renewable_gens = [gen for gen in network.generators.index 
                     if ("onshorewind" in gen or "solar" in gen or "hydro" in gen)]
    
    conventional_gens = [gen for gen in network.generators.index 
                        if ("gas" in gen or "nuclear" in gen)]
    
    # Get total generation over time
    renewable_gen = network.generators_t.p[renewable_gens].sum(axis=1)
    conventional_gen = network.generators_t.p[conventional_gens].sum(axis=1)
    total_gen = renewable_gen + conventional_gen
    
    # Calculate percentage of renewable generation
    renewable_percentage = renewable_gen / total_gen * 100
    
    # Calculate statistics
    avg_renewable_percentage = renewable_percentage.mean()
    min_renewable_percentage = renewable_percentage.min()
    max_renewable_percentage = renewable_percentage.max()
    
    # Calculate hours with high renewable generation (>80% and >90%)
    hours_above_80 = (renewable_percentage > 80).sum()
    hours_above_90 = (renewable_percentage > 90).sum()
    
    # Create plots
    plt.figure(figsize=(12, 7))
    
    # Monthly average renewable percentage
    renewable_percentage.resample('M').mean().plot(
        kind='bar', 
        color='green',
        alpha=0.7,
        width=0.8
    )
    
    plt.axhline(y=avg_renewable_percentage, color='r', linestyle='-', label=f'Annual Avg: {avg_renewable_percentage:.1f}%')
    plt.title('Monthly Renewable Energy Penetration')
    plt.xlabel('Month')
    plt.ylabel('Renewable Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/monthly_renewable_penetration.png", dpi=300)
    plt.close()
    
    # Create a histogram of hourly renewable penetration
    plt.figure(figsize=(12, 7))
    renewable_percentage.plot(
        kind='hist',
        bins=50,
        color='green',
        alpha=0.7,
        edgecolor='black'
    )
    
    plt.axvline(x=avg_renewable_percentage, color='r', linestyle='-', label=f'Avg: {avg_renewable_percentage:.1f}%')
    plt.title('Distribution of Hourly Renewable Energy Penetration')
    plt.xlabel('Renewable Percentage (%)')
    plt.ylabel('Hours')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/renewable_penetration_histogram.png", dpi=300)
    plt.close()
    
    # Create a summary dataframe
    renewable_stats = pd.DataFrame({
        'Metric': [
            'Average Renewable Penetration',
            'Minimum Renewable Penetration',
            'Maximum Renewable Penetration',
            'Hours with >80% Renewable Generation',
            'Percentage of Year with >80% Renewable',
            'Hours with >90% Renewable Generation',
            'Percentage of Year with >90% Renewable'
        ],
        'Value': [
            f"{avg_renewable_percentage:.2f}%",
            f"{min_renewable_percentage:.2f}%",
            f"{max_renewable_percentage:.2f}%",
            hours_above_80,
            f"{hours_above_80/len(renewable_percentage)*100:.2f}%",
            hours_above_90,
            f"{hours_above_90/len(renewable_percentage)*100:.2f}%"
        ]
    })
    
    # Save the summary to CSV
    renewable_stats.to_csv(f"{output_dir}/renewable_statistics.csv", index=False)
    
    # Print the summary
    print("\nRenewable Energy Statistics:")
    print(renewable_stats.to_string(index=False))
    
    return renewable_stats


def analyze_transmission_impacts(network):
    """Analyze the impact of transmission on system costs and operations."""
    # Calculate the transmission utilization
    link_utilization = {}
    
    for link in network.links.index:
        # Get flow data
        flow = network.links_t.p0[link]
        capacity = network.links.loc[link, "p_nom_opt"]
        
        # Calculate utilization metrics
        avg_utilization = flow.abs().mean() / capacity * 100
        max_utilization = flow.abs().max() / capacity * 100
        hours_above_90pct = (flow.abs() > 0.9 * capacity).sum()
        
        # Calculate congestion hours (flow at capacity)
        congestion_hours = (flow.abs() > 0.95 * capacity).sum()
        
        link_utilization[link] = {
            'Average Utilization (%)': avg_utilization,
            'Maximum Utilization (%)': max_utilization,
            'Hours Above 90% Capacity': hours_above_90pct,
            'Congestion Hours': congestion_hours,
            '% Time Congested': congestion_hours / len(flow) * 100
        }
    
    # Create a dataframe from the results
    transmission_df = pd.DataFrame(link_utilization).T
    
    # Save to CSV
    transmission_df.to_csv(f"{output_dir}/transmission_utilization.csv")
    
    # Create visualizations of transmission utilization
    plt.figure(figsize=(12, 6))
    
    ax = transmission_df['Average Utilization (%)'].plot(
        kind='bar',
        color='blue',
        alpha=0.7,
        width=0.4,
        position=0,
        label='Avg Utilization'
    )
    
    transmission_df['% Time Congested'].plot(
        kind='bar',
        color='red',
        alpha=0.7,
        width=0.4,
        position=1,
        label='% Time Congested',
        ax=ax
    )
    
    plt.title('Transmission Line Utilization')
    plt.xlabel('Transmission Line')
    plt.ylabel('Percentage')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/transmission_utilization.png", dpi=300)
    plt.close()
    
    # Print the results
    print("\nTransmission Line Utilization:")
    print(transmission_df.to_string())
    
    return transmission_df


def create_summary_report(network, renewable_stats, transmission_df):
    """Create a comprehensive text summary report of the model results."""
    # Calculate total generation capacity by technology
    generation_by_tech = {}
    for tech in ["onshorewind", "solar", "gas", "nuclear", "hydro"]:
        gen_list = [gen for gen in network.generators.index if tech in gen]
        if gen_list:
            capacity = 0
            for gen in gen_list:
                if network.generators.loc[gen, "p_nom_extendable"]:
                    capacity += network.generators.loc[gen, "p_nom_opt"]
                else:
                    capacity += network.generators.loc[gen, "p_nom"]
            generation_by_tech[tech] = capacity
    
    # Calculate total transmission capacity
    transmission_capacity = 0
    for link in network.links.index:
        transmission_capacity += network.links.loc[link, "p_nom_opt"]
    
    # Calculate system costs
    system_cost = network.objective / 1e6  # in million €
    
    # Create a summary text
    summary = []
    summary.append("=" * 80)
    summary.append(f"POWER SYSTEM MODEL RESULTS SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    summary.append("SYSTEM OVERVIEW")
    summary.append("-" * 80)
    summary.append(f"Total System Cost: {system_cost:.2f} million €")
    summary.append(f"Total Generation Capacity: {sum(generation_by_tech.values()):.2f} MW")
    summary.append(f"Total Transmission Capacity: {transmission_capacity:.2f} MW")
    summary.append("")
    
    summary.append("GENERATION CAPACITY BY TECHNOLOGY")
    summary.append("-" * 80)
    for tech, capacity in generation_by_tech.items():
        summary.append(f"{tech.capitalize()}: {capacity:.2f} MW ({capacity/sum(generation_by_tech.values())*100:.1f}%)")
    summary.append("")
    
    summary.append("GENERATION CAPACITY BY COUNTRY")
    summary.append("-" * 80)
    for country in ["ESP", "FRA", "PRT"]:
        country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        capacity = 0
        for gen in country_gens:
            if network.generators.loc[gen, "p_nom_extendable"]:
                capacity += network.generators.loc[gen, "p_nom_opt"]
            else:
                capacity += network.generators.loc[gen, "p_nom"]
        summary.append(f"{country}: {capacity:.2f} MW")
    summary.append("")
    
    summary.append("RENEWABLE ENERGY STATISTICS")
    summary.append("-" * 80)
    for idx, row in renewable_stats.iterrows():
        summary.append(f"{row['Metric']}: {row['Value']}")
    summary.append("")
    
    summary.append("TRANSMISSION UTILIZATION")
    summary.append("-" * 80)
    for link in transmission_df.index:
        summary.append(f"{link}:")
        summary.append(f"  - Average Utilization: {transmission_df.loc[link, 'Average Utilization (%)']:.1f}%")
        summary.append(f"  - Congestion Hours: {transmission_df.loc[link, 'Congestion Hours']:.0f} " + 
                      f"({transmission_df.loc[link, '% Time Congested']:.1f}% of time)")
    summary.append("")
    
    summary.append("NOTABLE FINDINGS")
    summary.append("-" * 80)
    
    # Add findings based on results (this could be expanded with more sophisticated analysis)
    if renewable_stats.loc[0, 'Value'].rstrip('%') > "50":
        summary.append("- The system achieves a high level of renewable penetration, with renewables providing " +
                      f"an average of {renewable_stats.loc[0, 'Value']} of electricity.")
    
    # Check for high congestion
    high_congestion = False
    for link in transmission_df.index:
        if transmission_df.loc[link, '% Time Congested'] > 10:
            high_congestion = True
            summary.append(f"- The {link} transmission line experiences significant congestion " +
                          f"({transmission_df.loc[link, '% Time Congested']:.1f}% of time), indicating a need for " +
                          "potential capacity expansion.")
    
    if not high_congestion:
        summary.append("- Transmission lines appear adequately sized with limited congestion.")
    
    # Look at optimal generation mix
    if "gas" in generation_by_tech and generation_by_tech["gas"] > 0:
        summary.append(f"- Gas generation capacity of {generation_by_tech['gas']:.1f} MW is needed for system reliability.")
    
    wind_solar_ratio = 0
    if "solar" in generation_by_tech and generation_by_tech["solar"] > 0:
        if "onshorewind" in generation_by_tech:
            wind_solar_ratio = generation_by_tech["onshorewind"] / generation_by_tech["solar"]
            summary.append(f"- The optimal wind-to-solar capacity ratio is approximately {wind_solar_ratio:.2f}.")
    
    # Add a timestamp
    summary.append("")
    summary.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 80)
    
    # Join all lines with newlines
    full_summary = "\n".join(summary)
    
    # Save to file
    with open(f"{output_dir}/model_summary_report.txt", "w") as f:
        f.write(full_summary)
    
    # Print the summary
    print(full_summary)


def plot_generation_capacity(network):
    """Plot the generation capacity by technology and country."""
    # Define colors for technologies
    tech_colors = {
        "onshorewind": "#3498db",  # Blue
        "solar": "#f1c40f",       # Yellow
        "gas": "#e74c3c",         # Red
        "nuclear": "#9b59b6",     # Purple
        "hydro": "#2ecc71"        # Green
    }

    # Prepare data for plotting
    tech_capacities = {}
    for tech in tech_colors.keys():
        tech_capacities[tech] = []
        for country in ["ESP", "FRA", "PRT"]:
            gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ") and tech in gen]
            capacity = sum(network.generators.loc[gen, "p_nom_opt"] if network.generators.loc[gen, "p_nom_extendable"] 
                           else network.generators.loc[gen, "p_nom"] for gen in gens)
            tech_capacities[tech].append(capacity)

    # Plot capacities as a grouped bar chart
    countries = ["ESP", "FRA", "PRT"]
    x = np.arange(len(countries))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (tech, capacities) in enumerate(tech_capacities.items()):
        ax.bar(x + i * width, capacities, width, label=tech.capitalize(), color=tech_colors[tech])

    # Add labels and formatting
    ax.set_ylabel("Capacity (MW)")
    ax.set_title("Generation Capacity by Technology and Country")
    ax.set_xticks(x + width * (len(tech_capacities) - 1) / 2)
    ax.set_xticklabels(countries)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/generation_capacity.png", dpi=300)
    plt.close()


# Main execution
if __name__ == "__main__":
    print("\nStarting PyPSA Power System Modeling for Three Countries...")
    
    # Run the optimization
    network.optimize(
       
        solver_name="gurobi",  # Can be changed to other solvers like 'glpk', 'cbc', etc.
        
        solver_options={"threads": 4, "method": 1, "crossover": 0}
     
    )
    
    print("\nOptimization complete. Processing results...")
    
    # Create output directory for results if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Display and save results
    print("\nOptimal generation capacities:")
    for gen in network.generators.index:
        if network.generators.loc[gen, "p_nom_extendable"]:
            opt_cap = network.generators.loc[gen, "p_nom_opt"]
            print(f"{gen}: {opt_cap:.2f} MW")
    
    print("\nOptimal transmission capacities:")
    for link in network.links.index:
        if network.links.loc[link, "p_nom_extendable"]:
            opt_cap = network.links.loc[link, "p_nom_opt"]
            print(f"{link}: {opt_cap:.2f} MW")
    
    # Generate visualizations
    print("\nCreating visualizations...")
    plot_generation_capacity(network)
    create_energy_flow_visualizations(network)
    
    # Generate analysis
    print("\nAnalyzing renewable integration...")
    renewable_stats = analyze_renewable_statistics(network)
    
    print("\nAnalyzing transmission impacts...")
    transmission_df = analyze_transmission_impacts(network)
    
    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(network, renewable_stats, transmission_df)
    
    print(f"\nAll results have been saved to the '{output_dir}' directory.")
    print("\nPyPSA Power System Modeling complete!")

#
#
#
###FRENCH NUCLEAR PHASE OUT
#
#
#

import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import warnings
from shapely.errors import ShapelyDeprecationWarning
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
output_dir = 'pypsa_output_nuclear_phaseout'
os.makedirs(output_dir, exist_ok=True)

network = pypsa.Network()

hours_in_2015 = pd.date_range('2015-01-01 00:00Z', '2015-12-31 23:00Z', freq='h')
network.set_snapshots(hours_in_2015.values)

countries = ['ESP', 'FRA', 'PRT']

for country in countries:
    network.add("Bus", f"{country} electricity bus")

print("Loading electricity demand data for all countries")
data_path = 'C:/Users/nasos/DTU/IEG(Env and source data)/'
df_elec = pd.read_csv(f'{data_path}electricity_demand.csv', sep=';', index_col=0)
df_elec.index = pd.to_datetime(df_elec.index)

for country in countries:
    print(f"Adding load data for {country}")
    network.add("Load", 
                f"{country} load",
                bus=f"{country} electricity bus",
                p_set=df_elec[country].values)

def annuity(n, r):
    if r > 0:
        return r/(1 - 1/(1 + r)**n)
    else:
        return 1/n

print("Loading solar capacity factors for all countries")
solar_cf = pd.read_csv(f'{data_path}pv_optimal.csv', sep=';', index_col=0)
solar_cf.index = pd.to_datetime(solar_cf.index)

formatted_snapshots = [hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]
CF_solar_ESP = solar_cf['ESP'][formatted_snapshots]
CF_solar_FRA = solar_cf['FRA'][formatted_snapshots]
CF_solar_PRT = solar_cf['PRT'][formatted_snapshots]

print("Loading wind capacity factors for all countries")
wind_cf = pd.read_csv(f'{data_path}onshore_wind_1979-2017.csv', sep=';', index_col=0)
wind_cf.index = pd.to_datetime(wind_cf.index)
CF_wind_ESP = wind_cf['ESP'][formatted_snapshots]
CF_wind_FRA = wind_cf['FRA'][formatted_snapshots]
CF_wind_PRT = wind_cf['PRT'][formatted_snapshots]

tech_params = {
    "onshorewind": {
        "lifetime": 30,
        "capital_costs": {
            "ESP": 1000000,
            "FRA": 1100000,
            "PRT": 950000
        },
        "fixed_om": 25000
    },
    "solar": {
        "lifetime": 25,
        "capital_costs": {
            "ESP": 600000,
            "FRA": 700000,
            "PRT": 650000
        },
        "fixed_om": 20000
    },
    "gas": {
        "lifetime": 25,
        "capital_costs": {
            "ESP": 800000,
            "FRA": 850000,
            "PRT": 820000
        },
        "marginal_costs": {
            "ESP": 50,
            "FRA": 55,
            "PRT": 53
        },
        "fixed_om": 20000
    },
    "hydro": {
        "lifetime": 80,
        "capital_cost": 2000000,
        "marginal_cost": 5,
        "fixed_om": 45000
    }
}

discount_rate = 0.07

print("Adding generators for Spain (ESP)")
network.add("Generator",
            "ESP onshorewind",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=100000,
            marginal_cost=0,
            capital_cost=annuity(tech_params["onshorewind"]["lifetime"], discount_rate) * 
                       (tech_params["onshorewind"]["capital_costs"]["ESP"] + 
                        tech_params["onshorewind"]["fixed_om"] / discount_rate),
            p_max_pu=CF_wind_ESP.values)

network.add("Generator",
            "ESP solar",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=100000,
            marginal_cost=0,
            capital_cost=annuity(tech_params["solar"]["lifetime"], discount_rate) * 
                       (tech_params["solar"]["capital_costs"]["ESP"] + 
                        tech_params["solar"]["fixed_om"] / discount_rate),
            p_max_pu=CF_solar_ESP.values)

network.add("Generator",
            "ESP gas",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            marginal_cost=tech_params["gas"]["marginal_costs"]["ESP"],
            capital_cost=annuity(tech_params["gas"]["lifetime"], discount_rate) * 
                       (tech_params["gas"]["capital_costs"]["ESP"] + 
                        tech_params["gas"]["fixed_om"] / discount_rate))

print("Adding generators for France (FRA) -- Nuclear PHASE-OUT")
network.add("Generator",
            "FRA onshorewind",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(tech_params["onshorewind"]["lifetime"], discount_rate) * 
                       (tech_params["onshorewind"]["capital_costs"]["FRA"] + 
                        tech_params["onshorewind"]["fixed_om"] / discount_rate),
            p_max_pu=CF_wind_FRA.values)

network.add("Generator",
            "FRA solar",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(tech_params["solar"]["lifetime"], discount_rate) * 
                       (tech_params["solar"]["capital_costs"]["FRA"] + 
                        tech_params["solar"]["fixed_om"] / discount_rate),
            p_max_pu=CF_solar_FRA.values)

network.add("Generator",
            "FRA gas",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            marginal_cost=tech_params["gas"]["marginal_costs"]["FRA"],
            capital_cost=annuity(tech_params["gas"]["lifetime"], discount_rate) * 
                       (tech_params["gas"]["capital_costs"]["FRA"] + 
                        tech_params["gas"]["fixed_om"] / discount_rate))

print("Adding generators for Portugal (PRT)")
network.add("Generator",
            "PRT hydro",
            bus="PRT electricity bus",
            p_nom=7000,
            p_nom_extendable=False,
            marginal_cost=tech_params["hydro"]["marginal_cost"],
            p_max_pu=0.5)

network.add("Generator",
            "PRT onshorewind",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(tech_params["onshorewind"]["lifetime"], discount_rate) * 
                       (tech_params["onshorewind"]["capital_costs"]["PRT"] + 
                        tech_params["onshorewind"]["fixed_om"] / discount_rate),
            p_max_pu=CF_wind_PRT.values)

network.add("Generator",
            "PRT solar",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(tech_params["solar"]["lifetime"], discount_rate) * 
                       (tech_params["solar"]["capital_costs"]["PRT"] + 
                        tech_params["solar"]["fixed_om"] / discount_rate),
            p_max_pu=CF_solar_PRT.values)

network.add("Generator",
            "PRT gas",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            marginal_cost=tech_params["gas"]["marginal_costs"]["PRT"],
            capital_cost=annuity(tech_params["gas"]["lifetime"], discount_rate) * 
                       (tech_params["gas"]["capital_costs"]["PRT"] + 
                        tech_params["gas"]["fixed_om"] / discount_rate))

print("Adding transmission links between countries")
transmission_params = {
    "ESP-FRA": {
        "p_nom": 2800,
        "capital_cost": 900000,
        "lifetime": 40
    },
    "ESP-PRT": {
        "p_nom": 4200,
        "capital_cost": 750000,
        "lifetime": 40
    }
}

network.add("Link",
            "ESP-FRA",
            bus0="ESP electricity bus",
            bus1="FRA electricity bus",
            p_nom=transmission_params["ESP-FRA"]["p_nom"],
            p_nom_extendable=True,
            p_min_pu=-1,
            capital_cost=annuity(transmission_params["ESP-FRA"]["lifetime"], discount_rate) * 
                       transmission_params["ESP-FRA"]["capital_cost"])

network.add("Link",
            "ESP-PRT",
            bus0="ESP electricity bus",
            bus1="PRT electricity bus",
            p_nom=transmission_params["ESP-PRT"]["p_nom"],
            p_nom_extendable=True,
            p_min_pu=-1,
            capital_cost=annuity(transmission_params["ESP-PRT"]["lifetime"], discount_rate) * 
                       transmission_params["ESP-PRT"]["capital_cost"])

if __name__ == "__main__":
    print("\nRunning PyPSA with France nuclear phase-out...\n")
    network.optimize(solver_name='gurobi', solver_options={"threads": 4, "method": 1, "crossover": 0})
    print("Optimization complete.")

    print("\nResulting generation capacities (MW):")
    for gen in network.generators.index:
        if network.generators.loc[gen, "p_nom_extendable"]:
            print(f"{gen}: {network.generators.loc[gen, 'p_nom_opt']:.1f} MW")
        else:
            print(f"{gen}: {network.generators.loc[gen, 'p_nom']:.1f} MW (fixed)")

    print("\nResulting transmission capacities (MW):")
    for link in network.links.index:
        print(f"{link}: {network.links.loc[link, 'p_nom_opt']:.1f} MW")

    print(f"\nTotal system cost: {network.objective/8760:.2f} €/h or {network.objective/1e6:.2f} M€/yr (approx.)")
