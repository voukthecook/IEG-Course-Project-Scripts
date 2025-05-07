import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt


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
df_elec = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/electricity_demand.csv', sep=';', index_col=0) # in MWh
df_elec.index = pd.to_datetime(df_elec.index)


# Add load for each country
for country in countries:
    print(f"Adding load data for {country}")
    network.add("Load", 
                f"{country} load",
                bus=f"{country} electricity bus",
                p_set=df_elec[country].values)  # Assuming column names match country codes


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
solar_cf = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/pv_optimal.csv', sep=';', index_col=0)
solar_cf.index = pd.to_datetime(solar_cf.index)
CF_solar_ESP = solar_cf['ESP'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
CF_solar_FRA = solar_cf['FRA'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
CF_solar_PRT = solar_cf['PRT'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]


print("Loading wind capacity factors for all countries")
wind_cf = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/onshore_wind_1979-2017.csv', sep=';', index_col=0)
wind_cf.index = pd.to_datetime(wind_cf.index)
CF_wind_ESP = wind_cf['ESP'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
CF_wind_FRA = wind_cf['FRA'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
CF_wind_PRT = wind_cf['PRT'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]


# 5. Add generators for each country
# For Spain (ESP)
print("Adding generators for Spain (ESP)")
network.add("Generator",
            "ESP onshorewind",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=100000,
            marginal_cost=0,
            capital_cost=annuity(30, 0.07) * 1000000,  # €/MW
            p_max_pu=CF_wind_ESP.values)


network.add("Generator",
            "ESP solar",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            p_nom_max=100000,
            marginal_cost=0,
            capital_cost=annuity(25, 0.07) * 600000,  # €/MW
            p_max_pu=CF_solar_ESP.values)


# Add gas as backup for Spain
network.add("Generator",
            "ESP gas",
            bus="ESP electricity bus",
            p_nom_extendable=True,
            marginal_cost=50,  # €/MWh
            capital_cost=annuity(25, 0.07) * 800000)  # €/MW


# For France (FRA)
print("Adding generators for France (FRA)")
# Add nuclear with fixed capacity and constant availability (not reading from file)
network.add("Generator",
            "FRA nuclear",
            bus="FRA electricity bus",
            p_nom=63000,  # MW of nuclear capacity
            p_nom_extendable=False,
            marginal_cost=20,  # €/MWh
            p_max_pu=0.85)  # Constant availability factor of 85%


network.add("Generator",
            "FRA onshorewind",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(30, 0.07) * 1100000,  # €/MW
            p_max_pu=CF_wind_FRA.values)


network.add("Generator",
            "FRA solar",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(25, 0.07) * 700000,  # €/MW
            p_max_pu=CF_solar_FRA.values)


# Add gas as backup for France
network.add("Generator",
            "FRA gas",
            bus="FRA electricity bus",
            p_nom_extendable=True,
            marginal_cost=55,  # €/MWh
            capital_cost=annuity(25, 0.07) * 850000)  # €/MW


# For Portugal (PRT)
print("Adding generators for Portugal (PRT)")
# Add hydro with fixed capacity and constant availability (not reading from file)
network.add("Generator",
            "PRT hydro",
            bus="PRT electricity bus",
            p_nom=7000,  # MW of hydro capacity
            p_nom_extendable=False,
            marginal_cost=5,  # €/MWh
            p_max_pu=0.5)  # Simplified constant availability of 50%


network.add("Generator",
            "PRT onshorewind",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(30, 0.07) * 950000,  # €/MW
            p_max_pu=CF_wind_PRT.values)


network.add("Generator",
            "PRT solar",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            p_nom_min=0,
            marginal_cost=0,
            capital_cost=annuity(25, 0.07) * 650000,  # €/MW
            p_max_pu=CF_solar_PRT.values)


# Add gas as backup for Portugal
network.add("Generator",
            "PRT gas",
            bus="PRT electricity bus",
            p_nom_extendable=True,
            marginal_cost=53,  # €/MWh
            capital_cost=annuity(25, 0.07) * 820000)  # €/MW


# 6. Add transmission links between countries
print("Adding transmission links between countries")
network.add("Link",
            "ESP-FRA",
            bus0="ESP electricity bus",
            bus1="FRA electricity bus",
            p_nom=2800,  # MW - current capacity
            p_nom_extendable=True,  # Allow optimization of capacity
            p_min_pu=-1,  # Allow bidirectional flow
            capital_cost=annuity(40, 0.07) * 900000)  # €/MW


network.add("Link",
            "ESP-PRT",
            bus0="ESP electricity bus",
            bus1="PRT electricity bus",
            p_nom=4200,  # MW - current capacity
            p_nom_extendable=True,  # Allow optimization of capacity
            p_min_pu=-1,  # Allow bidirectional flow
            capital_cost=annuity(40, 0.07) * 750000)  # €/MW


# 7. Run the optimization
print("Running optimization for interconnected ESP, FRA, and PRT system")
network.optimize( solver_name='gurobi')


# 8. Analyze results - printing optimal capacities
print("\nOptimal generation capacities (MW):")
print("Spain (ESP):")
print(f"  Wind: {network.generators.loc['ESP onshorewind', 'p_nom_opt']:.1f}")
print(f"  Solar: {network.generators.loc['ESP solar', 'p_nom_opt']:.1f}")
print(f"  Gas: {network.generators.loc['ESP gas', 'p_nom_opt']:.1f}")


print("\nFrance (FRA):")
print(f"  Nuclear: {network.generators.loc['FRA nuclear', 'p_nom']:.1f} (fixed)")
print(f"  Wind: {network.generators.loc['FRA onshorewind', 'p_nom_opt']:.1f}")
print(f"  Solar: {network.generators.loc['FRA solar', 'p_nom_opt']:.1f}")
print(f"  Gas: {network.generators.loc['FRA gas', 'p_nom_opt']:.1f}")


print("\nPortugal (PRT):")
print(f"  Hydro: {network.generators.loc['PRT hydro', 'p_nom']:.1f} (fixed)")
print(f"  Wind: {network.generators.loc['PRT onshorewind', 'p_nom_opt']:.1f}")
print(f"  Solar: {network.generators.loc['PRT solar', 'p_nom_opt']:.1f}")
print(f"  Gas: {network.generators.loc['PRT gas', 'p_nom_opt']:.1f}")


print("\nOptimal transmission capacities (MW):")
print(f"ESP-FRA: {network.links.loc['ESP-FRA', 'p_nom_opt']:.1f}")
print(f"ESP-PRT: {network.links.loc['ESP-PRT', 'p_nom_opt']:.1f}")


# 9. Calculate energy mix and system costs
total_demand = {country: df_elec[country].sum() for country in countries}
total_generation = {}


for country in countries:
    country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
    gen_by_type = {}
    for gen in country_gens:
        gen_type = gen.replace(f"{country} ", "")
        gen_by_type[gen_type] = network.generators_t.p[gen].sum()
    total_generation[country] = gen_by_type


print("\nEnergy Generation Mix (GWh):")
for country in countries:
    print(f"\n{country}:")
    for gen_type, energy in total_generation[country].items():
        percentage = (energy / total_demand[country]) * 100
        print(f"  {gen_type}: {energy/1000:.1f} GWh ({percentage:.1f}%)")


# 10. Plot power generation for each country (first week)
for country in countries:
    fig, ax = plt.subplots(figsize=(10, 6))
    generators_country = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
    generation_country = network.generators_t.p[generators_country].loc[network.snapshots[:168]]  # First week
    generation_country.columns = [col.replace(f"{country} ", "") for col in generation_country.columns]
    generation_country.plot.area(ax=ax, stacked=True)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power [MW]")
    ax.set_title(f"Electricity Generation in {country} - First Week")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{country.lower()}_generation.png")


# 11. Plot transmission flows
fig, ax = plt.subplots(figsize=(10, 4))
links = network.links_t.p0[["ESP-FRA", "ESP-PRT"]].loc[network.snapshots[:168]]  # First week
links.plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel("Power [MW]")
ax.set_title("Transmission Flows Between Countries (Positive: Export from Spain)")
ax.legend(["ESP → FRA", "ESP → PRT"])
plt.tight_layout()
plt.savefig("transmission_flows.png")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from shapely.errors import ShapelyDeprecationWarning
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.lines import Line2D

# Suppress ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

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
    
    # Convert timestamp to string for title
    time_str = pd.to_datetime(max_flow_time).strftime('%Y-%m-%d %H:%M')
    ax3.set_title(f"Energy Flow at Peak Time ({time_str})", fontsize=14)
    
    # PLOT 4: Flow variability visualization
    # Calculate standard deviation of flows
    flow_std = network.links_t.p0.std()
    
    # Plot network with flow variability
    collections = network.plot(
        ax=ax4,
        bus_sizes=bus_sizes/100000,
        bus_colors=bus_colors,
        line_widths=0,
        margin=0.2,
        color_geomap=True
    )
    
    # Add links manually with variability indicated by width
    for link_name in network.links.index:
        bus0 = network.links.loc[link_name, "bus0"]
        bus1 = network.links.loc[link_name, "bus1"]
        
        # Get coordinates
        x0, y0 = network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"]
        x1, y1 = network.buses.loc[bus1, "x"], network.buses.loc[bus1, "y"]
        
        # Get flow standard deviation
        std_dev = flow_std[link_name]
        
        # Width based on variability (std dev)
        width = std_dev / 10000
        
        # Draw bidirectional arrow (since we're showing variability)
        ax4.annotate(
            "",
            xy=(x1, y1), xycoords='data',
            xytext=(x0, y0), textcoords='data',
            arrowprops=dict(
                arrowstyle="<->",
                connectionstyle="arc3",
                lw=width*5,
                color="purple",
                alpha=0.7
            )
        )
        
        # Add variability label
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        ax4.text(
            mid_x, mid_y,
            f"σ = {std_dev:.0f} MW",
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7),
            ha='center', va='center'
        )
    
    ax4.set_title("Energy Flow Variability Visualization", fontsize=14)
    
    # PLOT 5: Time series of flows for a sample period
    ax5.plot(link_flows_week.index, link_flows_week["ESP-FRA"], linewidth=2, label="ESP → FRA")
    ax5.plot(link_flows_week.index, link_flows_week["ESP-PRT"], linewidth=2, label="ESP → PRT")
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Format x-axis to show dates nicely
    ax5.set_xlabel("Date", fontsize=12)
    ax5.set_ylabel("Power Flow (MW)", fontsize=12)
    ax5.set_title("Power Flows Between Countries (Sample Period)", fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="best")
    
    # Format x-axis with dates
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    # Add overall title
    fig.suptitle("Cross-Border Energy Flows Analysis", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the visualization
    plt.savefig('energy_flows_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional visualizations: Annual flow duration curves
    plt.figure(figsize=(12, 6))
    
    # Sort flows in descending order
    esp_fra_sorted = network.links_t.p0["ESP-FRA"].sort_values(ascending=False).reset_index(drop=True)
    esp_prt_sorted = network.links_t.p0["ESP-PRT"].sort_values(ascending=False).reset_index(drop=True)
    
    # Create x-axis as percentage of time
    hours = len(esp_fra_sorted)
    x_axis = np.arange(hours) / hours * 100
    
    plt.plot(x_axis, esp_fra_sorted, linewidth=2, label="ESP → FRA")
    plt.plot(x_axis, esp_prt_sorted, linewidth=2, label="ESP → PRT")
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title("Annual Flow Duration Curves", fontsize=14)
    plt.xlabel("Percentage of Time (%)", fontsize=12)
    plt.ylabel("Power Flow (MW)", fontsize=12)
    plt.legend()
    
    plt.savefig('flow_duration_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create pie charts of generation mix for each country
    countries = ["ESP", "FRA", "PRT"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, country in enumerate(countries):
        country_gens = gen_by_country_carrier[country]
        
        # Extract data for pie chart
        carriers = list(country_gens.keys())
        values = list(country_gens.values())
        colors_list = [carrier_colors.get(carrier, "gray") for carrier in carriers]
        
        # Create pie chart
        axes[i].pie(
            values, 
            labels=carriers, 
            colors=colors_list, 
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        axes[i].set_title(f"{country} Average Generation Mix", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('generation_mix_pie_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

# Add this at the end of your existing code to create the visualizations
create_energy_flow_visualizations(network)
