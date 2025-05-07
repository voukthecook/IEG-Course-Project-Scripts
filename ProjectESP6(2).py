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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Suppress ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

def create_all_enhanced_visualizations(network):

    """Create enhanced energy flow visualizations with directional arrows and energy mix coloring."""
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
    
    # Create a map using Cartopy
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add country borders and coastlines
    ax.coastlines(resolution='50m')

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
    
    # Calculate average generation by carrier for pie charts and mixed coloring
    gen_by_country_carrier = {}
    for country in ["ESP", "FRA", "PRT"]:
        country_gens = [gen for gen in network.generators.index if gen.startswith(f"{country} ")]
        gen_by_carrier = {}
        
        for gen in country_gens:
            carrier = gen.split(' ')[1]  # Extract the carrier type from generator name
            if carrier not in gen_by_carrier:
                gen_by_carrier[carrier] = 0
            gen_by_carrier[carrier] += network.generators_t.p[gen].mean()
            
        gen_by_country_carrier[country] = gen_by_carrier
    
    # PLOT 1: Energy Mix Visualization with Pie Charts
    bus_sizes = pd.Series(gen_by_country)
    
    # Plot the network base map
    collections = network.plot(
        ax=ax1,
        bus_sizes=0,  # We'll add custom circles for buses
        line_widths=0,  # No lines since we're using links
        margin=0.2,
        color_geomap=True
    )
    
    # Add energy mix pie charts for each country
    for country in ["ESP", "FRA", "PRT"]:
        bus = f"{country} electricity bus"
        x, y = network.buses.loc[bus, "x"], network.buses.loc[bus, "y"]
        
        # Get generation mix for this country
        country_mix = gen_by_country_carrier[country]
        
        # Calculate total generation
        total_gen = sum(country_mix.values())
        
        # Calculate size of the pie based on total generation
        size = total_gen / 20000
        
        # Create small pie chart for the energy mix
        carriers = list(country_mix.keys())
        values = list(country_mix.values())
        colors_list = [carrier_colors.get(carrier, "gray") for carrier in carriers]
        
        # Create a small axes for the pie chart
        pie_size = size * 0.1  # Adjust for visibility
        pie_ax = plt.axes([0, 0, 1, 1], projection=None)
        pie_ax.set_xlim(0, 1)
        pie_ax.set_ylim(0, 1)
        pie_ax.set_axis_off()
        
        # Convert geographic coordinates to figure coordinates
        fig_coord = ax1.transData.transform((x, y))
        fig_coord = fig.transFigure.inverted().transform(fig_coord)
        
        # Create inset axes for pie chart
        inset_ax = fig.add_axes([
            fig_coord[0] - pie_size/2, 
            fig_coord[1] - pie_size/2, 
            pie_size, 
            pie_size
        ])
        
        # Create pie chart
        if sum(values) > 0:
            wedges, texts, autotexts = inset_ax.pie(
                values, 
                colors=colors_list,
                autopct=lambda p: f'{p:.0f}%' if p > 5 else '',
                startangle=90,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Make percentage text smaller and white for better visibility
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        inset_ax.set_aspect('equal')
        
        # Add country label
        ax1.text(
            x, y - 2,  # Position below the pie chart
            country,
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )
        
        # Draw the transmission links with enhanced directional arrows
        for link_name in network.links.index:
            bus0 = network.links.loc[link_name, "bus0"]
            bus1 = network.links.loc[link_name, "bus1"]
            
            # Get coordinates
            x0, y0 = network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"]
            x1, y1 = network.buses.loc[bus1, "x"], network.buses.loc[bus1, "y"]
            
            # Get optimal capacity for thickness
            capacity = network.links.loc[link_name, "p_nom_opt"]
            width = capacity / 20000
            
            # Create arrow with a pronounced head for direction
            ax1.annotate(
                "",
                xy=(x1, y1), xycoords='data',
                xytext=(x0, y0), textcoords='data',
                arrowprops=dict(
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.1",
                    lw=width*5,
                    color="darkblue",
                    alpha=0.8,
                    mutation_scale=20  # Larger arrowhead
                )
            )
            
            # Add capacity label
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            ax1.text(
                mid_x, mid_y, 
                f"{capacity:.0f} MW",
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7),
                ha='center', va='center'
            )
    
    ax1.set_title("Energy Mix and Capacity Visualization", fontsize=14)
    
    # Add legend for generation types
    legend_elements = []
    for carrier, color in carrier_colors.items():
        legend_elements.append(
            Patch(facecolor=color, edgecolor='w', label=carrier)
        )
    
    ax1.legend(handles=legend_elements, loc='lower left', title="Generation Types")
    
    # PLOT 2: Flow Visualization with Bidirectional Arrows
    # Plot network base map
    collections = network.plot(
        ax=ax2,
        bus_sizes=bus_sizes/10000,
        bus_colors="lightgray",  # Neutral color for buses
        line_widths=0,
        margin=0.2,
        color_geomap=True
    )
    
    # Add country labels
    for country in ["ESP", "FRA", "PRT"]:
        bus = f"{country} electricity bus"
        x, y = network.buses.loc[bus, "x"], network.buses.loc[bus, "y"]
        ax2.text(
            x, y + 1.5,  # Position above the node
            country,
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )
    
    # Add links with flow direction and magnitude
    for link_name in network.links.index:
        bus0 = network.links.loc[link_name, "bus0"]
        bus1 = network.links.loc[link_name, "bus1"]
        
        # Extract country codes from bus names
        country0 = bus0.split(' ')[0]
        country1 = bus1.split(' ')[0]
        
        # Get coordinates
        x0, y0 = network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"]
        x1, y1 = network.buses.loc[bus1, "x"], network.buses.loc[bus1, "y"]
        
        # Get average flow (positive: bus0→bus1, negative: bus1→bus0)
        avg_flow = link_flow[link_name]
        
        # Width based on absolute flow
        width = abs(avg_flow) / 2000
        
        # Determine flow direction
        if avg_flow >= 0:
            # Flow from bus0 to bus1
            start_x, start_y = x0, y0
            end_x, end_y = x1, y1
            direction_text = f"{country0} → {country1}"
            color = "green"
        else:
            # Flow from bus1 to bus0
            start_x, start_y = x1, y1
            end_x, end_y = x0, y0
            direction_text = f"{country1} → {country0}"
            color = "red"
        
        # Draw arrow for the main direction
        ax2.annotate(
            "",
            xy=(end_x, end_y), xycoords='data',
            xytext=(start_x, start_y), textcoords='data',
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
                lw=width*5,
                color=color,
                alpha=0.8,
                mutation_scale=20  # Larger arrowhead
            )
        )
        
        # Add bidirectional flow indicator (thinner line in opposite direction)
        if abs(avg_flow) < network.links.loc[link_name, "p_nom_opt"] * 0.8:
            # Draw smaller arrow in opposite direction to indicate bidirectional potential
            ax2.annotate(
                "",
                xy=(start_x, start_y), xycoords='data',
                xytext=(end_x, end_y), textcoords='data',
                arrowprops=dict(
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.1",
                    lw=width*2,  # Thinner line
                    color="gray",
                    alpha=0.4,
                    mutation_scale=10  # Smaller arrowhead
                )
            )
        
        # Add flow label with direction information
        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        ax2.text(
            mid_x, mid_y,
            f"{abs(avg_flow):.0f} MW\n{direction_text}",
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7),
            ha='center', va='center'
        )
    
    ax2.set_title("Average Energy Flow with Direction Indicators", fontsize=14)
    
    # PLOT 3: Hourly Flow Direction
    # Find time with maximum flow in each direction for each link
    max_pos_flows = {}
    max_neg_flows = {}
    
    for link_name in network.links.index:
        link_data = network.links_t.p0[link_name]
        max_pos_time = link_data[link_data > 0].idxmax() if any(link_data > 0) else None
        max_neg_time = link_data[link_data < 0].idxmin() if any(link_data < 0) else None
        
        max_pos_flows[link_name] = (max_pos_time, link_data.loc[max_pos_time] if max_pos_time else 0)
        max_neg_flows[link_name] = (max_neg_time, link_data.loc[max_neg_time] if max_neg_time else 0)
    
    # Plot network base map
    collections = network.plot(
        ax=ax3,
        bus_sizes=bus_sizes/10000,
        bus_colors="lightgray",
        line_widths=0,
        margin=0.2,
        color_geomap=True
    )
    
    # Add country labels
    for country in ["ESP", "FRA", "PRT"]:
        bus = f"{country} electricity bus"
        x, y = network.buses.loc[bus, "x"], network.buses.loc[bus, "y"]
        ax3.text(
            x, y + 1.5,
            country,
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )
    
    # Add links with maximum flow in each direction
    for link_name in network.links.index:
        bus0 = network.links.loc[link_name, "bus0"]
        bus1 = network.links.loc[link_name, "bus1"]
        
        # Extract country codes from bus names
        country0 = bus0.split(' ')[0]
        country1 = bus1.split(' ')[0]
        
        # Get coordinates
        x0, y0 = network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"]
        x1, y1 = network.buses.loc[bus1, "x"], network.buses.loc[bus1, "y"]
        
        # Get maximum positive flow (bus0 to bus1)
        max_pos_time, max_pos_flow = max_pos_flows[link_name]
        
        # Get maximum negative flow (bus1 to bus0)
        max_neg_time, max_neg_flow = max_neg_flows[link_name]
        
        # Draw arrow for maximum positive flow if exists
        if max_pos_time is not None:
            width = abs(max_pos_flow) / 2000
            
            # Slightly offset the arrows for visibility
            dx, dy = (x1 - x0) * 0.05, (y1 - y0) * 0.05
            
            ax3.annotate(
                "",
                xy=(x1 - dx, y1 - dy), xycoords='data',
                xytext=(x0 + dx, y0 + dy), textcoords='data',
                arrowprops=dict(
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.1",
                    lw=width*5,
                    color="green",
                    alpha=0.8,
                    mutation_scale=20
                )
            )
            
            # Add flow label for positive direction
            mid_x, mid_y = (x0 + x1) / 2 + dx * 2, (y0 + y1) / 2 + dy * 2
            time_str = pd.to_datetime(max_pos_time).strftime('%m-%d %H:%M')
            ax3.text(
                mid_x, mid_y,
                f"{country0}→{country1}\n{abs(max_pos_flow):.0f} MW\n{time_str}",
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='green'),
                ha='center', va='center'
            )
        
        # Draw arrow for maximum negative flow if exists
        if max_neg_time is not None:
            width = abs(max_neg_flow) / 2000
            
            # Slightly offset in the opposite direction
            dx, dy = (x1 - x0) * 0.05, (y1 - y0) * 0.05
            
            ax3.annotate(
                "",
                xy=(x0 - dx, y0 - dy), xycoords='data',
                xytext=(x1 + dx, y1 + dy), textcoords='data',
                arrowprops=dict(
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=-0.1",
                    lw=width*5,
                    color="red",
                    alpha=0.8,
                    mutation_scale=20
                )
            )
            
            # Add flow label for negative direction
            mid_x, mid_y = (x0 + x1) / 2 - dx * 2, (y0 + y1) / 2 - dy * 2
            time_str = pd.to_datetime(max_neg_time).strftime('%m-%d %H:%M')
            ax3.text(
                mid_x, mid_y,
                f"{country1}→{country0}\n{abs(max_neg_flow):.0f} MW\n{time_str}",
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'),
                ha='center', va='center'
            )
    
    ax3.set_title("Maximum Flows in Each Direction with Timestamps", fontsize=14)
    
    # PLOT 4: Flow Patterns by Time of Day
    # Calculate average flows by hour of day
    hourly_flows = {}
    for link_name in network.links.index:
        flow_data = network.links_t.p0[link_name]
        hourly_avg = flow_data.groupby(flow_data.index.hour).mean()
        hourly_flows[link_name] = hourly_avg
    
    # Plot network base map
    collections = network.plot(
        ax=ax4,
        bus_sizes=bus_sizes/10000,
        bus_colors="lightgray",
        line_widths=0,
        margin=0.2,
        color_geomap=True
    )
    
    # Add country labels
    for country in ["ESP", "FRA", "PRT"]:
        bus = f"{country} electricity bus"
        x, y = network.buses.loc[bus, "x"], network.buses.loc[bus, "y"]
        ax4.text(
            x, y + 1.5,
            country,
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )
    
    # Add clock-based flow visualization for each link
    for link_name in network.links.index:
        bus0 = network.links.loc[link_name, "bus0"]
        bus1 = network.links.loc[link_name, "bus1"]
        
        # Extract country codes from bus names
        country0 = bus0.split(' ')[0]
        country1 = bus1.split(' ')[0]
        
        # Get coordinates
        x0, y0 = network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"]
        x1, y1 = network.buses.loc[bus1, "x"], network.buses.loc[bus1, "y"]
        
        # Get hourly flows
        hourly_avg = hourly_flows[link_name]
        
        # Find dominant direction (more hours in which direction)
        dominant_direction = "positive" if (hourly_avg > 0).sum() > (hourly_avg < 0).sum() else "negative"
        
        # Draw the main link line
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        
        # Create clock-like visualization of hourly flows
        clock_size = 1.5  # Size of the clock visualization
        
        # Create inset axes for clock chart
        fig_coord = ax4.transData.transform((mid_x, mid_y))
        fig_coord = fig.transFigure.inverted().transform(fig_coord)
        
        clock_ax = fig.add_axes([
            fig_coord[0] - clock_size/2, 
            fig_coord[1] - clock_size/2, 
            clock_size, 
            clock_size
        ], polar=True)
        
        # Setup clock face
        clock_ax.set_theta_zero_location('N')
        clock_ax.set_theta_direction(-1)  # Clockwise
        clock_ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
        clock_ax.set_xticklabels([])
        clock_ax.set_yticks([])
        
        # Plot hourly flows as bars on the clock
        theta = np.linspace(0, 2*np.pi, 24, endpoint=False)  # 24 hours
        radii = np.abs(hourly_avg.values)
        width = 2*np.pi / 24
        
        # Normalize to max flow for better visualization
        max_flow = radii.max() if radii.max() > 0 else 1
        radii = radii / max_flow
        
        # Color based on flow direction
        colors_hourly = ['green' if flow > 0 else 'red' for flow in hourly_avg.values]
        
        # Plot bars
        bars = clock_ax.bar(theta, radii, width=width, bottom=0.2, alpha=0.7, color=colors_hourly)
        
        # Add hour labels for key times
        key_hours = [0, 6, 12, 18]  # midnight, 6am, noon, 6pm
        for hour in key_hours:
            angle = 2 * np.pi * hour / 24
            x = 1.1 * np.sin(angle)
            y = 1.1 * np.cos(angle)
            clock_ax.text(angle, 1.2, f"{hour:02d}h", ha='center', va='center', fontsize=8)
        
        # Add center text with legend
        clock_ax.text(0, 0, f"{country0}↔{country1}", ha='center', va='center', fontsize=10)
        
        # Add annotations for main flow periods
        morning_flow = hourly_avg.loc[6:12].mean()
        evening_flow = hourly_avg.loc[18:23].mean()
        
        # Add direction arrows on main map
        # Morning arrow (6-12)
        if abs(morning_flow) > 0:
            morning_color = "green" if morning_flow > 0 else "red"
            start_x, start_y = (x0, y0) if morning_flow > 0 else (x1, y1)
            end_x, end_y = (x1, y1) if morning_flow > 0 else (x0, y0)
            
            # Offset for visibility
            dx, dy = (end_x - start_x) * 0.05, (end_y - start_y) * 0.05
            
            ax4.annotate(
                "AM",
                xy=(start_x + (end_x - start_x) * 0.2, start_y + (end_y - start_y) * 0.2), 
                xycoords='data',
                xytext=(start_x + (end_x - start_x) * 0.3, start_y + (end_y - start_y) * 0.3 + 1),
                textcoords='data',
                arrowprops=dict(
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.1",
                    lw=2,
                    color=morning_color,
                    alpha=0.8
                ),
                fontsize=9,
                color=morning_color,
                fontweight='bold'
            )
        
        # Evening arrow (18-23)
        if abs(evening_flow) > 0:
            evening_color = "green" if evening_flow > 0 else "red"
            start_x, start_y = (x0, y0) if evening_flow > 0 else (x1, y1)
            end_x, end_y = (x1, y1) if evening_flow > 0 else (x0, y0)
            
            # Offset for visibility
            dx, dy = (end_x - start_x) * 0.05, (end_y - start_y) * 0.05
            
            ax4.annotate(
                "PM",
                xy=(start_x + (end_x - start_x) * 0.2, start_y + (end_y - start_y) * 0.2), 
                xycoords='data',
                xytext=(start_x + (end_x - start_x) * 0.3, start_y + (end_y - start_y) * 0.3 - 1),
                textcoords='data',
                arrowprops=dict(
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=-0.1",
                    lw=2,
                    color=evening_color,
                    alpha=0.8
                ),
                fontsize=9,
                color=evening_color,
                fontweight='bold'
            )
    
    ax4.set_title("Daily Flow Patterns (Green: Forward, Red: Reverse)", fontsize=14)
    
    # PLOT 5: Time series of flows for a sample period with enhanced visualization
    # Sample a week of flows for visualization
    sample_week = 24 * 7  # First week
    link_flows_week = network.links_t.p0[["ESP-FRA", "ESP-PRT"]].iloc[:sample_week]
    
    # Plot flows with enhanced styling
    ax5.plot(link_flows_week.index, link_flows_week["ESP-FRA"], linewidth=2, label="ESP → FRA", color="blue")
    ax5.plot(link_flows_week.index, link_flows_week["ESP-PRT"], linewidth=2, label="ESP → PRT", color="purple")
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Fill areas to highlight direction
    ax5.fill_between(link_flows_week.index, link_flows_week["ESP-FRA"], 0, 
                    where=(link_flows_week["ESP-FRA"] > 0), color="blue", alpha=0.3, label="ESP exporting to FRA")
    ax5.fill_between(link_flows_week.index, link_flows_week["ESP-FRA"], 0, 
                    where=(link_flows_week["ESP-FRA"] < 0), color="red", alpha=0.3, label="FRA exporting to ESP")
    
    ax5.fill_between(link_flows_week.index, link_flows_week["ESP-PRT"], 0, 
                    where=(link_flows_week["ESP-PRT"] > 0), color="purple", alpha=0.3, label="ESP exporting to PRT")
    ax5.fill_between(link_flows_week.index, link_flows_week["ESP-PRT"], 0, 
                    where=(link_flows_week["ESP-PRT"] < 0), color="orange", alpha=0.3, label="PRT exporting to ESP")
    
    # Add arrows to indicate flow direction
    span = (link_flows_week.index.max() - link_flows_week.index.min()).total_seconds() / 86400  # span in days
    arrow_indices = np.linspace(0, len(link_flows_week) - 1, 14).astype(int)  # 14 arrows across the week
    
    for i in arrow_indices:
       # ESP-PRT flows
        if link_flows_week["ESP-PRT"].iloc[i] > 0:
            ax5.annotate("", xy=(link_flows_week.index[i], link_flows_week["ESP-PRT"].iloc[i]),
            xytext=(link_flows_week.index[i], link_flows_week["ESP-PRT"].iloc[i] + 500),
            arrowprops=dict(arrowstyle="->", color="purple", alpha=0.7)
        )
        elif link_flows_week["ESP-PRT"].iloc[i] < 0:
            ax5.annotate("", xy=(link_flows_week.index[i], link_flows_week["ESP-PRT"].iloc[i]),
            xytext=(link_flows_week.index[i], link_flows_week["ESP-PRT"].iloc[i] - 500),
            arrowprops=dict(arrowstyle="->", color="orange", alpha=0.7)
        )
    
    # Enhance grid and formatting
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.set_xlabel('Time', fontsize=12)
    ax5.set_ylabel('Power Flow (MW)', fontsize=12)
    ax5.set_title('Transmission Flows Over Time (Sample Week)', fontsize=14)
    
    # Annotate key patterns
    for link_name in ["ESP-FRA", "ESP-PRT"]:
        # Find maximum and minimum flow points for each link
        max_flow_idx = link_flows_week[link_name].idxmax()
        min_flow_idx = link_flows_week[link_name].idxmin()
        
        # Annotate max flow point (export)
        max_val = link_flows_week[link_name].max()
        ax5.annotate(
            f"Max {link_name}: {max_val:.0f} MW",
            xy=(max_flow_idx, max_val),
            xytext=(max_flow_idx, max_val + 1000),
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7)
        )
        
        # Annotate min flow point (import) if negative
        min_val = link_flows_week[link_name].min()
        if min_val < 0:
            ax5.annotate(
                f"Min {link_name}: {min_val:.0f} MW",
                xy=(min_flow_idx, min_val),
                xytext=(min_flow_idx, min_val - 1000),
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7)
            )
    
    # Add legend with enhanced styling
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='ESP → FRA'),
        Line2D([0], [0], color='purple', lw=2, label='ESP → PRT'),
        Patch(facecolor="blue", alpha=0.3, label="ESP exporting to FRA"),
        Patch(facecolor="red", alpha=0.3, label="FRA exporting to ESP"),
        Patch(facecolor="purple", alpha=0.3, label="ESP exporting to PRT"),
        Patch(facecolor="orange", alpha=0.3, label="PRT exporting to ESP"),
    ]
    ax5.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)
    
    # Add explanation text
    fig.text(0.05, 0.03, 
             "Directional arrows show energy flows between countries.\n"
             "Green arrows indicate forward flows, red arrows indicate reverse flows.\n"
             "Pie charts show energy mix by generation type.", 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", alpha=0.8))
    
    # Set global title
    fig.suptitle("Enhanced Energy Flow Visualization with Directional Indicators", fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    return fig

def analyze_hourly_flow_patterns(network):
    """Analyze hourly flow patterns and produce enhanced visualizations."""
    # Create a figure for hourly patterns
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
     # Add timestamp information to the flow data
    flow_data = network.links_t.p0.copy()
    flow_data.index = pd.to_datetime(flow_data.index)

    # Analyze hourly patterns for each link
    for i, link_name in enumerate(["ESP-FRA", "ESP-PRT"]):
        link_flows = flow_data[link_name]
        
        # Calculate hourly averages and standard deviations
        hourly_avg = link_flows.groupby(link_flows.index.hour).mean()
        hourly_std = link_flows.groupby(link_flows.index.hour).std()
        
        # Plot hourly averages with error bands
        x = np.arange(24)
        axes[i].plot(x, hourly_avg.values, 'o-', linewidth=2, 
                    color='blue' if link_name == "ESP-FRA" else "purple",
                    label=f'Average Flow {link_name}')

    # Analyze flows by hour of day for each link
    for i, link_name in enumerate(["ESP-FRA", "ESP-PRT"]):
        # Get hourly averages
        link_flows = network.links_t.p0[link_name]
        hourly_avg = link_flows.groupby(link_flows.index.hour).mean()
        hourly_std = link_flows.groupby(link_flows.index.hour).std()
        
        # Plot hourly averages with error bands
        x = np.arange(24)
        axes[i].plot(x, hourly_avg.values, 'o-', linewidth=2, 
                    color='blue' if link_name == "ESP-FRA" else "purple",
                    label=f'Average Flow {link_name}')
        
        # Add error bands
        axes[i].fill_between(
            x, 
            hourly_avg.values - hourly_std.values,
            hourly_avg.values + hourly_std.values,
            alpha=0.3,
            color='blue' if link_name == "ESP-FRA" else "purple",
            label='±1 Standard Deviation'
        )
        
        # Add zero line
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Shade regions based on flow direction
        axes[i].fill_between(
            x, hourly_avg.values, 0,
            where=(hourly_avg.values > 0),
            color='green', alpha=0.2,
            label=f"{link_name.split('-')[0]} exporting to {link_name.split('-')[1]}"
        )
        axes[i].fill_between(
            x, hourly_avg.values, 0,
            where=(hourly_avg.values < 0),
            color='red', alpha=0.2,
            label=f"{link_name.split('-')[1]} exporting to {link_name.split('-')[0]}"
        )
        
        # Enhance grid and formatting
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].set_xticks(np.arange(0, 24, 2))
        axes[i].set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
        axes[i].set_xlabel('Hour of Day', fontsize=12)
        axes[i].set_ylabel('Average Power Flow (MW)', fontsize=12)
        axes[i].set_title(f'Average Hourly Flow Pattern for {link_name}', fontsize=14)
        
        # Add annotations for peak hours
        max_hour = hourly_avg.idxmax()
        min_hour = hourly_avg.idxmin()
        
        # Annotate peak export hour
        axes[i].annotate(
            f"Peak Export: {hourly_avg.max():.0f} MW at {max_hour:02d}:00",
            xy=(max_hour, hourly_avg.max()),
            xytext=(max_hour, hourly_avg.max() + hourly_std.max()),
            arrowprops=dict(arrowstyle="->", color="green"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )
        
        # Annotate peak import hour if flow goes negative
        if hourly_avg.min() < 0:
            axes[i].annotate(
                f"Peak Import: {abs(hourly_avg.min()):.0f} MW at {min_hour:02d}:00",
                xy=(min_hour, hourly_avg.min()),
                xytext=(min_hour, hourly_avg.min() - hourly_std.max()),
                arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        
        # Add legend
        axes[i].legend(loc='best', framealpha=0.9)
        
        # Add directional arrows to highlight flow patterns
        # Morning pattern (6-12)
        morning_flow = hourly_avg.loc[6:12].mean()
        if morning_flow > 0:
            axes[i].annotate(
                "Morning Flow",
                xy=(9, morning_flow),
                xytext=(9, morning_flow + hourly_std.loc[6:12].max() * 1.5),
                arrowprops=dict(arrowstyle="->", color="green"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        elif morning_flow < 0:
            axes[i].annotate(
                "Morning Flow",
                xy=(9, morning_flow),
                xytext=(9, morning_flow - hourly_std.loc[6:12].max() * 1.5),
                arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        
        # Evening pattern (18-23)
        evening_flow = hourly_avg.loc[18:23].mean()
        if evening_flow > 0:
            axes[i].annotate(
                "Evening Flow",
                xy=(20, evening_flow),
                xytext=(20, evening_flow + hourly_std.loc[18:23].max() * 1.5),
                arrowprops=dict(arrowstyle="->", color="green"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        elif evening_flow < 0:
            axes[i].annotate(
                "Evening Flow",
                xy=(20, evening_flow),
                xytext=(20, evening_flow - hourly_std.loc[18:23].max() * 1.5),
                arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
    
    # Add overall title
    fig.suptitle("Enhanced Hourly Flow Pattern Analysis", fontsize=18, y=0.98)
    
    # Add explanation text
    fig.text(0.5, 0.01, 
             "Green shading indicates export flows, red shading indicates import flows.\n"
             "Arrows highlight peak flow periods and patterns.", 
             fontsize=12, ha='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def seasonal_flow_patterns(network):
    """Analyze seasonal flow patterns and visualize them with enhanced graphics."""
    # Create a figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Add timestamp information to the flow data
    flow_data = network.links_t.p0.copy()
    flow_data.index = pd.to_datetime(flow_data.index)
    
    # Add month and season information
    flow_data['month'] = flow_data.index.month
    flow_data['season'] = flow_data.index.month.map({
        1: 'Winter', 2: 'Winter', 3: 'Spring', 
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Autumn',
        10: 'Autumn', 11: 'Autumn', 12: 'Winter'
    })
    
    # Define colors for seasons
    season_colors = {
        'Winter': 'blue',
        'Spring': 'green',
        'Summer': 'red',
        'Autumn': 'orange'
    }
    
    # Analyze seasonal patterns for each link
    for i, link_name in enumerate(["ESP-FRA", "ESP-PRT"]):
        # Calculate monthly averages
        monthly_avg = flow_data.groupby('month')[link_name].mean()
        monthly_std = flow_data.groupby('month')[link_name].std()
        
        # Get season for each month for coloring
        month_seasons = {month: season for month, season in zip(
            flow_data['month'].unique(), 
            flow_data.groupby('month')['season'].first().values
        )}
        
        # Plot monthly averages with seasonal coloring
        x = np.arange(1, 13)
        colors = [season_colors[month_seasons[month]] for month in x]
        
        # Create bars with seasonal coloring
        bars = axes[i].bar(x, monthly_avg.values, alpha=0.7, color=colors)
        
        # Add error bars
        axes[i].errorbar(x, monthly_avg.values, yerr=monthly_std.values, fmt='none', 
                        ecolor='black', capsize=5, alpha=0.5)
        
        # Add zero line
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Enhance grid and formatting
        axes[i].grid(True, linestyle='--', alpha=0.7, axis='y')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[i].set_xlabel('Month', fontsize=12)
        axes[i].set_ylabel('Average Power Flow (MW)', fontsize=12)
        axes[i].set_title(f'Seasonal Flow Pattern for {link_name}', fontsize=14)
        
        # Add seasonal annotations
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            # Get months for this season
            season_months = [m for m, s in month_seasons.items() if s == season]
            if not season_months:
                continue
                
            # Calculate average flow for this season
            season_avg = monthly_avg.loc[season_months].mean()
            
            # Find middle month of season for annotation placement
            mid_month = season_months[len(season_months)//2]
            
            # Add seasonal annotation with arrow
            y_offset = 500 if season_avg >= 0 else -500
            axes[i].annotate(
                f"{season}: {season_avg:.0f} MW",
                xy=(mid_month, season_avg),
                xytext=(mid_month, season_avg + y_offset),
                arrowprops=dict(arrowstyle="->", color=season_colors[season]),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                ha='center'
            )
        
        # Add directional indicators
        for month in range(1, 13):
            flow = monthly_avg.loc[month]
            if flow > 0:
                axes[i].text(
                    month, flow / 2,
                    "↑",
                    ha='center',
                    va='center',
                    fontsize=14,
                    color='white',
                    weight='bold'
                )
            elif flow < 0:
                axes[i].text(
                    month, flow / 2,
                    "↓",
                    ha='center',
                    va='center',
                    fontsize=14,
                    color='white',
                    weight='bold'
                )
    
    # Add legend for seasons
    legend_elements = [
        Patch(facecolor=color, label=season)
        for season, color in season_colors.items()
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    
    # Add direction legend
    direction_legend = [
        ('↑', 'Export Flow'),
        ('↓', 'Import Flow')
    ]
    direction_elements = [
        Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', 
              markersize=10, label=label)
        for marker, label in direction_legend
    ]
    fig.legend(handles=direction_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add explanation text
    fig.text(0.5, 0.01, 
             "Color coding indicates seasons: blue (Winter), green (Spring), red (Summer), orange (Autumn).\n"
             "Arrows show the direction of energy flows: up (export), down (import).", 
             fontsize=12, ha='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig

def create_all_enhanced_visualizations(network):
    """Create and save all enhanced visualizations."""
    # Create the main visualization
    fig1 = create_energy_flow_visualizations(network)  # Corrected function name
    fig1.savefig('enhanced_energy_flow_visualization.png', dpi=300, bbox_inches='tight')
    
    # Create hourly analysis 
    fig2 = analyze_hourly_flow_patterns(network)
    fig2.savefig('enhanced_hourly_flow_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create seasonal analysis
    fig3 = seasonal_flow_patterns(network)
    fig3.savefig('enhanced_seasonal_flow_analysis.png', dpi=300, bbox_inches='tight')
    
    return "All enhanced visualizations created successfully!"

# Example usage:
# network = pypsa.Network('path_to_your_network.nc')
# create_energy_flow_visualizations(network)

# Call the visualization functions
fig1 = create_all_enhanced_visualizations(network)
fig2 = analyze_hourly_flow_patterns(network)
fig3 = seasonal_flow_patterns(network)

# Display the plots
plt.figure(fig1.number)
plt.show()
plt.figure(fig2.number)
plt.show()
plt.figure(fig3.number)
plt.show()
