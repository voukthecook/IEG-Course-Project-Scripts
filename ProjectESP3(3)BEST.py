#BEST ONE
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Function to create and solve the model for a specific weather year
def run_model_for_year(weather_year, demand_year=2015):
    """
    Run the optimization model using weather data from a specific year
    but keeping the same demand profile
    
    Parameters:
    -----------
    weather_year : int
        Year to use for wind and solar profiles
    demand_year : int
        Year to use for electricity demand profile (default: 2015)
    
    Returns:
    --------
    dict
        Dictionary with optimization results
    """
    print(f"Running model for weather year {weather_year}...")
    
    # Create network
    network = pypsa.Network()
    
    # Set time period (always use 2015 as the reference year for timestamps)
    hours_in_year = pd.date_range(f'{demand_year}-01-01 00:00Z',
                                 f'{demand_year}-12-31 23:00Z',
                                 freq='h')
    network.set_snapshots(hours_in_year.values)
    
    # Add bus
    network.add("Bus", "electricity bus")
    
    # Load electricity demand data (always use 2015 demand)
    df_elec = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/electricity_demand.csv', sep=';', index_col=0) # in MWh
    df_elec.index = pd.to_datetime(df_elec.index)
    country = 'ESP'
    
    # Add load to the bus
    network.add("Load",
                "load",
                bus="electricity bus",
                p_set=df_elec[country].values)
    
    # Annuity function
    def annuity(n, r):
        """ Calculate the annuity factor for an asset with lifetime n years and
        discount rate r """
        if r > 0:
            return r/(1. - 1./(1.+r)**n)
        else:
            return 1/n
    
    # Add carriers
    network.add("Carrier", "gas", co2_emissions=0.19)  # in t_CO2/MWh_th
    network.add("Carrier", "onshorewind")
    network.add("Carrier", "solar")
    
    # Load wind data for the specific weather year
    df_onshorewind = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/onshore_wind_1979-2017.csv', sep=';', index_col=0)
    df_onshorewind.index = pd.to_datetime(df_onshorewind.index)
    
    # Filter for the specific weather year
    wind_data_year = df_onshorewind[
        (df_onshorewind.index >= f'{weather_year}-01-01') & 
        (df_onshorewind.index < f'{weather_year+1}-01-01')
    ]
    
    # Map the weather year data to our reference year timestamps
    CF_wind = wind_data_year[country].values[:len(network.snapshots)]
    
    # Add onshore wind generator
    capital_cost_onshorewind = annuity(30, 0.07) * 910000 * (1 + 0.033)  # in €/MW
    network.add("Generator",
                "onshorewind",
                bus="electricity bus",
                p_nom_extendable=True,
                carrier="onshorewind",
                capital_cost=capital_cost_onshorewind,
                marginal_cost=0,
                p_max_pu=CF_wind)
    
    # Load solar data for the specific weather year
    df_solar = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/pv_optimal.csv', sep=';', index_col=0)
    df_solar.index = pd.to_datetime(df_solar.index)
    
    # Filter for the specific weather year
    solar_data_year = df_solar[
        (df_solar.index >= f'{weather_year}-01-01') & 
        (df_solar.index < f'{weather_year+1}-01-01')
    ]
    
    # Map the weather year data to our reference year timestamps
    CF_solar = solar_data_year[country].values[:len(network.snapshots)]
    
    # Add solar PV generator
    capital_cost_solar = annuity(25, 0.07) * 425000 * (1 + 0.03)  # in €/MW
    network.add("Generator",
                "solar",
                bus="electricity bus",
                p_nom_extendable=True,
                carrier="solar",
                capital_cost=capital_cost_solar,
                marginal_cost=0,
                p_max_pu=CF_solar)
    
    # Add OCGT generator
    capital_cost_OCGT = annuity(25, 0.07) * 560000 * (1 + 0.033)  # in €/MW
    fuel_cost = 25  # in €/MWh_th
    efficiency = 0.39  # MWh_elec/MWh_th
    marginal_cost_OCGT = fuel_cost / efficiency  # in €/MWh_el
    network.add("Generator",
                "OCGT",
                bus="electricity bus",
                p_nom_extendable=True,
                carrier="gas",
                capital_cost=capital_cost_OCGT,
                marginal_cost=marginal_cost_OCGT)
    
    # Add CO2 constraint
    co2_limit = 1000000  # tonCO2
    network.add("GlobalConstraint",
                "co2_limit",
                type="primary_energy",
                carrier_attribute="co2_emissions",
                sense="<=",
                constant=co2_limit)
    
    # Optimize
    network.optimize(solver_name='gurobi')
    
    # Collect results
    results = {
        'year': weather_year,
        'objective': network.objective,
        'generators': {
            name: {
                'p_nom_opt': value,
                'p_sum': network.generators_t.p[name].sum(),
                'capacity_factor': network.generators_t.p[name].sum() / (value * 8760) if value > 0 else 0
            } for name, value in network.generators.p_nom_opt.items()
        },
        'co2_emissions': (network.generators_t.p['OCGT'].sum() * network.carriers.loc['gas', 'co2_emissions'] / 0.39)
    }
    
    return results

# Define range of years to analyze
weather_years = range(1979, 2017)  # Adjust range based on your available data

# Run model for each year and collect results
results = []
for year in weather_years:
    try:
        result = run_model_for_year(year)
        results.append(result)
    except Exception as e:
        print(f"Error running model for year {year}: {e}")

# Convert results to DataFrames for easier analysis
df_capacity = pd.DataFrame({
    'Year': [r['year'] for r in results],
    'Onshore Wind': [r['generators']['onshorewind']['p_nom_opt'] for r in results],
    'Solar': [r['generators']['solar']['p_nom_opt'] for r in results],
    'OCGT': [r['generators']['OCGT']['p_nom_opt'] for r in results]
})

df_generation = pd.DataFrame({
    'Year': [r['year'] for r in results],
    'Onshore Wind': [r['generators']['onshorewind']['p_sum'] for r in results],
    'Solar': [r['generators']['solar']['p_sum'] for r in results],
    'OCGT': [r['generators']['OCGT']['p_sum'] for r in results]
})

df_capacity_factors = pd.DataFrame({
    'Year': [r['year'] for r in results],
    'Onshore Wind': [r['generators']['onshorewind']['capacity_factor'] for r in results],
    'Solar': [r['generators']['solar']['capacity_factor'] for r in results],
    'OCGT': [r['generators']['OCGT']['capacity_factor'] for r in results]
})

# Plotting functions
def plot_capacity_variability():
    """Plot optimal capacity for each generator across different weather years"""
    # Set up colors for each technology
    colors = {
        'Onshore Wind': 'blue',
        'Solar': 'orange',
        'OCGT': 'brown'
    }
    
    # Calculate statistics
    mean_capacities = df_capacity.drop('Year', axis=1).mean()
    std_capacities = df_capacity.drop('Year', axis=1).std()
    
    # Create bar plot with error bars
    tech_names = ['Onshore Wind', 'Solar', 'OCGT']
    x = np.arange(len(tech_names))
    width = 0.7
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(x, [mean_capacities[tech] for tech in tech_names], 
                  width, 
                  yerr=[std_capacities[tech] for tech in tech_names],
                  color=[colors[tech] for tech in tech_names],
                  capsize=10)
    
    # Add labels and formatting
    ax.set_ylabel('Optimal Capacity (MW)', fontsize=14)
    ax.set_title('Average Optimal Capacity and Variability across Weather Years (1979-2017)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(tech_names, rotation=45, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1*height,
                f'{int(height)}±{int(std_capacities[tech_names[i]])}',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('capacity_variability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed line plot showing year-to-year variations
    plt.figure(figsize=(14, 8))
    
    for tech in tech_names:
        plt.plot(df_capacity['Year'], df_capacity[tech], 'o-', label=tech, color=colors[tech])
    
    plt.xlabel('Weather Year', fontsize=14)
    plt.ylabel('Optimal Capacity (MW)', fontsize=14)
    plt.title('Optimal Capacity by Generation Technology for Different Weather Years', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('capacity_by_year.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_capacity_factor_variability():
    """Plot capacity factors for variable renewables across different weather years"""
    plt.figure(figsize=(14, 8))
    
    vres_techs = ['Onshore Wind', 'Solar']
    
    for tech in vres_techs:
        plt.plot(df_capacity_factors['Year'], df_capacity_factors[tech], 'o-', label=tech)
    
    plt.xlabel('Weather Year', fontsize=14)
    plt.ylabel('Capacity Factor', fontsize=14)
    plt.title('Variable Renewable Capacity Factors across Different Weather Years', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('capacity_factors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a box plot showing the distribution of capacity factors
    plt.figure(figsize=(10, 6))
    
    data = [df_capacity_factors[tech] for tech in vres_techs]
    
    bp = plt.boxplot(data, labels=vres_techs, patch_artist=True)
    
    # Set colors for each box
    colors = ['blue', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Capacity Factor', fontsize=14)
    plt.title('Distribution of Capacity Factors across Weather Years (1979-2017)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add statistics as text
    for i, tech in enumerate(vres_techs):
        mean_cf = df_capacity_factors[tech].mean()
        std_cf = df_capacity_factors[tech].std()
        min_cf = df_capacity_factors[tech].min()
        max_cf = df_capacity_factors[tech].max()
        
        plt.text(i+1.2, mean_cf, 
                 f'Mean: {mean_cf:.3f}\nStd: {std_cf:.3f}\nMin: {min_cf:.3f}\nMax: {max_cf:.3f}',
                 fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('capacity_factor_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_generation_mix_variability():
    """Plot generation mix across different weather years"""
    # Calculate percentage of total generation for each technology
    total_gen = df_generation.drop('Year', axis=1).sum(axis=1)
    gen_percentage = df_generation.copy()
    
    for tech in ['Onshore Wind', 'Solar', 'OCGT']:
        gen_percentage[tech] = gen_percentage[tech] / total_gen * 100
    
    # Create stacked bar chart
    plt.figure(figsize=(14, 8))
    
    bottom = np.zeros(len(gen_percentage))
    colors = {'Onshore Wind': 'blue', 'Solar': 'orange', 'OCGT': 'brown'}
    
    for tech in ['Onshore Wind', 'Solar', 'OCGT']:
        plt.bar(gen_percentage['Year'], gen_percentage[tech], bottom=bottom, 
                label=tech, color=colors[tech])
        bottom += gen_percentage[tech]
    
    plt.xlabel('Weather Year', fontsize=14)
    plt.ylabel('Share of Total Generation (%)', fontsize=14)
    plt.title('Electricity Generation Mix across Different Weather Years', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('generation_mix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create scatter plot showing relationship between wind/solar CF and OCGT capacity
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df_capacity_factors['Onshore Wind'], df_capacity['OCGT'], 
                label='Wind CF vs OCGT Capacity', alpha=0.7)
    
    plt.xlabel('Onshore Wind Capacity Factor', fontsize=14)
    plt.ylabel('OCGT Capacity (MW)', fontsize=14)
    plt.title('Relationship Between Wind Capacity Factor and OCGT Capacity', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('wind_cf_vs_ocgt.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a summary pie chart showing the average mix
    avg_gen = df_generation.drop('Year', axis=1).mean()
    avg_gen_pct = avg_gen / avg_gen.sum() * 100
    
    plt.figure(figsize=(10, 8))
    plt.pie(avg_gen_pct, labels=avg_gen_pct.index, autopct='%1.1f%%', 
            colors=[colors[tech] for tech in avg_gen_pct.index], 
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
    plt.title('Average Electricity Generation Mix (1979-2017)', fontsize=16)
    plt.tight_layout()
    plt.savefig('average_generation_mix.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run all analysis functions
plot_capacity_variability()
plot_capacity_factor_variability()
plot_generation_mix_variability()

# Create a summary table
summary_capacity = pd.DataFrame({
    'Technology': ['Onshore Wind', 'Solar', 'OCGT'],
    'Mean Capacity (MW)': [df_capacity[tech].mean() for tech in ['Onshore Wind', 'Solar', 'OCGT']],
    'Std Dev (MW)': [df_capacity[tech].std() for tech in ['Onshore Wind', 'Solar', 'OCGT']],
    'Min Capacity (MW)': [df_capacity[tech].min() for tech in ['Onshore Wind', 'Solar', 'OCGT']],
    'Max Capacity (MW)': [df_capacity[tech].max() for tech in ['Onshore Wind', 'Solar', 'OCGT']],
    'Coefficient of Variation (%)': [df_capacity[tech].std()/df_capacity[tech].mean()*100 if df_capacity[tech].mean() > 0 else 0 
                                   for tech in ['Onshore Wind', 'Solar', 'OCGT']]
})

print("Summary of Capacity Variability Across Weather Years:")
print(summary_capacity.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

summary_cf = pd.DataFrame({
    'Technology': ['Onshore Wind', 'Solar'],
    'Mean CF': [df_capacity_factors[tech].mean() for tech in ['Onshore Wind', 'Solar']],
    'Std Dev': [df_capacity_factors[tech].std() for tech in ['Onshore Wind', 'Solar']],
    'Min CF': [df_capacity_factors[tech].min() for tech in ['Onshore Wind', 'Solar']],
    'Max CF': [df_capacity_factors[tech].max() for tech in ['Onshore Wind', 'Solar']],
    'Coefficient of Variation (%)': [df_capacity_factors[tech].std()/df_capacity_factors[tech].mean()*100 
                                   for tech in ['Onshore Wind', 'Solar']]
})

print("\nSummary of Capacity Factor Variability for Renewables:")
print(summary_cf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Calculate correlation between wind/solar CF and OCGT capacity
corr_wind_ocgt = np.corrcoef(df_capacity_factors['Onshore Wind'], df_capacity['OCGT'])[0,1]
corr_solar_ocgt = np.corrcoef(df_capacity_factors['Solar'], df_capacity['OCGT'])[0,1]

print("\nCorrelation between Wind CF and OCGT Capacity:", f"{corr_wind_ocgt:.3f}")
print("Correlation between Solar CF and OCGT Capacity:", f"{corr_solar_ocgt:.3f}")