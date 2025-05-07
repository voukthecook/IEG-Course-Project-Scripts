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
    network.optimize(solver_name='gurobi')
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
