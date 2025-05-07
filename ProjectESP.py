import pandas as pd
import pypsa
import matplotlib.pyplot as plt

# We start by creating the network. In this example, the country is modelled as a single node, so the network includes only one bus.
# We select the year 2015 and set the hours in that year as snapshots.
# We select a country, in this case Spain (ESP), and add one node (electricity bus) to the network.

network = pypsa.Network()
hours_in_2015 = pd.date_range('2015-01-01 00:00Z',
                              '2015-12-31 23:00Z',
                              freq='h')

network.set_snapshots(hours_in_2015.values)

network.add("Bus",
            "electricity bus")

network.snapshots
print('The network has been created with the following snapshots:')
print(network.snapshots)

# The demand is represented by the historical electricity demand in 2015 with hourly resolution.
# The file with historical hourly electricity demand for every European country is available in the data folder.
# The electricity demand time series were obtained from ENTSOE through the very convenient compilation carried out by the Open Power System Data (OPSD).

# load electricity demand data
df_elec = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/electricity_demand.csv', sep=';', index_col=0) # in MWh
df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime
country='ESP'
print(df_elec[country].head())

# add load to the bus
network.add("Load",
            "load",
            bus="electricity bus",
            p_set=df_elec[country].values)

#Print the load time series to check that it has been properly added (you should see numbers and not 'NaN')
network.loads_t.p_set
print('The load time series has been added to the network:')
print(network.loads_t.p_set)

# In the optimization, we will minimize the annualized system costs.
# We will need to annualize the cost of every generator, we build a function to do it.

def annuity(n,r):
    """ Calculate the annuity factor for an asset with lifetime n years and
    discount rate  r """

    if r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n

# We include solar PV, onshore wind, biomass, and OCGT generators

# add the different carriers, gas and biomass emit CO2
network.add("Carrier", "gas", co2_emissions=0.19) # in t_CO2/MWh_th
network.add("Carrier", "biomass", co2_emissions=0.0) # Considered carbon neutral in many scenarios
network.add("Carrier", "onshorewind")
network.add("Carrier", "solar")
network.add("Carrier", "hydro")
#CAPACITIES NEED TO BE VERIFIED WITH REAL DATA, EPSECIALLY HYDRO AND BIOMASS
# add onshore wind generator
df_onshorewind = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)
CF_wind = df_onshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
capital_cost_onshorewind = annuity(30,0.07)*910000*(1+0.033) # in €/MW
network.add("Generator",
            "onshorewind",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="onshorewind",
            #p_nom_max=1000, # maximum capacity can be limited due to environmental constraints
            capital_cost = capital_cost_onshorewind,
            marginal_cost = 0,
            p_max_pu = CF_wind.values)

# add solar PV generator
df_solar = pd.read_csv('C:/Users/nasos/DTU/IEG(Env and source data)/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index)
CF_solar = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
capital_cost_solar = annuity(25,0.07)*425000*(1+0.03) # in €/MW
network.add("Generator",
            "solar",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="solar",
            #p_nom_max=1000, # maximum capacity can be limited due to environmental constraints
            capital_cost = capital_cost_solar,
            marginal_cost = 0,
            p_max_pu = CF_solar.values)

# add OCGT (Open Cycle Gas Turbine) generator
capital_cost_OCGT = annuity(25,0.07)*560000*(1+0.033) # in €/MW
fuel_cost = 25 # in €/MWh_th
efficiency = 0.39 # MWh_elec/MWh_th
marginal_cost_OCGT = fuel_cost/efficiency # in €/MWh_el
network.add("Generator",
            "OCGT",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="gas",
            #p_nom_max=1000,
            capital_cost = capital_cost_OCGT,
            marginal_cost = marginal_cost_OCGT)

# add Biomass generator (new)
capital_cost_biomass = annuity(25,0.07)*2500000*(1+0.033) # in €/MW (higher capital cost than OCGT)
biomass_fuel_cost = 30 # in €/MWh_th (higher fuel cost than natural gas)
biomass_efficiency = 0.35 # MWh_elec/MWh_th (slightly lower than OCGT)
marginal_cost_biomass = biomass_fuel_cost/biomass_efficiency # in €/MWh_el
# Add potential resource limit for biomass
biomass_p_nom_max = 5000 # Maximum capacity in MW (can be adjusted based on resource availability)

network.add("Generator",
            "biomass",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="biomass",
            p_nom_max=biomass_p_nom_max,  # Maximum capacity based on resource constraint
            capital_cost=capital_cost_biomass,
            marginal_cost=marginal_cost_biomass)

# Assuming you have a time series for hydro generation profile
capital_cost_hydro = annuity(30, 0.07) * 2000000  # Adjust based on real data for hydro
hydro_fuel_cost = 0  # Hydro is typically free of fuel costs
hydro_efficiency = 0.9  # Efficiency of hydro generation
hydro_p_nom_max = 1000  # Maximum capacity in MW (can be adjusted based on resource availability)
# Add potential resource limit for hydro

network.add("Generator",
            "hydro",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="hydro",
            capital_cost=capital_cost_hydro,
            marginal_cost=0,
            p_nom_max=hydro_p_nom_max,)  # Maximum capacity based on resource constraint)


#Print the generator Capacity factor time series to check that it has been properly added (you should see numbers and not 'NaN')
network.generators_t.p_max_pu
print('The generator Capacity factor time series has been added to the network:')
print(network.generators_t.p_max_pu)

# We find the optimal solution using Gurobi as solver.
# In this case, we are optimizing the installed capacity and dispatch of every generator to minimize the total system cost.

network.optimize(solver_name='gurobi')

#The total cost can be read from the network objetive.
print('The total cost of the system is:')
print(network.objective/1000000, '10^6 €')

#The cost per MWh of electricity produced can also be calculated.
print('The cost per MWh of electricity produced is:')
print(network.objective/network.loads_t.p.sum()/1000000, '10^6 €/MWh')

#The optimal capacity for every generator can be shown.
network.generators.p_nom_opt # in MW
print('The optimal capacity for every generator is:')
print(network.generators.p_nom_opt)

# Function to plot dispatch for a specific week
def plot_dispatch(network, start_hour, end_hour, title):
    plt.figure(figsize=(12, 6))
    plt.plot(network.loads_t.p['load'][start_hour:end_hour], color='black', label='demand')
    plt.plot(network.generators_t.p['onshorewind'][start_hour:end_hour], color='blue', label='onshore wind')
    plt.plot(network.generators_t.p['solar'][start_hour:end_hour], color='orange', label='solar')
    plt.plot(network.generators_t.p['biomass'][start_hour:end_hour], color='green', label='biomass')
    plt.plot(network.generators_t.p['OCGT'][start_hour:end_hour], color='brown', label='gas (OCGT)')
    plt.plot(network.generators_t.p['hydro'][start_hour:end_hour], color='pink', label='hydro')
    plt.legend(fancybox=True, shadow=True, loc='best')
    plt.title(title)
    plt.xlabel('Hour')
    plt.ylabel('Power (MW)')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Plot dispatch for first week of January (hours 0-167)
plot_dispatch(network, 0, 168, 'Electricity demand and generation in the first week of January 2015')

# Plot dispatch for first week of June (hours 3624-3791)
# June 1st starts at hour 3624 (24 hours/day * 151 days)
june_start = 24 * 151
plot_dispatch(network, june_start, june_start + 168, 'Electricity demand and generation in the first week of June 2015')

# Pie chart of technologies
labels = ['onshore wind',
          'solar',
          'biomass',
          'gas (OCGT)',
          'hydro']  # Added missing label for hydro
sizes = [network.generators_t.p['onshorewind'].sum(),
         network.generators_t.p['solar'].sum(),
         network.generators_t.p['biomass'].sum(),
         network.generators_t.p['OCGT'].sum(),
         network.generators_t.p['hydro'].sum()]

colors=['blue', 'orange', 'green', 'brown', 'pink']

plt.figure(figsize=(8, 6))
plt.pie(sizes,
        colors=colors,
        labels=labels,
        autopct='%1.1f%%',
        wedgeprops={'linewidth':0})
plt.axis('equal')
plt.title('Electricity mix', y=1.07)
plt.show()

# Now we add a global CO2 constraint and solve again
co2_limit=1000000 #tonCO2
network.add("GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit)
network.optimize(solver_name='gurobi')

# The optimal capacity for every generator with CO2 constraint
print('The optimal capacity for every generator with CO2 constraint is:')
print(network.generators.p_nom_opt)

# Plot dispatch for first week of January with CO2 constraint
plot_dispatch(network, 0, 168, 'Electricity demand and generation in the first week of January 2015 (with CO2 constraint)')

# Plot dispatch for first week of June with CO2 constraint
plot_dispatch(network, june_start, june_start + 168, 'Electricity demand and generation in the first week of June 2015 (with CO2 constraint)')

# Updated pie chart with CO2 constraint
plt.figure(figsize=(8, 6))
plt.pie([network.generators_t.p['onshorewind'].sum(),
         network.generators_t.p['solar'].sum(),
         network.generators_t.p['biomass'].sum(),
         network.generators_t.p['OCGT'].sum(),
         network.generators_t.p['hydro'].sum()],
        colors=colors,
        labels=labels,
        autopct='%1.1f%%',
        wedgeprops={'linewidth':0})
plt.axis('equal')
plt.title('Electricity mix with CO2 constraint', y=1.07)
plt.show()

# Optional: Print the total CO2 emissions from each generator
print('CO2 emissions from OCGT (tons):', 
      (network.generators_t.p['OCGT'].sum() * network.carriers.loc['gas', 'co2_emissions']/0.39))
print('CO2 emissions from biomass (tons):', 
      (network.generators_t.p['biomass'].sum() * network.carriers.loc['biomass', 'co2_emissions']/0.35))
print('Total CO2 emissions (tons):', 
      (network.generators_t.p['OCGT'].sum() * network.carriers.loc['gas', 'co2_emissions']/0.39) + 
      (network.generators_t.p['biomass'].sum() * network.carriers.loc['biomass', 'co2_emissions']/0.35))

