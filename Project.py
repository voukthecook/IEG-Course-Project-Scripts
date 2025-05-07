import pandas as pd
import pypsa

# We start by creating the network. In this example, the country is modelled as a single node, so the network includes only one bus.

# We select the year 2015 and set the hours in that year as snapshots.

# We select a country, in this case Denmark (DNK), and add one node (electricity bus) to the network.

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

# The electricity demand time series were obtained from ENTSOE through the very convenient compilation carried out by the Open Power System Data (OPSD). https://data.open-power-system-data.org/time_series/

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

#Print the load time series to check that it has been properly added (you should see numbers and not ‘NaN’)
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

# We include solar PV and onshore wind generators.

# The capacity factors representing the availability of those generators for every European country can be downloaded from the following repositories (select ‘optimal’ for PV and onshore for wind).

# https://zenodo.org/record/3253876#.XSiVOEdS8l0

# https://zenodo.org/record/2613651#.XSiVOkdS8l0

# We include also Open Cycle Gas Turbine (OCGT) generators

# The cost assumed for the generators are the same as in Table 1 in the paper https://doi.org/10.1016/j.enconman.2019.111977 (open version: https://arxiv.org/pdf/1906.06936.pdf)

# add the different carriers, only gas emits CO2
network.add("Carrier", "gas", co2_emissions=0.19) # in t_CO2/MWh_th
network.add("Carrier", "onshorewind")
network.add("Carrier", "solar")

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
fuel_cost = 21.6 # in €/MWh_th
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

#Print the generator Capacity factor time series to check that it has been properly added (you should see numbers and not ‘NaN’)
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



# We can plot now the dispatch of every generator during the first week of the year and the electricity demand. We import the matplotlib package which is very useful to plot results.

# We can also plot the electricity mix.

import matplotlib.pyplot as plt

plt.plot(network.loads_t.p['load'][0:96], color='black', label='demand')
plt.plot(network.generators_t.p['onshorewind'][0:96], color='blue', label='onshore wind')
plt.plot(network.generators_t.p['solar'][0:96], color='orange', label='solar')
plt.plot(network.generators_t.p['OCGT'][0:96], color='brown', label='gas (OCGT)')
plt.legend(fancybox=True, shadow=True, loc='best')
plt.title('Electricity demand and generation in the first week of 2015')
plt.xlabel('Hour')
plt.ylabel('Power (MW)')
plt.grid()
plt.show()

#pie chart of technoligies
labels = ['onshore wind',
          'solar',
          'gas (OCGT)']
sizes = [network.generators_t.p['onshorewind'].sum(),
         network.generators_t.p['solar'].sum(),
         network.generators_t.p['OCGT'].sum()]

colors=['blue', 'orange', 'brown']

plt.pie(sizes,
        colors=colors,
        labels=labels,
        wedgeprops={'linewidth':0})
plt.axis('equal')

plt.title('Electricity mix', y=1.07)
plt.show()

#We can add a global CO2 constraint and solve again.

co2_limit=1000000 #tonCO2
network.add("GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit)
network.optimize(solver_name='gurobi')


#The optimal capacity for every generator can be shown.

network.generators.p_nom_opt # in MW
print('The optimal capacity for every generator is:')
print(network.generators.p_nom_opt)

plt.plot(network.loads_t.p['load'][0:96], color='black', label='demand')
plt.plot(network.generators_t.p['onshorewind'][0:96], color='blue', label='onshore wind')
plt.plot(network.generators_t.p['solar'][0:96], color='orange', label='solar')
plt.plot(network.generators_t.p['OCGT'][0:96], color='brown', label='gas (OCGT)')
plt.legend(fancybox=True, shadow=True, loc='best')
plt.title('Electricity demand and generation in the first week of 2015')
plt.xlabel('Hour')
plt.ylabel('Power (MW)')
plt.grid()
plt.show()

labels = ['onshore wind', 'solar', 'gas (OCGT)' ]
sizes = [network.generators_t.p['onshorewind'].sum(),
         network.generators_t.p['solar'].sum(),
         network.generators_t.p['OCGT'].sum()]

colors = ['blue', 'orange', 'brown']

plt.pie(sizes,
        colors=colors,
        labels=labels,
        wedgeprops={'linewidth':0})
plt.axis('equal')

plt.title('Electricity mix', y=1.07)
plt.show()

