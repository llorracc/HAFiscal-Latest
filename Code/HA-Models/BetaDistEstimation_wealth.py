'''
This is the main file for the cstwMPC project.  It estimates one version of the model
each time it is executed.  The following parameters *must* be defined in the __main__
namespace in order for this file to run correctly:
    
param_name : str
    Which parameter to introduce heterogeneity in (usually DiscFac).
dist_type : str
    Which type of distribution to use (can be 'uniform' or 'lognormal').
do_param_dist : bool
    Do param-dist version if True, param-point if False.
do_lifecycle : bool
    Use lifecycle model if True, perpetual youth if False.
do_agg_shocks : bool
    Whether to solve the FBS aggregate shocks version of the model or use idiosyncratic shocks only.
do_liquid : bool
    Matches liquid assets data when True, net worth data when False.
do_tractable : bool
    Whether to use an extremely simple alternate specification of households' optimization problem.
run_estimation : bool
    Whether to actually estimate the model specified by the other options.
run_sensitivity : [bool]
    Whether to run each of eight sensitivity analyses; currently inoperative.  Order:
    rho, xi_sigma, psi_sigma, mu, urate, mortality, g, R
find_beta_vs_KY : bool
    Whether to computes K/Y ratio for a wide range of beta; should have do_param_dist = False and param_name = 'DiscFac'.
    Currently inoperative.
path_to_models : str
    Absolute path to the location of this file.
    
All of these parameters are set when running this file from one of the do_XXX.py
files in the root directory.
'''
from __future__ import division, print_function
from __future__ import absolute_import

from builtins import str
from builtins import range

import os

import numpy as np
from copy import copy, deepcopy
from time import clock
from HARK.utilities import approxMeanOneLognormal, combineIndepDstns, approxUniform, \
                           getPercentiles, getLorenzShares, calcSubpopAvg, approxLognormal
from HARK.simulation import drawDiscrete
from HARK import Market
import HARK.ConsumptionSaving.ConsIndShockModel as Model
from HARK.ConsumptionSaving.ConsAggShockModel import CobbDouglasEconomy, AggShockConsumerType
from HARK.cstwMPC.cstwMPC import cstwMPCagent, cstwMPCmarket, getKYratioDifference, findLorenzDistanceAtTargetKY, calcStationaryAgeDstn
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

from IPython import get_ipython # Needed to test whether being run from command line or interactively

import SetupParamsCSTW as Params

mystr = lambda number : "{:.3f}".format(number)


#%% Minimum example to understand natural borrowing constraint 

# Make AgentTypes for estimation
Test = cstwMPCagent(**Params.init_infinite)
Test.solve()
print('BoroCnstArt',Test.BoroCnstArt)
print('IncUnemp:',Test.IncUnemp)
print('UnempPrb:',Test.UnempPrb)
print('Limit:',Test.solution[0].cFunc.functions[0].x_list[0])


# Make AgentTypes for estimation
Test2 = cstwMPCagent(**Params.init_infinite)
Test2.IncUnemp = 0.68 #net unemp replacement rate in Norway
Test2.solve()
print('BoroCnstArt',Test2.BoroCnstArt)
print('IncUnemp:',Test2.IncUnemp)
print('UnempPrb:',Test2.UnempPrb)
print('Limit:',Test2.solution[0].cFunc.functions[0].x_list[0])

# Make AgentTypes for estimation
Test3 = cstwMPCagent(**Params.init_infinite)
Test3.BoroCnstArt = -20 #5 times perm (annual) income
Test3.IncUnemp = 0.68

Test3.solve()
print('BoroCnstArt',Test3.BoroCnstArt)
print('IncUnemp:',Test3.IncUnemp)
print('UnempPrb:',Test3.UnempPrb)
print('Limit:',Test3.solution[0].cFunc.functions[0].x_list[0])

#%%
run_estimation = True
dist_type = 'uniform'


param_name = 'DiscFac'
param_text = 'beta'
do_lifecycle = False
life_text = 'PY'
do_param_dist = True
model_text = 'Dist'
do_liquid = False
wealth_text = 'NetWorth'
do_agg_shocks = False
shock_text = 'Ind'
spec_name = life_text + param_text + model_text + shock_text + wealth_text

if do_param_dist:
    pref_type_count = 7       # Number of discrete beta types in beta-dist
else:
    pref_type_count = 1       # Just one beta type in beta-point
    
#%%    

###############################################################################
### ACTUAL WORK BEGINS BELOW THIS LINE  #######################################
###############################################################################


  
# Set targets for K/Y and the Lorenz curve based on the data
if do_liquid:
    lorenz_target = np.array([0.0, 0.004, 0.025,0.117]) #This is still US data
    KY_target = 9.2088
else: # This is hacky until I can find the liquid wealth data and import it
    lorenz_target = np.array([-0.042, -0.022, 0.080,0.294]) 
    lorenz_target_interp = np.interp(np.arange(0.01,1.00,0.01),np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),np.array([-0.039, -0.042, -0.039, -0.022, 0.017, 0.080, 0.17, 0.294, 0.472, 1]))
    lorenz_long_data = np.hstack((np.array(0.0),lorenz_target_interp,np.array(1.0)))  
    KY_target = 9.2088
    
# Set total number of simulated agents in the population
if do_param_dist:
    if do_agg_shocks:
        Population = Params.pop_sim_agg_dist
    else:
        Population = Params.pop_sim_ind_dist
else:
    if do_agg_shocks:
        Population = Params.pop_sim_agg_point
    else:
        Population = Params.pop_sim_ind_point
    


# Make AgentTypes for estimation
PerpetualYouthType = cstwMPCagent(**Params.init_infinite)
PerpetualYouthType.AgeDstn = np.array(1.0)
# Set Borrowing constraint to -5 of permanent income
# since a quarterly model, -20 of qu. permanent income
PerpetualYouthType.BoroCnstArt = -20
PerpetualYouthType.IncUnemp = 0.68

#%%

EstimationAgentList = []
for n in range(pref_type_count):
    EstimationAgentList.append(deepcopy(PerpetualYouthType))

# Give all the AgentTypes different seeds
for j in range(len(EstimationAgentList)):
    EstimationAgentList[j].seed = j

# Make an economy for the consumers to live in
market_dict = copy(Params.init_market)
market_dict['AggShockBool'] = do_agg_shocks
market_dict['Population'] = Population
EstimationEconomy = cstwMPCmarket(**market_dict)

# set Replacement rate to 68% following https://stats.oecd.org/Index.aspx?DataSetCode=NRR
EstimationEconomy.IncUnemp = 0.68

EstimationEconomy.agents = EstimationAgentList
EstimationEconomy.KYratioTarget = KY_target
EstimationEconomy.LorenzTarget = lorenz_target
EstimationEconomy.LorenzData = lorenz_long_data

EstimationEconomy.PopGroFac = 1.0
EstimationEconomy.TypeWeight = [1.0]
EstimationEconomy.act_T = Params.T_sim_PY
EstimationEconomy.ignore_periods = Params.ignore_periods_PY

#%%



#%%
center=0.9879177102415481
spread=0.004534079415384556
EstimationEconomy(LorenzBool = False, ManyStatsBool = False) # Make sure we're not wasting time calculating stuff
EstimationEconomy.distributeParams(param_name,pref_type_count,center,spread,dist_type) # Distribute parameters
EstimationEconomy.solve()


#%%
# Estimate the model as requested
if run_estimation:
    print('Beginning an estimation with the specification name ' + spec_name + '...')
    
    # Choose the bounding region for the parameter search
    param_range = [0.95,0.995]
    spread_range = [0.006,0.008]

    if do_param_dist:
        # Run the param-dist estimation
        paramDistObjective = lambda spread : findLorenzDistanceAtTargetKY(
                                                        Economy = EstimationEconomy,
                                                        param_name = param_name,
                                                        param_count = pref_type_count,
                                                        center_range = param_range,
                                                        spread = spread,
                                                        dist_type = dist_type)
        t_start = clock()
        spread_estimate = (minimize_scalar(paramDistObjective,bracket=spread_range,tol=1e-2,method='brent')).x
        center_estimate = EstimationEconomy.center_save
        t_end = clock()
    else:
        # Run the param-point estimation only
        paramPointObjective = lambda center : getKYratioDifference(Economy = EstimationEconomy,
                                             param_name = param_name,
                                             param_count = pref_type_count,
                                             center = center,
                                             spread = 0.0,
                                             dist_type = dist_type)
        t_start = clock()
        center_estimate = brentq(paramPointObjective,param_range[0],param_range[1],xtol=1e-2)
        spread_estimate = 0.0
        t_end = clock()

    # Display statistics about the estimated model
    #center_estimate = 0.986609223266
    #spread_estimate = 0.00853886395698
    EstimationEconomy.LorenzBool = True
    EstimationEconomy.ManyStatsBool = True
    EstimationEconomy.distributeParams(param_name, pref_type_count,center_estimate,spread_estimate, dist_type)
    EstimationEconomy.solve()
    EstimationEconomy.calcLorenzDistance()
    print('Estimate is center=' + str(center_estimate) + ', spread=' + str(spread_estimate) + ', took ' + str(t_end-t_start) + ' seconds.')
    EstimationEconomy.center_estimate = center_estimate
    EstimationEconomy.spread_estimate = spread_estimate
    EstimationEconomy.showManyStats(spec_name)
    print('These results have been saved to ./Code/Results/' + spec_name + '.txt\n\n')
