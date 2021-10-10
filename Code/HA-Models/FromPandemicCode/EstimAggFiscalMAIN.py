'''
This is the main script for estimating the discount factor distributions.
'''
from time import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import namedtuple 
import pickle
import random 
from HARK.distribution import DiscreteDistribution, Uniform
from HARK import multiThreadCommands, multiThreadCommandsFake
from HARK.utilities import getPercentiles, getLorenzShares
from HARK.estimation import minimizeNelderMead
from OtherFunctions import getSimulationDiff, getSimulationPercentDiff, getStimulus, getNPVMultiplier, \
                    saveAsPickleUnderVarName, loadPickle, namestr     
from EstimParameters import T_sim, init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
     DiscFacCount, AgentCountTotal, base_dict, figs_dir, num_max_iterations_solvingAD,\
     convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
     data_LorenzPts, data_LorenzPtsAll, data_avgLWPI, data_LWoPI, data_EducShares, data_WealthShares,\
     DiscFacInit, DiscFacSpread
from EstimAggFiscalModel import AggFiscalType, AggregateDemandEconomy
mystr = lambda x : '{:.2f}'.format(x)

# -----------------------------------------------------------------------------
def calcEstimStats(Agents):
    '''
    Calculate the average LW/PI-ratio and total LW / total PI for each education
    type. Also calculate the 20th, 40th, 60th, and 80th percentile points of the
    Lorenz curve for (liquid) wealth for all agents. 
    Assumption: Agents is organized by EducType and there are DiscFacCount
    AgentTypes of each EducType. 
    
    Parameters
    ----------
    Agents : [AgentType]
        List of AgentTypes in the economy.
        
    Returns
    -------
    Stats : namedtuple("avgLWPI", "LWoPI", "LorenzPts")
    avgLWPI : [float] 
        The weighted average of LW/PI-ratio for each education type.
    LWoPI : [flota]
        Total liquid wealth / total permanent income for each education type. 
    LorenzPts : [float]
        The 20th, 40th, 60th, and 80th percentile points of the Lorenz curve for 
        (liquid) wealth.
    '''

    aLvlAll = np.concatenate([ThisType.aLvlNow for ThisType in Agents])
    numAgents = 0
    for ThisType in Agents: 
        numAgents += ThisType.AgentCount
    weights = np.ones(numAgents) / numAgents      # just using equal weights for now

    # Lorenz points:
    LorenzPts = 100*getLorenzShares(aLvlAll, weights=weights, percentiles = [0.2, 0.4, 0.6, 0.8] )

    avgLWPI = [0]*num_types
    LWoPI = [0]*num_types 
    for e in range(num_types):
        aNrmAll_byEd = []
        aNrmAll_byEd = np.concatenate([ThisType.aNrmNow for ThisType in \
                          Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        weights = np.ones(len(aNrmAll_byEd))/len(aNrmAll_byEd)
        avgLWPI[e] = np.dot(aNrmAll_byEd, weights) * 100
        
        aLvlAll_byEd = []
        aLvlAll_byEd = np.concatenate([ThisType.aLvlNow for ThisType in \
                          Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        pLvlAll_byEd = []
        pLvlAll_byEd = np.concatenate([ThisType.pLvlNow for ThisType in \
                          Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        LWoPI[e] = np.dot(aLvlAll_byEd, weights) / np.dot(pLvlAll_byEd, weights) * 100

    Stats = namedtuple("Stats", ["avgLWPI", "LWoPI", "LorenzPts"])

    return Stats(avgLWPI, LWoPI, LorenzPts) 
# -----------------------------------------------------------------------------
def calcWealthShareByEd(Agents):
    '''
    Calculate the share of total wealth held by each education type. 
    Assumption: Agents is organized by EducType and there are DiscFacCount
    AgentTypes of each EducType. 
    
    Parameters
    ----------
    Agents : [AgentType]
        List of all AgentTypes in the economy. They are assumed to differ in 
        their EducType attribute.

    Returns
    -------
    WealthShares : np.array(float)
        The share of total liquid wealth held by each education type. 
    '''
    aLvlAll = np.concatenate([ThisType.aLvlNow for ThisType in Agents])
    totLiqWealth = np.sum(aLvlAll)
    
    WealthShares = [0]*num_types
    for e in range(num_types):
        aLvlAll_byEd = []
        aLvlAll_byEd = np.concatenate([ThisType.aLvlNow for ThisType in \
                                       Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        WealthShares[e] = np.sum(aLvlAll_byEd)/totLiqWealth * 100
    
    return np.array(WealthShares)
# -----------------------------------------------------------------------------
def calcLorenzPts(Agents):
    '''
    Calculate the 20th, 40th, 60th, and 80th percentile points of the
    Lorenz curve for (liquid) wealth for the given set of Agents. 

    Parameters
    ----------
    Agents : [AgentType]
        List of AgentTypes.

    Returns
    -------
    LorenzPts : [float]
        The 20th, 40th, 60th, and 80th percentile points of the Lorenz curve for 
        (liquid) wealth.
    '''
    aLvlAll = np.concatenate([ThisType.aLvlNow for ThisType in Agents])
    numAgents = 0
    for ThisType in Agents: 
        numAgents += ThisType.AgentCount
    weights = np.ones(numAgents) / numAgents      # just using equal weights for now
    
    # Lorenz points:
    LorenzPts = 100*getLorenzShares(aLvlAll, weights=weights, percentiles = [0.2, 0.4, 0.6, 0.8] )

    return LorenzPts
# -----------------------------------------------------------------------------
#%% 
# Make education types
num_types = 3
# This is not the number of discount factors, but the number of household types

InfHorizonTypeAgg_d = AggFiscalType(**init_dropout)
InfHorizonTypeAgg_d.cycles = 0
InfHorizonTypeAgg_h = AggFiscalType(**init_highschool)
InfHorizonTypeAgg_h.cycles = 0
InfHorizonTypeAgg_c = AggFiscalType(**init_college)
InfHorizonTypeAgg_c.cycles = 0
AggDemandEconomy = AggregateDemandEconomy(**init_ADEconomy)
InfHorizonTypeAgg_d.getEconomyData(AggDemandEconomy)
InfHorizonTypeAgg_h.getEconomyData(AggDemandEconomy)
InfHorizonTypeAgg_c.getEconomyData(AggDemandEconomy)
BaseTypeList = [InfHorizonTypeAgg_d, InfHorizonTypeAgg_h, InfHorizonTypeAgg_c ]
      
# Fill in the Markov income distribution for each base type
# NOTE: THIS ASSUMES NO LIFECYCLE
IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg_d.IncUnemp])])
IncomeDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg_d.IncUnempNoBenefits])])
    
for ThisType in BaseTypeList:
    EmployedIncomeDstn = deepcopy(ThisType.IncomeDstn[0])
    ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0]] + [IncomeDstn_unemp]*UBspell_normal + [IncomeDstn_unemp_nobenefits] 
    ThisType.IncomeDstn_base = ThisType.IncomeDstn
    
# Make the overall list of types
TypeList = []
n = 0
for e in range(num_types):
    for b in range(DiscFacCount):
        DiscFac = DiscFacDstns[e].X[b]
        AgentCount = int(np.floor(AgentCountTotal*data_EducShares[e]*DiscFacDstns[e].pmf[b]))
        ThisType = deepcopy(BaseTypeList[e])
        ThisType.AgentCount = AgentCount
        ThisType.DiscFac = DiscFac
        ThisType.seed = n
        TypeList.append(ThisType)
        n += 1
base_dict['Agents'] = TypeList    

AggDemandEconomy.agents = TypeList
AggDemandEconomy.solve()

AggDemandEconomy.reset()
for agent in AggDemandEconomy.agents:
    agent.initializeSim()
    agent.AggDemandFac = 1.0
    agent.RfreeNow = 1.0
    agent.CaggNow = 1.0

AggDemandEconomy.makeHistory()   
AggDemandEconomy.saveState()   
#AggDemandEconomy.switchToCounterfactualMode("base")
#AggDemandEconomy.makeIdiosyncraticShockHistories()

output_keys = ['NPV_AggIncome', 'NPV_AggCons', 'AggIncome', 'AggCons']


#%% 
# -----------------------------------------------------------------------------
def betasObjFunc(betas, spreads, target_option=1, print_mode=False):
    '''
    Objective function for the estimation of discount factor distributions for the 
    three education groups. The groups differ in the centering of their discount 
    factor distributions, but have the same spread around the central value.
    
    Parameters
    ----------
    betas : [float]
        Central values of the discount factor distributions for each education
        level.
    spread : float
        Half the width of each discount factor distribution.
    print_mode : boolean, optional
        If true, statistics for each education level are printed. The default is False.
    target_option : integer
        = 1: Target avgLWPI and LorenzPtsAll 
        = 2: Target avgLWPI and LorenzPts_d, _h and _c

    Returns
    -------
    distance : float
        The distance of the estimation targets between those in the data and those
        produced by the model. 
    '''
    # # Set seed to ensure distance only changes due to different parameters 
    # random.seed(1234)

    beta_d, beta_h, beta_c = betas
    spread_d, spread_h, spread_c = spreads

    # # Overwrite the discount factor distribution for each education level with new values
    dfs_d = Uniform(beta_d-spread_d, beta_d+spread_d).approx(DiscFacCount)
    dfs_h = Uniform(beta_h-spread_h, beta_h+spread_h).approx(DiscFacCount)
    dfs_c = Uniform(beta_c-spread_c, beta_c+spread_c).approx(DiscFacCount)
    dfs = [dfs_d, dfs_h, dfs_c]

    # # Update discount factors of all agents 
    # for e in range(num_types):
    #     for b in range(DiscFacCount):
    #         TypeList[b+e*DiscFacCount].DiscFac = DiscFacDstns[e].X[b]
    #         TypeList[b+e*DiscFacCount].seed = n

    # Make a new list of types with updated discount factors 
    TypeListNew = []
    n = 0
    for e in range(num_types):
        for b in range(DiscFacCount):
            AgentCount = int(np.floor(AgentCountTotal*data_EducShares[e]*dfs[e].pmf[b]))
            ThisType = deepcopy(BaseTypeList[e])
            ThisType.AgentCount = AgentCount
            ThisType.DiscFac = dfs[e].X[b]
            ThisType.seed = n
            TypeListNew.append(ThisType)
            n += 1
    base_dict['Agents'] = TypeListNew

    AggDemandEconomy.agents = TypeListNew
    AggDemandEconomy.solve()

    AggDemandEconomy.reset()
    for agent in AggDemandEconomy.agents:
        agent.initializeSim()
        agent.AggDemandFac = 1.0
        agent.RfreeNow = 1.0
        agent.CaggNow = 1.0

    AggDemandEconomy.makeHistory()   
    AggDemandEconomy.saveState()   

    # Simulate each type to get a new steady state solution 
    # solve: done in AggDemandEconomy.solve(), initializeSim: done in AggDemandEconomy.reset() 
    # baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()']
    baseline_commands = ['simulate()', 'saveState()']
    multiThreadCommandsFake(TypeListNew, baseline_commands)
    
    Stats = calcEstimStats(TypeListNew)
    
    sumSquares = np.sum((np.array(Stats.avgLWPI)-data_avgLWPI)**2)
    
    if target_option == 1:
        sumSquares += np.sum((np.array(10*Stats.LorenzPts) - 10*data_LorenzPtsAll)**2)
    elif target_option == 2:
        lp_d = calcLorenzPts(TypeListNew[0:DiscFacCount])
        lp_h = calcLorenzPts(TypeListNew[DiscFacCount:2*DiscFacCount])
        lp_c = calcLorenzPts(TypeListNew[2*DiscFacCount:3*DiscFacCount])
        
        sumSquares += np.sum((10*np.array(lp_d)-10*data_LorenzPts[0])**2)
        sumSquares += np.sum((10*np.array(lp_h)-10*data_LorenzPts[1])**2)
        sumSquares += np.sum((10*np.array(lp_c)-10*data_LorenzPts[2])**2)
    
    distance = np.sqrt(sumSquares)

    # When testing, print stats by education level
    if print_mode:
        print('Average LW/PI-ratios: D = ' + mystr(Stats.avgLWPI[0]) + ' H = ' + mystr(Stats.avgLWPI[1]) \
              + ' C = ' + mystr(Stats.avgLWPI[2])) 
        print('Lorenz shares - all:')
        print(Stats.LorenzPts)
        if target_option == 2:
            print('Lorenz shares - Dropouts:')
            print(lp_d)
            print('Lorenz shares - Highschool:')
            print(lp_h)
            print('Lorenz shares - College:')
            print(lp_c) 
        
        print('Distance = ' + mystr(distance))
        print('Total LW/Total PI: D = ' + mystr(Stats.LWoPI[0]) + ' H = ' + mystr(Stats.LWoPI[1]) \
              + ' C = ' + mystr(Stats.LWoPI[2]))
        WealthShares = calcWealthShareByEd(TypeListNew)
        print('Wealth Shares: D = ' + mystr(WealthShares[0]) + \
              ' H = ' + mystr(WealthShares[1]) + ' C = ' + mystr(WealthShares[2]))

    return distance 
# -----------------------------------------------------------------------------
#%%
# Test function (check that repeated calls yield same answer -> no difference due 
# to random numbers generated)
for i in range(3):
    betasObjFunc([0.94,0.95,0.96], 0.02, print_mode=True)

initValues = deepcopy(DiscFacInit)
initValues.append(DiscFacSpread)
betasObjFunc(initValues[0:3], initValues[3], print_mode=True)

betasObjFunc([0.90, 0.93, 0.96], 0.02, print_mode=True)   


betasObjFunc([0.94, 0.95, 0.96], 0.02, print_mode=True)   
betasObjFunc([0.92, 0.95, 0.97], 0.02, print_mode=True)   
betasObjFunc([0.92, 0.95, 0.97], 0.03, print_mode=True)   
betasObjFunc([0.90, 0.94, 0.98], 0.015, print_mode=True)   

#%%
# Estimate discount factor distributions 

f_temp = lambda x : betasObjFunc(x[0:3],x[3], target_option=1)
initValues = deepcopy(DiscFacInit)
initValues.append(DiscFacSpread)
opt_params = minimizeNelderMead(f_temp, initValues, verbose=True)

print('Finished estimating. Optimal betas are:')
print(opt_params[0:3]) 
print('Optimal spread = ' + str(opt_params[3]) )

betasObjFunc(opt_params[0:3], opt_params[3], target_option = 2, print_mode=True)
#%%
# Estimate discount factor distributions with separate spreads

f_temp = lambda x : betasObjFunc(x[0:3],x[3:6], target_option=2)
initValues = deepcopy(DiscFacInit)
initValues.append(DiscFacSpread)
initValues.append(DiscFacSpread)
initValues.append(DiscFacSpread)
opt_params = minimizeNelderMead(f_temp, initValues, verbose=True)

print('Finished estimating. Optimal betas are:')
print(opt_params[0:3]) 
print('Optimal spreads are:')
print(opt_params[3:6])
betasObjFunc(opt_params[0:3], opt_params[3:6], target_option = 2, print_mode=True)

#%% Some stored results: 
# Estimation 1: 
opt_params = [0.96825, 0.97116, 0.97213, 0.02505]    
# Estimation 2: 
opt_params = [0.96971, 0.98628, 0.98764, 0.00981]