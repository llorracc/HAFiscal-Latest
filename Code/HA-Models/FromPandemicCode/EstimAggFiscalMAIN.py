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
from EstimParameters import init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
     DiscFacCount, Splurge, AgentCountTotal, base_dict, figs_dir, UBspell_normal, \
     data_LorenzPts, data_LorenzPtsAll, data_avgLWPI, data_LWoPI, data_medianLWPI,\
     data_EducShares, data_WealthShares
from EstimAggFiscalModel import AggFiscalType, AggregateDemandEconomy
mystr = lambda x : '{:.2f}'.format(x)
mystr4 = lambda x : '{:.4f}'.format(x)

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
    LWoPI : [float]
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
    medianLWPI = [0]*num_types 
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

        medianLWPI[e] = 100*getPercentiles(aNrmAll_byEd,weights=weights,percentiles=[0.5])

    Stats = namedtuple("Stats", ["avgLWPI", "LWoPI", "medianLWPI", "LorenzPts"])

    return Stats(avgLWPI, LWoPI, medianLWPI, LorenzPts) 
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
def calcMPCbyEd(Agents):
    '''
    Calculate the average MPC for each education type. 
    Assumption: Agents is organized by EducType and there are DiscFacCount
    AgentTypes of each EducType. 
    
    Parameters
    ----------
    Agents : [AgentType]
        List of all AgentTypes in the economy. They are assumed to differ in 
        their EducType attribute.

    Returns
    -------
    MPCs : namedtuple("MPCsQ", "MPCsA")    
    MPCsQ : [float]
        The average MPC for each education type - Quarterly, ignores splurge.
    MPCsA : [float]
        The average MPC for each education type - Annualized, taking splurge into account. 
    MPCsAalt : [float]
        Alternative measure of average annual MPC: Only splurge in the first quarter.
    '''
    MPCsQ = [0]*(num_types+1)
    MPCsA = [0]*(num_types+1)       # Annual MPCs with splurge every Q
    MPCsAalt = [0]*(num_types+1)    # Annual MPCs with splurge in Q1 only
    for e in range(num_types):
        MPC_byEd = []
        MPC_byEd = np.concatenate([ThisType.MPCnow for ThisType in \
                                       Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        MPCsQ[e] = np.mean(MPC_byEd)     # Avg. quarterly MPC for each ed group
        
        MPCsA[e] = Splurge + (1-Splurge)*MPCsQ[e]
        for qq in range(3):
            MPCsA[e] += (1-MPCsA[e])*(Splurge + (1-Splurge)*MPCsQ[e])
        
        MPCsAalt[e] = Splurge + (1-Splurge)*MPCsQ[e]
        for qq in range(3):
            MPCsAalt[e] += (1-MPCsAalt[e])*MPCsQ[e]

    MPC_all = np.concatenate([ThisType.MPCnow for ThisType in Agents])
    MPCsQ[e+1] = np.mean(MPC_all)

    MPCsA[e+1] = Splurge + (1-Splurge)*MPCsQ[e+1]
    for qq in range(3):
        MPCsA[e+1] += (1-MPCsA[e+1])*(Splurge + (1-Splurge)*MPCsQ[e+1])

    MPCsAalt[e+1] = Splurge + (1-Splurge)*MPCsQ[e+1]   # Splurge only in first Q
    for qq in range(3):
        MPCsAalt[e+1] += (1-MPCsAalt[e+1])*MPCsQ[e+1]

    MPCs = namedtuple("MPCs", ["MPCsQ", "MPCsA", "MPCsAalt"])

    return MPCs(MPCsQ,MPCsA,MPCsAalt)
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
    three education groups. The groups can differ in the centering of their discount 
    factor distributions, and in the spread around the central value.
    
    Parameters
    ----------
    betas : [float]
        Central values of the discount factor distributions for each education
        level.
    spreads : [float]
        Half the width of each discount factor distribution. If we want the same spread
        for each education group we simply impose that the spreads are all the same.
        That is done outside this function. 
    target_option : integer
        = 1: Target medianLWPI and LorenzPtsAll 
        = 2: Target avgLWPI and LorenzPts_d, _h and _c
    print_mode : boolean, optional
        If true, statistics for each education level are printed. The default is False.
    
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
    
    if target_option == 1:
        sumSquares = 10*np.sum((Stats.medianLWPI-data_medianLWPI)**2)
        sumSquares += np.sum((np.array(Stats.LorenzPts) - data_LorenzPtsAll)**2)
    elif target_option == 2:
        lp_d = calcLorenzPts(TypeListNew[0:DiscFacCount])
        lp_h = calcLorenzPts(TypeListNew[DiscFacCount:2*DiscFacCount])
        lp_c = calcLorenzPts(TypeListNew[2*DiscFacCount:3*DiscFacCount])
        sumSquares = np.sum((np.array(Stats.avgLWPI)-data_avgLWPI)**2)
        sumSquares += np.sum((np.array(lp_d)-data_LorenzPts[0])**2)
        sumSquares += np.sum((np.array(lp_h)-data_LorenzPts[1])**2)
        sumSquares += np.sum((np.array(lp_c)-data_LorenzPts[2])**2)
    
    distance = np.sqrt(sumSquares)

    # If not estimating, print stats by education level
    if print_mode:
        print('Dropouts: beta = ', mystr(beta_d), ' spread = ', mystr(spread_d))
        print('Highschool: beta = ', mystr(beta_h), ' spread = ', mystr(spread_h))
        print('College: beta = ', mystr(beta_c), ' spread = ', mystr(spread_c))
        print('Median LW/PI-ratios: D = ' + mystr(Stats.medianLWPI[0][0]) + ' H = ' + mystr(Stats.medianLWPI[1][0]) \
              + ' C = ' + mystr(Stats.medianLWPI[2][0])) 
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
        print('Average LW/PI-ratios: D = ' + mystr(Stats.avgLWPI[0]) + ' H = ' + mystr(Stats.avgLWPI[1]) \
              + ' C = ' + mystr(Stats.avgLWPI[2])) 
        print('Total LW/Total PI: D = ' + mystr(Stats.LWoPI[0]) + ' H = ' + mystr(Stats.LWoPI[1]) \
              + ' C = ' + mystr(Stats.LWoPI[2]))
        WealthShares = calcWealthShareByEd(TypeListNew)
        print('Wealth Shares: D = ' + mystr(WealthShares[0]) + \
              ' H = ' + mystr(WealthShares[1]) + ' C = ' + mystr(WealthShares[2]))
        MPCs = calcMPCbyEd(TypeListNew)
        print('Average Quarterly MPCs: D = ' + mystr(MPCs.MPCsQ[0]) + ' H = ' + mystr(MPCs.MPCsQ[1]) + \
              ' C = ' + mystr(MPCs.MPCsQ[2]) + ' All = ' + mystr(MPCs.MPCsQ[3]))
        print('Average annual MPCs, incl. splurge: D = ' + mystr(MPCs.MPCsA[0]) + ' H = ' + mystr(MPCs.MPCsA[1]) + \
              ' C = ' + mystr(MPCs.MPCsA[2]) + ' All = ' + mystr(MPCs.MPCsA[3]))
        print('Average annual MPCs, splurge Q1 only: D = ' + mystr(MPCs.MPCsAalt[0]) + ' H = ' + mystr(MPCs.MPCsAalt[1]) + \
              ' C = ' + mystr(MPCs.MPCsAalt[2]) + ' All = ' + mystr(MPCs.MPCsAalt[3]))
            
    return distance 
# -----------------------------------------------------------------------------
def betasObjFuncEduc(beta, spread, educ_type=2, print_mode=False):
    '''
    Objective function for the estimation of a discount factor distribution for
    a single education group.
    
    Parameters
    ----------
    beta : float
        Central value of the discount factor distribution.
    spread : float
        Half the width of the discount factor distribution.
    educ_type : integer
        The education type to estimate a discount factor distribution for.     
        Targets are avgLWPI[educ_type] and LorenzPts[educ_type]
    print_mode : boolean, optional
        If true, statistics are printed. The default is False.
    
    Returns
    -------
    distance : float
        The distance of the estimation targets between those in the data and those
        produced by the model. 
    '''
    # # Set seed to ensure distance only changes due to different parameters 
    # random.seed(1234)

    dfs = Uniform(beta-spread, beta+spread).approx(DiscFacCount)

    # Make a new list of types with updated discount factors for the given educ type
    TypeListNewEduc = []
    n = 0
    for b in range(DiscFacCount):
        AgentCount = int(np.floor(AgentCountTotal*data_EducShares[educ_type]*dfs.pmf[b]))
        ThisType = deepcopy(BaseTypeList[educ_type])
        ThisType.AgentCount = AgentCount
        ThisType.DiscFac = dfs.X[b]
        ThisType.seed = n
        TypeListNewEduc.append(ThisType)
        n += 1
    TypeListAll = AggDemandEconomy.agents
    TypeListAll[educ_type*DiscFacCount:(educ_type+1)*DiscFacCount] = TypeListNewEduc
            
    base_dict['Agents'] = TypeListAll
    AggDemandEconomy.agents = TypeListAll
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
    multiThreadCommandsFake(TypeListAll, baseline_commands)
    
    Stats = calcEstimStats(TypeListAll)
    
    sumSquares = np.sum((Stats.medianLWPI[educ_type]-data_medianLWPI[educ_type])**2)
    lp = calcLorenzPts(TypeListNewEduc)
    sumSquares += np.sum((np.array(lp) - data_LorenzPts[educ_type])**2)
#    sumSquares = np.sum((Stats.avgLWPI[educ_type]-data_avgLWPI[educ_type])**2)
   
    distance = np.sqrt(sumSquares)

    # If not estimating, print stats by education level
    if print_mode:
        print('Median LW/PI-ratio for group e = ' + mystr(educ_type) + ' is: ' \
              + mystr(Stats.medianLWPI[educ_type][0]))
        if educ_type == 0:
            print('Lorenz shares - Dropouts:')
        elif educ_type == 1:
            print('Lorenz shares - Highschool:')
        elif educ_type == 2:
            print('Lorenz shares - College:')
        print(lp)
        print('Distance = ' + mystr(distance))
        print('Non-targeted moments:')
        print('Average LW/PI-ratios for group e = ' + mystr(educ_type) + ' is: ' \
              + mystr(Stats.avgLWPI[educ_type]))
        print('Lorenz shares - all:')
        print(Stats.LorenzPts)
        
    return distance 
# -----------------------------------------------------------------------------
#%%
# Estimate discount factor distributions with one spread

f_temp = lambda x : betasObjFunc(x[0:3],3*[x[3]], target_option=1)
initValues = [0.88, 0.963, 0.988, 0.01]
opt_params = minimizeNelderMead(f_temp, initValues, verbose=True)

print('Finished estimating. Optimal betas are:')
print(opt_params[0:3]) 
print('Optimal spread = ' + str(opt_params[3]) )

betasObjFunc(opt_params[0:3], 3*[opt_params[3]], target_option = 2, print_mode=True)

testVals = [0.88, 0.963, 0.988, 0.01]
betasObjFunc(testVals[0:3], 3*[testVals[3]], target_option = 2, print_mode=True)
#%%
# Estimate discount factor distributions with separate spreads

f_temp = lambda x : betasObjFunc(x[0:3],x[3:6], target_option=1)
initValues = [0.88, 0.963, 0.988, 0.05, 0.02, 0.01]
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
opt_params = [0.96971, 0.98628, 0.98764, 0.009815]
# Estimation 3: 
opt_params = [0.88622648, 0.98265369, 0.97298556, 0.12578517, 0.01426293, 0.0271796 ]
betasObjFunc(opt_params[0:3], 3*[opt_params[3]], target_option = 2, print_mode=True)    

# Implied discount factor distributions: 
DiscFacDstnD = Uniform(opt_params[0]-opt_params[3], opt_params[0]+opt_params[3]).approx(DiscFacCount)
DiscFacDstnH = Uniform(opt_params[1]-opt_params[3], opt_params[1]+opt_params[3]).approx(DiscFacCount)
DiscFacDstnC = Uniform(opt_params[2]-opt_params[3], opt_params[2]+opt_params[3]).approx(DiscFacCount)
print([DiscFacDstnD.X[0], DiscFacDstnD.X[6]])
print([DiscFacDstnH.X[0], DiscFacDstnH.X[6]])
print([DiscFacDstnC.X[0], DiscFacDstnC.X[6]])

#%% A test: 
test_vals = [0.88622648, 0.98265369, 0.98764, 0.12578517, 0.01426293, 0.00981 ]
betasObjFunc(test_vals[0:3], test_vals[3:6], target_option = 2, print_mode=True)

#%% Estimate discount factor distribution for one education type at a time

edType = 1
f_temp = lambda x : betasObjFuncEduc(x[0],x[1], educ_type=edType)
if edType == 0:
    initValues = [0.80, 0.15]       # Dropouts
elif edType == 1:
    initValues = [0.966, 0.03]      # HighSchool
elif edType == 2:
    initValues = [0.985, 0.012]     # College
else:
    initValues = [0.95,0.02]
opt_params = minimizeNelderMead(f_temp, initValues, verbose=True)
print('Finished estimating. Optimal beta and spread are:')
print(opt_params) 
betasObjFuncEduc(opt_params[0], opt_params[1], educ_type = edType, print_mode=True)


# Estimates targeting median LW/PI (v3.0): 
estimates_d = [0.79939338, 0.2279055 ]  # Dropouts only 
estimates_h = [0.93744293, 0.06607694]  # Highschool only
estimates_c = [0.98525333, 0.01241598] # College only
betasObjFuncEduc(estimates_d[0], estimates_d[1], educ_type = 0, print_mode=True)
betasObjFuncEduc(0.80, 0.15, educ_type = 0, print_mode=True)

betasObjFunc([estimates_d[0], estimates_h[0], estimates_c[0]], \
             [estimates_d[1], estimates_h[1], estimates_c[1]], \
             target_option = 1, print_mode=True)

# Implied discount factor distributions:
DiscFacDstnD = Uniform(estimates_d[0]-estimates_d[1], estimates_d[0]+estimates_d[1]).approx(DiscFacCount)
DiscFacDstnH = Uniform(estimates_h[0]-estimates_h[1], estimates_h[0]+estimates_h[1]).approx(DiscFacCount)
DiscFacDstnC = Uniform(estimates_c[0]-estimates_c[1], estimates_c[0]+estimates_c[1]).approx(DiscFacCount)
print('Discount factor distribution end points: ')
print('Dropouts:\t ', mystr4(DiscFacDstnD.X[0]), ' to ', mystr4(DiscFacDstnD.X[6]))    
print('Highschool:\t ', mystr4(DiscFacDstnH.X[0]), ' to ', mystr4(DiscFacDstnH.X[6])) 
print('College:\t ', mystr4(DiscFacDstnC.X[0]), ' to ', mystr4(DiscFacDstnC.X[6])) 
    

#%% 
# Estimates targeting median LW/PI (v2.5): 
#TranShkStd = [0.12]  #[np.sqrt(0.12)]
#PermShkStd = [0.003] #[np.sqrt(0.003)]
estimates_d = [0.87691380, 0.13952285]  # Dropouts only 
estimates_h = [0.97097283, 0.02996364]  # Highschool only
estimates_c = [0.99264389, 0.00596917]  # College only

# Old estimates targeting median LW/PI (v2.0): 
estimates_d = [0.87509113, 0.13891492]  # Dropouts only 
estimates_h = [0.96597689, 0.03307152]  # Highschool only
estimates_c = [0.9886787, 0.00772621]   # College only

# Old estimates targeting average LW/PI: 
estimates_d = [0.79602650, 0.02785033]  # Dropouts only
estimates_h = [0.98051198, 0.01675674]  # Highschool only
estimates_c = [0.99160378, 0.00480153]  # College only
 