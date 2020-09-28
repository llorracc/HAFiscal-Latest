'''
This is the main script for the paper
'''
#$$$$$$$$$$ represents places in the code that need to be adjusted when the markov state space is changed
from Parameters import T_sim, init_infhorizon, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, figs_dir
from FiscalModel import FiscalType
from FiscalTools import runExperiment
from HARK import multiThreadCommands, multiThreadCommandsFake
from HARK.distribution import DiscreteDistribution
from time import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


if __name__ == '__main__':
    
    mystr = lambda x : '{:.2f}'.format(x)
    t_start = time()

    # Make baseline types - for now only one type, might add more
    num_types = 1
    InfHorizonType = FiscalType(**init_infhorizon)
    InfHorizonType.cycles = 0
    BaseTypeList = [InfHorizonType]
    
    # Fill in the Markov income distribution for each base type
    # NOTE: THIS ASSUMES NO LIFECYCLE
    IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonType.IncUnemp])])
    IncomeDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonType.IncUnempNoBenefits])])
    IncomeDstn_big = []
    for ThisType in BaseTypeList:
        IncomeDstn_big.append([ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp]) #$$$$$$$$$$ six markov states
        ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp]
        ThisType.IncomeDstn_big = IncomeDstn_big
            
    # Make the overall list of types
    TypeList = []
    n = 0
    for b in range(DiscFacDstns[0].X.size):
        for e in range(num_types):
            DiscFac = DiscFacDstns[e].X[b]
            AgentCount = int(np.floor(AgentCountTotal*TypeShares[e]*DiscFacDstns[e].pmf[b]))
            ThisType = deepcopy(BaseTypeList[e])
            ThisType.AgentCount = AgentCount
            ThisType.DiscFac = DiscFac
            ThisType.seed = n
            TypeList.append(ThisType)
            n += 1
    base_dict['Agents'] = TypeList
       
    # Solve and simulate each type to get to the initial distribution of states
    # and then prepare for new counterfactual simulations
    t0 = time()
    baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()',
                         'switchToCounterfactualMode()', 'makeAlternateShockHistories()']
    multiThreadCommandsFake(TypeList, baseline_commands)
    t1 = time()
    print('Making the baseline distribution of states and preparing to run counterfactual simulations took ' + mystr(t1-t0) + ' seconds.')

    # Define dictionaries to be used in counterfactual scenarios
    recession_dict = base_dict.copy()
    recession_dict.update(**recession_changes)

    # Run the baseline consumption level
    t0 = time()
    base_results = runExperiment(**base_dict)
    t1 = time()
    print('Calculating baseline consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # Run the recession consumption level
    t0 = time()
    recession_results = runExperiment(**recession_dict)
    t1 = time()
    print('Calculating recession consumption took ' + mystr(t1-t0) + ' seconds.')
 
    t_end = time()
    print('Doing everything took ' + mystr(t_end-t_start) + ' seconds in total.')
    
