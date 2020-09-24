'''
This is the main script for the paper
'''
from Parameters import T_sim, init_infhorizon, DiscFacDstns,\
     AgentCountTotal, figs_dir
from FiscalModel import FiscalType
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
    BaseTypeList = [InfHorizonType]
    
    # Fill in the Markov income distribution for each base type
    # NOTE: THIS ASSUMES 4 MARKOV STATES, AND NO LIFECYCLE
    IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonType.IncUnemp])])
    IncomeDstn_big = []
    for ThisType in BaseTypeList:
        IncomeDstn_big.append([ThisType.IncomeDstn[0], IncomeDstn_unemp, ThisType.IncomeDstn[0], IncomeDstn_unemp])
        ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0], IncomeDstn_unemp]
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

    # Run the baseline consumption level
    t0 = time()
    C_base, X_base, Z_base, cAll_base, Weight_base, Mrkv_base, U_base, ltAll_base, LT_by_inc_base = runExperiment(**base_dict)
    t1 = time()
    print('Calculating baseline consumption took ' + mystr(t1-t0) + ' seconds.')
 
    t_end = time()
    print('Doing everything took ' + mystr(t_end-t_start) + ' seconds in total.')
    
