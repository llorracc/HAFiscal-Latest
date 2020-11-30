'''
This is the main script for the paper
'''
#$$$$$$$$$$ represents places in the code that need to be adjusted when the markov state space is changed
from Parameters import T_sim, init_infhorizon, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
     figs_dir
from FiscalModel import FiscalType
from AggFiscalModel import AggFiscalType, AggregateDemandEconomy
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

    InfHorizonType = FiscalType(**init_infhorizon)
    InfHorizonType.cycles = 0
    base_dict['Agents'] = [InfHorizonType]
    
    InfHorizonTypeAgg = AggFiscalType(**init_infhorizon)
    InfHorizonTypeAgg.cycles = 0
    base_dict_agg = deepcopy(base_dict)
    AggDemandEconomy = AggregateDemandEconomy(**init_ADEconomy)
    InfHorizonTypeAgg.getEconomyData(AggDemandEconomy)
    AggDemandEconomy.Agents = [InfHorizonTypeAgg]
    base_dict_agg['Agents'] = [InfHorizonTypeAgg]
  
    # Fill in the Markov income distribution for each base type
    #$$$$$$$$$$
    # NOTE: THIS ASSUMES NO LIFECYCLE
    IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonType.IncUnemp])])
    IncomeDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonType.IncUnempNoBenefits])])
    IncomeDstn_big = []
    for ThisType in [InfHorizonType, InfHorizonTypeAgg]:
        IncomeDstn_taxcut = deepcopy(ThisType.IncomeDstn[0])
        IncomeDstn_taxcut.X[1] = IncomeDstn_taxcut.X[1]*ThisType.TaxCutIncFactor
        IncomeDstn_big.append([ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, extended UI
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, extended UI
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp])  # recession, payroll tax cut
        ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp]
        ThisType.IncomeDstn_big = IncomeDstn_big
        ThisType.AgentCount = AgentCountTotal
        ThisType.DiscFac = 0.96
        ThisType.seed = 0
    
    baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()',
                         'switchToCounterfactualMode()', 'makeAlternateShockHistories()']

    InfHorizonType.solve()
    InfHorizonTypeAgg.solve()
    
    InfHorizonType.initializeSim()
    InfHorizonTypeAgg.initializeSim()
    InfHorizonTypeAgg.AggDemandFac = 1.0
    InfHorizonTypeAgg.RfreeNow = 1.0
    InfHorizonTypeAgg.CaggNow = 1.0
    
    InfHorizonType.simulate()
    InfHorizonTypeAgg.simulate()
    
    InfHorizonType.saveState()
    InfHorizonTypeAgg.saveState()
    
    InfHorizonType.switchToCounterfactualMode()
    AggDemandEconomy.switchToCounterfactualMode()
    # InfHorizonTypeAgg.switchToCounterfactualMode()
    
    InfHorizonType.makeAlternateShockHistories()
    InfHorizonTypeAgg.makeAlternateShockHistories()
    
    # Run the baseline consumption level
    t0 = time()
    base_results = runExperiment(**base_dict)
    t1 = time()
    print('Calculating baseline consumption took ' + mystr(t1-t0) + ' seconds.')
    
    ##WHY DOES THIS TAKE SO LONG???? Because it has to solve the model, and this takes much longer here.
    # Run the baseline consumption level
    t0 = time()
    agg_results = runExperiment(**base_dict_agg)
    t1 = time()
    print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')


plt.plot(np.sum(base_results['mNrm_all'],1))
plt.plot(np.sum(agg_results['mNrm_all'],1))

plt.plot(np.sum(base_results['cLvl_all_splurge'],1))
plt.plot(np.sum(agg_results['cLvl_all_splurge'],1))

# x_array = np.linspace(0,100,2000)
# base_array = base0(x_array)
# agg_array = agg0(x_array,np.ones_like(x_array))
# plt.plot(x_array,base_array)
# plt.plot(x_array,agg_array)
# plt.plot(x_array,base_array-agg_array)