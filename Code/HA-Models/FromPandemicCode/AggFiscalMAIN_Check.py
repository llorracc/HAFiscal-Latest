'''
This is the main script for the paper
'''
#$$$$$$$$$$ represents places in the code that need to be adjusted when the markov state space is changed
from Parameters import T_sim, init_infhorizon, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
     Check_changes, recession_Check_changes, \
     figs_dir, num_max_iterations_solvingAD, convergence_tol_solvingAD
from FiscalModel import FiscalType
from AggFiscalModel import AggFiscalType, AggregateDemandEconomy
from HARK.distribution import DiscreteDistribution
from time import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
from OtherFunctions import getSimulationDiff, getSimulationPercentDiff, getStimulus, getNPVMultiplier, \
                    saveAsPickleUnderVarName, loadPickle, namestr     
mystr = lambda x : '{:.2f}'.format(x)



## Which experiments to run / plots to show
Run_Baseline            = True
Run_UB_Ext_Recession    = True
Run_Recession           = False
Run_TaxCut_Recession    = False
Make_Plots              = True

Run_NonAD               = False





# Setting up AggDemandEconmy
from setupEconomy import AggDemandEconomy, base_dict_agg, max_recession_duration, output_keys, recession_prob_array, \
                         max_policy_duration, policy_prob_array
    
    

#base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = False)

# Run Check experiment
t0 = time()
Check_dict = base_dict_agg.copy()
Check_dict.update(**Check_changes)
Check_dict['EconomyMrkv_init'] = [36]
Check_results = AggDemandEconomy.runExperiment(**Check_dict, Full_Output = False)
t1 = time()
print('Calculating recession and extended UI consumption  (no AD) ' + mystr(t1-t0) + ' seconds.')

    
#%% 


    




