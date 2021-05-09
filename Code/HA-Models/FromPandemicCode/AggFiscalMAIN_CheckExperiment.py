# This is the main script for the paper

#$$$$$$$$$$ represents places in the code that need to be adjusted when the markov state space is changed
from Parameters import T_sim, init_infhorizon, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
     Check_changes, recession_Check_changes,\
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



Run_Recession           = True
Run_Check               = True
Run_Baseline_Check      = False
Run_NonAD_Simulations   = False



# Setting up AggDemandEconmy
t0 = time()
from setupEconomy import AggDemandEconomy, base_dict_agg, max_recession_duration, output_keys, recession_prob_array
t1 = time()
print('Setting up everything took ' + mystr(t1-t0) + ' seconds.')

#%%                        

# Run the baseline consumption level
t0 = time()
base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = True)
t1 = time()
print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
AggDemandEconomy.storeBaseline(base_results['AggCons']) 


#%% 

if Run_Recession:
    # Solving recession under Agg Multiplier   
    t0 = time()
    AggDemandEconomy.solveAD_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession')
    t1 = time()
    print('Solving recession took ' + mystr(t1-t0) + ' seconds.')
    

    # Run the recession consumption level in presence of the Agg Multiplier
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'Recession')
    recession_dict = base_dict_agg.copy()
    recession_dict.update(**recession_changes)
    recession_all_results_AD = []
    recession_results_AD = dict()
    for t in range(max_recession_duration):
        recession_dict['EconomyMrkv_init'] = [1]*(t+1)
        this_recession_results_AD = AggDemandEconomy.runExperiment(**recession_dict, Full_Output = False)
        recession_all_results_AD += [this_recession_results_AD]
    for key in output_keys:
        recession_results_AD[key] = np.sum(np.array([recession_all_results_AD[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    saveAsPickleUnderVarName(recession_all_results_AD,figs_dir,locals())
    saveAsPickleUnderVarName(recession_results_AD,figs_dir,locals())
    t1 = time()
    print('Calculating recession consumption took ' + mystr(t1-t0) + ' seconds.')



#%%

if Run_Check:
    # get AD Solution  
    t0 = time()
    AggDemandEconomy.solveAD_Check_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Check_Rec')
    t1 = time()
    print('Solving Check during recession took ' + mystr(t1-t0) + ' seconds.')
    
    # Recession with Check with AD Effects
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'Check_Rec')
    recession_Check_dict = base_dict_agg.copy()
    recession_Check_dict.update(**recession_Check_changes)
    recession_Check_all_results_AD = []
    recession_Check_results_AD = dict()
    #  running recession with diferent lengths up to 20q then averaging the result
    for t in range(max_recession_duration):
        recession_Check_dict['EconomyMrkv_init'] = [1]*(t+1)
        recession_Check_dict['EconomyMrkv_init'][0] = 37
        print(recession_Check_dict['EconomyMrkv_init'])
        this_sim_results = AggDemandEconomy.runExperiment(**recession_Check_dict, Full_Output = True)
        recession_Check_all_results_AD += [this_sim_results]
    for key in output_keys:
        recession_Check_results_AD[key] = np.sum(np.array([recession_Check_all_results_AD[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    saveAsPickleUnderVarName(recession_Check_all_results_AD,figs_dir,locals())
    saveAsPickleUnderVarName(recession_Check_results_AD,figs_dir,locals())
    t1 = time()
    print('Calculating recession + check consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')
        
    


 
#%% Check during baseline 
    
if Run_Baseline_Check: 

       
    # Run Check experiment during baseline
    t0 = time()
    Check_dict = base_dict_agg.copy()
    Check_dict.update(**Check_changes)
    Check_dict['EconomyMrkv_init'] = [36]
    Check_results = AggDemandEconomy.runExperiment(**Check_dict, Full_Output = True)
    t1 = time()
    print('Calculating check experiment ' + mystr(t1-t0) + ' seconds.')
    

            
    # Investigate
    
    def InvestigateResults(base,check):
        x=getSimulationDiff(base,check,'pLvl_all')
        print('Maximum difference in pLvl_all: ', np.max(x), ' (should be 0)')
        
        x=getSimulationDiff(base,check,'TranShk_all')
        print('Diff in Trans Shk for t>1: ', np.max(x[1:]), ' (should be 0)')
        print('Trans Shocks for Agents: ',x[0][0:5])
        
        base_IndIncome  =  base['TranShk_all']*base['pLvl_all']
        Check_IndIncome =  check['TranShk_all']*check['pLvl_all']
        x = Check_IndIncome-base_IndIncome
        print('Diff in total Inc for t>1: ', np.max(x[1:]), ' (should be 0)')
        
        FullCheckAgents = np.less(AggDemandEconomy.agents[0].pLvl_base,AggDemandEconomy.agents[0].CheckStimLvl_PLvl_Cutoff_start)
        print('These agents should all get the 100 % of the Stimulus : ', x[0][FullCheckAgents]/AggDemandEconomy.agents[0].CheckStimLvl)
        
        ZeroCheckAgents = np.greater(AggDemandEconomy.agents[0].pLvl_base,AggDemandEconomy.agents[0].CheckStimLvl_PLvl_Cutoff_end)
        print('These agents should all get 0 % percent of the Stimulus : ', x[0][ZeroCheckAgents]/AggDemandEconomy.agents[0].CheckStimLvl)
        
        SomeCheckAgents = np.logical_and(np.greater(AggDemandEconomy.agents[0].pLvl_base,AggDemandEconomy.agents[0].CheckStimLvl_PLvl_Cutoff_start), np.less(AggDemandEconomy.agents[0].pLvl_base,AggDemandEconomy.agents[0].CheckStimLvl_PLvl_Cutoff_end))
        print('These agents should get between 0 and 100% of the stimulus : ', x[0][SomeCheckAgents]/AggDemandEconomy.agents[0].CheckStimLvl)
        
      
    InvestigateResults(base_results,Check_results)     
    
    # Plot Baseline Check experiment
    
    AddCons               = getSimulationPercentDiff(base_results,    Check_results,'AggCons')
    AddInc                = getSimulationPercentDiff(base_results,    Check_results,'AggIncome')
    
    max_T = 20
    x_axis = np.arange(1,21)
    
    plt.figure(figsize=(15,10))
    plt.plot(x_axis,AddCons[0:max_T], color='blue',linestyle='-')
    plt.plot(x_axis,AddInc[0:max_T],  color='blue',linestyle='--')
    plt.legend(['Cons','Inc'], fontsize=14)
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('quarter', fontsize=18)
    plt.show() 
    
    

    #%% Check during recession but no AD effects 
    if Run_Recession:
        
        # Recession only
        output_keys = ['NPV_AggIncome', 'NPV_AggCons', 'AggIncome', 'AggCons', 'pLvl_all','TranShk_all']
        
        
        
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'baseline')
        recession_dict = base_dict_agg.copy()
        recession_dict.update(**recession_changes)
        recession_all_results = []
        recession_results = dict()
        #  running recession with diferent lengths up to 20q then averaging the result
        for t in range(max_recession_duration):
            recession_dict['EconomyMrkv_init'] = [1]*(t+1)
            print(recession_dict['EconomyMrkv_init'])
            this_sim_results = AggDemandEconomy.runExperiment(**recession_dict, Full_Output = True)
            recession_all_results += [this_sim_results]
        for key in output_keys:
            recession_results[key] = np.sum(np.array([recession_all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
        t1 = time()
        print('Calculating recession consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')
        
        # Recession with Check, no AD
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'baseline')
        recession_Check_dict = base_dict_agg.copy()
        recession_Check_dict.update(**recession_Check_changes)
        recession_Check_all_results = []
        recession_Check_results = dict()
        #  running recession with diferent lengths up to 20q then averaging the result
        for t in range(max_recession_duration):
            recession_Check_dict['EconomyMrkv_init'] = [1]*(t+1)
            recession_Check_dict['EconomyMrkv_init'][0] = 37
            print(recession_Check_dict['EconomyMrkv_init'])
            this_sim_results = AggDemandEconomy.runExperiment(**recession_Check_dict, Full_Output = True)
            recession_Check_all_results += [this_sim_results]
        for key in output_keys:
            recession_Check_results[key] = np.sum(np.array([recession_Check_all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
        t1 = time()
        print('Calculating recession + check consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')
      
    
        InvestigateResults(recession_results,recession_Check_results)    
        #%% 
        
        AddCons               = getSimulationPercentDiff(base_results,    Check_results,'AggCons')
        AddInc                = getSimulationPercentDiff(base_results,    Check_results,'AggIncome')
        AddCons_Rec           = getSimulationPercentDiff(recession_results,    recession_Check_results,'AggCons')
        AddInc_Rec            = getSimulationPercentDiff(recession_results,    recession_Check_results,'AggIncome')
        
        max_T = 20
        x_axis = np.arange(1,21)
        
        plt.figure(figsize=(15,10))
        plt.plot(x_axis,AddCons[0:max_T], color='blue',linestyle='-')
        plt.plot(x_axis,AddInc[0:max_T],  color='blue',linestyle='--')
        plt.plot(x_axis,AddCons_Rec[0:max_T], color='red',linestyle='-')
        plt.plot(x_axis,AddInc_Rec[0:max_T],  color='red',linestyle='--')
        plt.legend(['Cons','Inc','Cons Rec','Inc Rec'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.show()     


