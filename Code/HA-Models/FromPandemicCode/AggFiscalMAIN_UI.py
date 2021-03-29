'''
This is the main script for the paper
'''
#$$$$$$$$$$ represents places in the code that need to be adjusted when the markov state space is changed
from Parameters import T_sim, init_infhorizon, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
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
Run_UB_Ext              = True
Run_Recession           = True


if __name__ == '__main__':
    

    
    # Setting up AggDemandEconmy
    from setupEconomy import AggDemandEconomy, base_dict_agg, max_recession_duration, output_keys, recession_prob_array, \
                             max_policy_duration, policy_prob_array
        
        
    if Run_Baseline:   
        # Run the baseline consumption level
        t0 = time()
        base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = False)
        saveAsPickleUnderVarName(base_results,figs_dir,locals())
        AggDemandEconomy.storeBaseline(base_results['AggCons'])     
        t1 = time()
        print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
        
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
        saveAsPickleUnderVarName(recession_results_AD,figs_dir,locals())
        t1 = time()
        print('Calculating recession consumption took ' + mystr(t1-t0) + ' seconds.')
    
    if Run_UB_Ext:
        
        print('Solving and simulating unemployment extension experiment')
        
        # Solving tax cut under Agg Multiplier  
        t0 = time()
        AggDemandEconomy.solveAD_UIExtension_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'UI_Rec')
        t1 = time()
        print('Solving UI during recession took ' + mystr(t1-t0) + ' seconds.')
           
  
        # Run the recession and extended UI consumption level
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'UI_Rec')
        recession_UI_dict = base_dict_agg.copy()
        recession_UI_dict.update(**recession_UI_changes)
        recession_UI_all_results = []
        recession_UI_results_AD = dict()
        for t_R in range(max_recession_duration):
            for t_Policy in range(max_policy_duration):
                recession_UI_dict['EconomyMrkv_init'] = np.array([0]*max(max_recession_duration,max_policy_duration))
                recession_UI_dict['EconomyMrkv_init'][0:t_R+1] += 1
                recession_UI_dict['EconomyMrkv_init'][0:t_Policy+1] +=2
                print('Running Experiment with Mrkv history: ', recession_UI_dict['EconomyMrkv_init'])
                this_recession_UI_results = AggDemandEconomy.runExperiment(**recession_UI_dict, Full_Output = False)
                recession_UI_all_results += [this_recession_UI_results]
        for key in output_keys:
            recession_UI_results_AD[key] = np.zeros_like(recession_UI_all_results[0][key])
            count = 0
            for t_R in range(max_recession_duration):
                for t_Policy in range(max_policy_duration):
                    recession_UI_results_AD[key] += recession_UI_all_results[count][key]*recession_prob_array[t_R]*policy_prob_array[t_Policy]
                    count += 1
        saveAsPickleUnderVarName(recession_UI_results_AD,figs_dir,locals())
        t1 = time()
        print('Calculating recession and extended UI consumption took ' + mystr(t1-t0) + ' seconds.')
    
#%%
    
    max_T = 20
    x_axis = np.arange(1,21)
    
    dir_baserun   = './Figures/Full_Run_Mar11_AD_Elas05/'
    dir_recession = './Figures/Full_Run_Mar11_AD_Elas05/'
    dir_AD        = './Figures/UI_AD05_new/'
    
    base_results                = loadPickle('base_results',            dir_baserun,locals())
    recession_results           = loadPickle('recession_results',       dir_baserun,locals())
    recession_results_AD        = loadPickle('recession_results_AD',    dir_recession,locals())
    recession_UI_results        = loadPickle('recession_UI_results',    dir_baserun,locals())
    recession_UI_results_AD     = loadPickle('recession_UI_results_AD', dir_AD,locals())
    
    
    
    AddCons_UI_Ext_Rec_RelRec       = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggCons')
    AddInc_UI_Ext_Rec_RelRec        = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggIncome') 
     
    NPV_AddInc_UI_Rec               = getSimulationDiff(recession_results,           recession_UI_results,'NPV_AggIncome')
    Stimulus_UI_Rec                 = getStimulus(recession_results,                 recession_UI_results,NPV_AddInc_UI_Rec[-1]) 
    NPV_Multiplier_UI_Rec           = getNPVMultiplier(recession_results,recession_UI_results,NPV_AddInc_UI_Rec)


    AddCons_UI_Ext_Rec_RelRec_AD    = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggCons')
    AddInc_UI_Ext_Rec_RelRec_AD     = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggIncome')  

    Stimulus_UI_Rec_AD              = getStimulus(recession_results_AD,                 recession_UI_results_AD,NPV_AddInc_UI_Rec[-1]) 
    NPV_Multiplier_UI_Rec_AD        = getNPVMultiplier(recession_results_AD,            recession_UI_results_AD,NPV_AddInc_UI_Rec)        
    
   
    
    plt.figure(figsize=(15,10))
    plt.title('Recession + UI extension', size=30)
    plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec[0:max_T],     color='blue',linestyle='-')
    plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_AD[0:max_T],  color='blue',linestyle='--')
    plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec[0:max_T],    color='red',linestyle='-')
    plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_AD[0:max_T], color='red',linestyle='--')
    plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=14)
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('quarter', fontsize=18)
    plt.ylabel('% diff. rel. to recession', fontsize=16)
    plt.savefig(figs_dir +'recession_UI_relrecession.pdf')
    plt.show()  
    
    
    
    
    




