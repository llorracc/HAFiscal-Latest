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
from OtherFunctions import getSimulationDiff, getSimulationPercentDiff, getStimulus, saveAsPickleUnderVarName, \
                           loadPickle, namestr     
mystr = lambda x : '{:.2f}'.format(x)



## Which experiments to run / plots to show
Run_Baseline            = False
Run_TaxCut              = True
Run_Recession           = True
Run_TaxCut_Recession    = True
Run_UB_Ext              = True
Make_Plots              = True
Show_TestPlots          = False



if __name__ == '__main__':
    
    t_start = time()
    
    if Run_Baseline:
        # Setting up AggDemandEconmy
        from setupEconomy import AggDemandEconomy, base_dict_agg, max_recession_duration, output_keys, recession_prob_array, \
                                 max_policy_duration, policy_prob_array
        
        
        
        
        
        # Run the baseline consumption level
        t0 = time()
        base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = False)
        saveAsPickleUnderVarName(base_results,figs_dir,locals())
        AggDemandEconomy.storeBaseline(base_results['AggCons'])     
        t1 = time()
        print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
    
    
    if Run_UB_Ext:
        
        print('Solving and simulating unemployment extension experiment')
        
        # Run the extended UI consumption level
        t0 = time()
        UI_dict = base_dict_agg.copy()
        UI_dict.update(**UI_changes)
        UI_all_results = []
        UI_results = dict()
        for t_Policy in range(max_policy_duration):
            UI_dict['EconomyMrkv_init'] = [2]*(t_Policy+1)
            print('Running Experiment with Mrkv history: ', UI_dict['EconomyMrkv_init'])
            this_UI_results = AggDemandEconomy.runExperiment(**UI_dict, Full_Output = False)
            UI_all_results += [this_UI_results]
        for key in output_keys:
            UI_results[key] = np.sum(np.array([UI_all_results[t][key]*policy_prob_array[t]  for t in range(max_policy_duration)]), axis=0)
        saveAsPickleUnderVarName(UI_results,figs_dir,locals())
        t1 = time()
        print('Calculating extended UI consumption took ' + mystr(t1-t0) + ' seconds.')
    
  
        # Run the recession and extended UI consumption level
        t0 = time()
        recession_UI_dict = base_dict_agg.copy()
        recession_UI_dict.update(**recession_UI_changes)
        recession_UI_all_results = []
        recession_UI_results = dict()
        for t_R in range(max_recession_duration):
            for t_Policy in range(max_policy_duration):
                recession_UI_dict['EconomyMrkv_init'] = np.array([0]*max(max_recession_duration,max_policy_duration))
                recession_UI_dict['EconomyMrkv_init'][0:t_R+1] += 1
                recession_UI_dict['EconomyMrkv_init'][0:t_Policy+1] +=2
                print('Running Experiment with Mrkv history: ', recession_UI_dict['EconomyMrkv_init'])
                this_recession_UI_results = AggDemandEconomy.runExperiment(**recession_UI_dict, Full_Output = False)
                recession_UI_all_results += [this_recession_UI_results]
        for key in output_keys:
            recession_UI_results[key] = np.zeros_like(recession_UI_all_results[0][key])
            count = 0
            for t_R in range(max_recession_duration):
                for t_Policy in range(max_policy_duration):
                    recession_UI_results[key] += recession_UI_all_results[count][key]*recession_prob_array[t_R]*policy_prob_array[t_Policy]
                    count += 1
        saveAsPickleUnderVarName(recession_UI_results,figs_dir,locals())
        t1 = time()
        print('Calculating recession and extended UI consumption took ' + mystr(t1-t0) + ' seconds.')
    
    
    
    #%% Solving and Simulating
    
    if Run_TaxCut:
        # Solving tax cut under Agg Multiplier  
        t0 = time()
        AggDemandEconomy.solveAD_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'TaxCut')
        t1 = time()
        print('Solving payroll tax cut took ' + mystr(t1-t0) + ' seconds.')
        
        # Run the payroll tax cut consumption level in absence of Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'baseline')
        TaxCut_dict = base_dict_agg.copy()
        TaxCut_dict.update(**TaxCut_changes)
        TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
        TaxCut_results_intermediate = AggDemandEconomy.runExperiment(**TaxCut_dict, Full_Output = False)
        # only save important output keys
        TaxCut_results = dict()
        for key in output_keys:
            TaxCut_results[key] = TaxCut_results_intermediate[key]
        saveAsPickleUnderVarName(TaxCut_results,figs_dir,locals())
        t1 = time()
        print('Calculating payroll tax cut consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')
     
        # Run the payroll tax cut consumption level in presence of Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'TaxCut')
        TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
        TaxCut_results_AD_intermediate = AggDemandEconomy.runExperiment(**TaxCut_dict, Full_Output = False)
        # only save important output keys
        TaxCut_results_AD = dict()
        for key in output_keys:
            TaxCut_results_AD[key] = TaxCut_results_AD_intermediate[key]
        saveAsPickleUnderVarName(TaxCut_results_AD,figs_dir,locals())
        t1 = time()
        print('Calculating payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
    
    if Run_Recession:
        # Solving recession under Agg Multiplier   
        t0 = time()
        AggDemandEconomy.solveAD_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession')
        t1 = time()
        print('Solving recession took ' + mystr(t1-t0) + ' seconds.')
        
        
        # Run the recession consumption level in absence of Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'baseline')
        recession_dict = base_dict_agg.copy()
        recession_dict.update(**recession_changes)
        recession_all_results = []
        recession_results = dict()
        #  running recession with diferent lengths up to 20q then averaging the result
        for t in range(max_recession_duration):
            recession_dict['EconomyMrkv_init'] = [1]*(t+1)
            this_recession_results = AggDemandEconomy.runExperiment(**recession_dict, Full_Output = False)
            recession_all_results += [this_recession_results]
        for key in output_keys:
            recession_results[key] = np.sum(np.array([recession_all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
        saveAsPickleUnderVarName(recession_results,figs_dir,locals())
        t1 = time()
        print('Calculating recession consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')    
        
           
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
        
    if Run_TaxCut_Recession:
        # Solving tax cut during recession under Agg Multiplier  
        t0 = time()
        AggDemandEconomy.solveAD_Recession_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_TaxCut')
        t1 = time()
        print('Solving payroll tax cut during recession took ' + mystr(t1-t0) + ' seconds.')
        
        # Run the payroll tax cut during recession consumption level in absence of Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'baseline')
        recession_TaxCut_dict = base_dict_agg.copy()
        recession_TaxCut_dict.update(**recession_TaxCut_changes)
        recession_TaxCut_all_results = []
        recession_TaxCut_results = dict()
        # construct history of markov states (considering interaction between lenth of recession and payroll tax cuts)
        for t in range(max_recession_duration):
            if t<7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1
                recession_TaxCut_dict['EconomyMrkv_init'][t+1:8] -= 1
            if t==7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18, -1])+1
            if t>7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.concatenate((np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1, np.array([1]*(t-7))))
            this_recession_results = AggDemandEconomy.runExperiment(**recession_TaxCut_dict, Full_Output = False)
            recession_TaxCut_all_results += [this_recession_results]
        for key in output_keys:
            recession_TaxCut_results[key] = np.sum(np.array([recession_TaxCut_all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
        saveAsPickleUnderVarName(recession_TaxCut_results,figs_dir,locals())
        t1 = time()
        print('Calculating payroll tax cut during recession consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')    
        
        
        # Run the payroll tax cut during recession consumption level in presence of the Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'Recession_TaxCut')
        recession_TaxCut_dict = base_dict_agg.copy()
        recession_TaxCut_dict.update(**recession_TaxCut_changes)
        recession_TaxCut_all_results_AD = []
        recession_TaxCut_results_AD = dict()
        for t in range(max_recession_duration):
            if t<7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1
                recession_TaxCut_dict['EconomyMrkv_init'][t+1:8] -= 1
            if t==7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18, -1])+1
            if t>7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.concatenate((np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1, np.array([1]*(t-7))))
            #print(recession_TaxCut_dict['EconomyMrkv_init'])
            this_recession_results_AD = AggDemandEconomy.runExperiment(**recession_TaxCut_dict, Full_Output = False)
            recession_TaxCut_all_results_AD += [this_recession_results_AD]
        for key in output_keys:
            recession_TaxCut_results_AD[key] = np.sum(np.array([recession_TaxCut_all_results_AD[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
        saveAsPickleUnderVarName(recession_TaxCut_results_AD,figs_dir,locals())
        t1 = time()
        print('Calculating payroll tax cut during recession consumption took ' + mystr(t1-t0) + ' seconds.')
        
        
    #%% Plotting
    if Make_Plots:
        max_T = 20
        x_axis = np.arange(1,21)
        
        base_results                = loadPickle('base_results',figs_dir,locals())
        UI_results                  = loadPickle('UI_results',figs_dir,locals())
        recession_UI_results        = loadPickle('recession_UI_results',figs_dir,locals())
        recession_results           = loadPickle('recession_results',figs_dir,locals())
        TaxCut_results              = loadPickle('TaxCut_results',figs_dir,locals())
        TaxCut_results_AD           = loadPickle('TaxCut_results_AD',figs_dir,locals())
        recession_results_AD        = loadPickle('recession_results_AD',figs_dir,locals())
        recession_TaxCut_results    = loadPickle('recession_TaxCut_results',figs_dir,locals())
        recession_TaxCut_results_AD = loadPickle('recession_TaxCut_results_AD',figs_dir,locals())
     
        
        if Run_UB_Ext:
            AddCons_UI_Ext              = getSimulationPercentDiff(base_results,    UI_results,'AggCons')
            AddInc_UI_Ext               = getSimulationPercentDiff(base_results,    UI_results,'AggIncome')
            NPV_AddInc_UI               = getSimulationDiff(base_results,           UI_results,'NPV_AggIncome') 
            AddInc_UI_Abs               = getSimulationDiff(base_results,           UI_results,'AggIncome') 
            Stimulus_UI                 = getStimulus(base_results,                 UI_results,NPV_AddInc_UI[-1]) 
            Stimulus_UI_perPeriod       = getStimulus(base_results,                 UI_results,AddInc_UI_Abs) 
            Stimulus_UI_perPeriod[6:]   = 0

            plt.figure(figsize=(15,10))
            plt.title('UI extension, no recession', size=30)
            plt.plot(x_axis,AddInc_UI_Ext[0:max_T],     color='blue',linestyle='-')
            #plt.plot(x_axis,AddInc_TaxCut_AD[0:max_T],  color='blue',linestyle='--')
            plt.plot(x_axis,AddCons_UI_Ext[0:max_T],    color='red',linestyle='-')
            #plt.plot(x_axis,AddCons_TaxCut_AD[0:max_T], color='red',linestyle='--')
            #plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=14)
            plt.legend(['Income, no AD effects','Consumption, no AD effects'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% diff. rel. to baseline', fontsize=16)
            plt.savefig(figs_dir +'UI_ext.pdf')
            plt.show()
            
            
            AddCons_UI_Ext_Rec_RelRec       = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggCons')
            AddInc_UI_Ext_Rec_RelRec        = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggIncome')      
            NPV_AddInc_UI_Rec               = getSimulationDiff(recession_results,           recession_UI_results,'NPV_AggIncome')
            AddInc_UI_Rec_Abs               = getSimulationDiff(recession_results,           recession_UI_results,'AggIncome')
            Stimulus_UI_Rec                 = getStimulus(recession_results,                 recession_UI_results,NPV_AddInc_UI_Rec[-1]) 
            Stimulus_UI_Rec_perPeriod       = getStimulus(recession_results,                 recession_UI_results,AddInc_UI_Rec_Abs) 
            Stimulus_UI_Rec_perPeriod[6:]   = 0
            
            plt.figure(figsize=(15,10))
            plt.title('Recession + UI extension', size=30)
            plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec[0:max_T],     color='blue',linestyle='-')
            #plt.plot(x_axis,AddInc_Rec_TaxCut_AD_RelRec[0:max_T],  color='blue',linestyle='--')
            plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec[0:max_T],    color='red',linestyle='-')
            #plt.plot(x_axis,AddCons_Rec_TaxCut_AD_RelRec[0:max_T], color='red',linestyle='--')
            #plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=14)
            plt.legend(['Income, no AD effects','Consumption, no AD effects'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% diff. rel. to recession', fontsize=16)
            plt.savefig(figs_dir +'recession_UI_relrecession.pdf')
            plt.show()  
            
        if Run_TaxCut:   
            AddCons_TaxCut              = getSimulationPercentDiff(base_results,    TaxCut_results,'AggCons')
            AddCons_TaxCut_AD           = getSimulationPercentDiff(base_results,    TaxCut_results_AD,'AggCons')
            AddInc_TaxCut               = getSimulationPercentDiff(base_results,    TaxCut_results,'AggIncome')
            AddInc_TaxCut_AD            = getSimulationPercentDiff(base_results,    TaxCut_results_AD,'AggIncome')
            
            # Value of policy expenditure (need to consider non-AD solution)
            NPV_AddInc_TaxCut           = getSimulationDiff(base_results,TaxCut_results,'NPV_AggIncome')
            AddInc_TaxCut_Abs           = getSimulationDiff(base_results,TaxCut_results,'AggIncome') 
            Stimulus_TaxCut             = getStimulus(base_results,TaxCut_results,NPV_AddInc_TaxCut[-1]) 
            Stimulus_TaxCut_AD          = getStimulus(base_results,TaxCut_results_AD,NPV_AddInc_TaxCut[-1])
            Stimulus_TaxCut_perPeriod   = getStimulus(base_results,TaxCut_results,AddInc_TaxCut_Abs) 
            Stimulus_TaxCut_perPeriod[8:] = 0
            Stimulus_TaxCut_AD_perPeriod= getStimulus(base_results,TaxCut_results_AD,AddInc_TaxCut_Abs)
            Stimulus_TaxCut_perPeriod[8:] = 0
            
            plt.figure(figsize=(15,10))
            plt.title('Tax Cut, no recession', size=30)
            plt.plot(x_axis,AddInc_TaxCut[0:max_T],     color='blue',linestyle='-')
            plt.plot(x_axis,AddInc_TaxCut_AD[0:max_T],  color='blue',linestyle='--')
            plt.plot(x_axis,AddCons_TaxCut[0:max_T],    color='red',linestyle='-')
            plt.plot(x_axis,AddCons_TaxCut_AD[0:max_T], color='red',linestyle='--')
            plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% diff. rel. to baseline', fontsize=16)
            plt.savefig(figs_dir +'tax_cut.pdf')
            plt.show()
                   
        
        if Run_Recession:
            AddCons_Rec                 = getSimulationPercentDiff(base_results,    recession_results,'AggCons')
            AddCons_Rec_AD              = getSimulationPercentDiff(base_results,    recession_results_AD,'AggCons')
            AddInc_Rec                  = getSimulationPercentDiff(base_results,    recession_results,'AggIncome')
            AddInc_Rec_AD               = getSimulationPercentDiff(base_results,    recession_results_AD,'AggIncome')
        
            plt.figure(figsize=(15,10))
            plt.title('Recession', size=30)
            plt.plot(x_axis,AddInc_Rec[0:max_T],     color='blue',linestyle='-')
            plt.plot(x_axis,AddInc_Rec_AD[0:max_T],  color='blue',linestyle='--')
            plt.plot(x_axis,AddCons_Rec[0:max_T],    color='red',linestyle='-')
            plt.plot(x_axis,AddCons_Rec_AD[0:max_T], color='red',linestyle='--')
            plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% diff. rel. to baseline', fontsize=16)
            plt.savefig(figs_dir +'recession.pdf')
            plt.show()
        
    
        if Run_TaxCut_Recession:
            AddCons_Rec_TaxCut         = getSimulationPercentDiff(base_results,     recession_TaxCut_results,'AggCons')
            AddCons_Rec_TaxCut_AD      = getSimulationPercentDiff(base_results,     recession_TaxCut_results_AD,'AggCons')
            AddInc_Rec_TaxCut          = getSimulationPercentDiff(base_results,     recession_TaxCut_results,'AggIncome')
            AddInc_Rec_TaxCut_AD       = getSimulationPercentDiff(base_results,     recession_TaxCut_results_AD,'AggIncome')
            
            plt.figure(figsize=(15,10))
            plt.title('Recession + tax cut', size=30)
            plt.plot(x_axis,AddInc_Rec_TaxCut[0:max_T],     color='blue',linestyle='-')
            plt.plot(x_axis,AddInc_Rec_TaxCut_AD[0:max_T],  color='blue',linestyle='--')
            plt.plot(x_axis,AddCons_Rec_TaxCut[0:max_T],    color='red',linestyle='-')
            plt.plot(x_axis,AddCons_Rec_TaxCut_AD[0:max_T], color='red',linestyle='--')
            plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% diff. rel. to baseline', fontsize=16)
            plt.savefig(figs_dir +'recession_taxcut_relbaseline.pdf')
            plt.show()
            
    
        
        if Run_TaxCut_Recession and Run_Recession:
            AddCons_Rec_TaxCut_RelRec         = getSimulationPercentDiff(recession_results,    recession_TaxCut_results,'AggCons')
            AddCons_Rec_TaxCut_AD_RelRec      = getSimulationPercentDiff(recession_results_AD, recession_TaxCut_results_AD,'AggCons')
            AddInc_Rec_TaxCut_RelRec          = getSimulationPercentDiff(recession_results,    recession_TaxCut_results,'AggIncome')
            AddInc_Rec_TaxCut_AD_RelRec       = getSimulationPercentDiff(recession_results_AD, recession_TaxCut_results_AD,'AggIncome')
        
            # Value of policy expenditure (need to consider non-AD solution)
            NPV_AddInc_Rec_TaxCut       = getSimulationDiff(recession_results,recession_TaxCut_results,'NPV_AggIncome') 
            AddInc_Rec_TaxCu_Abs        = getSimulationDiff(recession_results,recession_TaxCut_results,'AggIncome') 
            Stimulus_Rec_TaxCut         = getStimulus(recession_results,recession_TaxCut_results,NPV_AddInc_Rec_TaxCut[-1]) 
            Stimulus_Rec_TaxCut_AD      = getStimulus(recession_results_AD,recession_TaxCut_results_AD,NPV_AddInc_Rec_TaxCut[-1])
            Stimulus_Rec_TaxCut_perPeriod         = getStimulus(recession_results,recession_TaxCut_results,AddInc_Rec_TaxCu_Abs) 
            Stimulus_Rec_TaxCut_perPeriod[8:]     = 0
            Stimulus_Rec_TaxCut_AD_perPeriod      = getStimulus(recession_results_AD,recession_TaxCut_results_AD,AddInc_Rec_TaxCu_Abs)
            Stimulus_Rec_TaxCut_AD_perPeriod[8:]     = 0
        
            plt.figure(figsize=(15,10))
            plt.title('Recession + tax cut', size=30)
            plt.plot(x_axis,AddInc_Rec_TaxCut_RelRec[0:max_T],     color='blue',linestyle='-')
            plt.plot(x_axis,AddInc_Rec_TaxCut_AD_RelRec[0:max_T],  color='blue',linestyle='--')
            plt.plot(x_axis,AddCons_Rec_TaxCut_RelRec[0:max_T],    color='red',linestyle='-')
            plt.plot(x_axis,AddCons_Rec_TaxCut_AD_RelRec[0:max_T], color='red',linestyle='--')
            plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% diff. rel. to recession', fontsize=16)
            plt.savefig(figs_dir +'recession_taxcut_relrecession.pdf')
            plt.show()   
            

        Plot_Stimulus = True
        if Plot_Stimulus:
            

            # stimulus effects without AD, NPV as base
            plt.figure(figsize=(15,10))
            plt.title('Stimu. cons. per period relative to NPV of policy, no AD effects', size=30)
            plt.plot(x_axis,Stimulus_UI[0:max_T],           color='blue',linestyle='-')
            plt.plot(x_axis,Stimulus_UI_Rec[0:max_T],       color='blue',linestyle='--')
            plt.plot(x_axis,Stimulus_TaxCut[0:max_T],       color='red',linestyle='-')
            plt.plot(x_axis,Stimulus_Rec_TaxCut[0:max_T],   color='red',linestyle='--')
            plt.legend(['UI extension','UI extension in rec','Tax cut','Tax cut in rec'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% of policy NPV expended', fontsize=16)
            plt.savefig(figs_dir +'NPV_Multiplier_no_AD.pdf')
            plt.show()

            # stimulus effects without AD, period income as base
            plt.figure(figsize=(15,10))
            plt.title('Stimu. cons. per period relative to periods policy exp., no AD effects', size=30)
            plt.plot(x_axis,Stimulus_UI_perPeriod[0:max_T],           color='blue',linestyle='-')
            plt.plot(x_axis,Stimulus_UI_Rec_perPeriod[0:max_T],       color='blue',linestyle='--')
            plt.plot(x_axis,Stimulus_TaxCut_perPeriod[0:max_T],       color='red',linestyle='-')
            plt.plot(x_axis,Stimulus_Rec_TaxCut_perPeriod[0:max_T],   color='red',linestyle='--')
            plt.legend(['UI extension','UI extension in rec','Tax cut','Tax cut in rec'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% of periods policy expenditure expended', fontsize=16)
            plt.savefig(figs_dir +'Period_Multiplier_no_AD.pdf')
            plt.show()
            
            # stimulus effects with AD
            plt.figure(figsize=(15,10))
            plt.title('Stimu. cons. per period relative to NPV of policy, AD effects', size=30)
            plt.plot(x_axis,Stimulus_TaxCut_AD[0:max_T], color='blue',linestyle='-')
            plt.plot(x_axis,Stimulus_Rec_TaxCut_AD[0:max_T], color='blue',linestyle='--')
            plt.legend(['Tax cut, no recession, AD effects','Tax cut, recession, AD effects'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% of policy NPV expended', fontsize=16)
            plt.savefig(figs_dir +'NPV_Multiplier_TaxCut_AD.pdf')
            plt.show()
        
   
    #%% testing
    if Show_TestPlots:
        max_T = 20
        plt.figure(figsize=(15,10))
        plt.plot(recession_TaxCut_results_AD['AggIncome'][0:max_T]-recession_results_AD['AggIncome'][0:max_T])
        plt.plot(recession_TaxCut_all_results_AD[0]['AggIncome'][0:max_T]-recession_all_results_AD[0]['AggIncome'][0:max_T])
        plt.plot(recession_TaxCut_all_results_AD[4]['AggIncome'][0:max_T]-recession_all_results_AD[4]['AggIncome'][0:max_T])
        plt.plot(recession_TaxCut_all_results_AD[8]['AggIncome'][0:max_T]-recession_all_results_AD[8]['AggIncome'][0:max_T])
        plt.plot(recession_TaxCut_all_results_AD[12]['AggIncome'][0:max_T]-recession_all_results_AD[12]['AggIncome'][0:max_T])
        plt.plot(recession_TaxCut_all_results_AD[16]['AggIncome'][0:max_T]-recession_all_results_AD[16]['AggIncome'][0:max_T])
        plt.legend(['Weighted','0','4','8','12','16'], fontsize=20)
        plt.show()

   
        x = (recession_all_results_AD[-1]['Cratio_hist'][1:19]-1) #should this be really form 0 to 19, not from 1 to 19?
        y = recession_all_results_AD[-1]['Cratio_hist'][2:20]
        s, i = np.polyfit(x, y, 1)
        
        startt = 2
        max_recession = 19
        slope_if_recession     = (recession_all_results_AD[-1]['Cratio_hist'][startt+1] - recession_all_results_AD[-1]['Cratio_hist'][max_recession-1])/(recession_all_results_AD[-1]['Cratio_hist'][startt] - recession_all_results_AD[-1]['Cratio_hist'][max_recession-2])
        intercept_if_recession =  recession_all_results_AD[-1]['Cratio_hist'][startt+1] - slope_if_recession*(recession_all_results_AD[-1]['Cratio_hist'][startt]-1)
               
        
        plt.figure(figsize=(15,10))
        plt.plot(x, y, 'o')
        plt.plot(x, s*x + i)
        plt.plot(x, slope_if_recession*x + intercept_if_recession)
        plt.legend(['points','best line fit','old'])
        plt.show()
    
    
    
    t_end = time()
    print('Doing everything took ' + mystr(t_end-t_start) + ' seconds in total.')
   
    
    
    
    
    




