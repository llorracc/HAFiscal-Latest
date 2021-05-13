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
Run_UB_Ext_Recession    = True
Run_Recession           = True
Run_TaxCut_Recession    = False
Run_NonAD               = True #whether to run nonAD experiments as well
Make_Plots              = False


#%% 

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
        
    #%% 
    
    if Run_UB_Ext_Recession:
        
        print('Solving and simulating unemployment extension experiment')
        
        # Solving tax cut under Agg Multiplier  
        t0 = time()
        AggDemandEconomy.solveAD_UIExtension_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'UI_Rec')
        t1 = time()
        print('Solving UI during recession took ' + mystr(t1-t0) + ' seconds.')
        
        if Run_NonAD:
            # Run UI ext during recession with no AD effects
            t0 = time()
            AggDemandEconomy.restoreADsolution(name = 'baseline')
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
            saveAsPickleUnderVarName(recession_UI_all_results,figs_dir,locals())
            saveAsPickleUnderVarName(recession_UI_results,figs_dir,locals())
            t1 = time()
            print('Calculating recession and extended UI consumption  (no AD) ' + mystr(t1-t0) + ' seconds.')
            
            
        # Run UI ext during recession with AD effects
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'UI_Rec')
        recession_UI_dict = base_dict_agg.copy()
        recession_UI_dict.update(**recession_UI_changes)
        recession_UI_all_results_AD = []
        recession_UI_results_AD = dict()
        for t_R in range(max_recession_duration):
            for t_Policy in range(max_policy_duration):
                recession_UI_dict['EconomyMrkv_init'] = np.array([0]*max(max_recession_duration,max_policy_duration))
                recession_UI_dict['EconomyMrkv_init'][0:t_R+1] += 1
                recession_UI_dict['EconomyMrkv_init'][0:t_Policy+1] +=2
                print('Running Experiment with Mrkv history: ', recession_UI_dict['EconomyMrkv_init'])
                this_recession_UI_results = AggDemandEconomy.runExperiment(**recession_UI_dict, Full_Output = False)
                recession_UI_all_results_AD += [this_recession_UI_results]
        for key in output_keys:
            recession_UI_results_AD[key] = np.zeros_like(recession_UI_all_results_AD[0][key])
            count = 0
            for t_R in range(max_recession_duration):
                for t_Policy in range(max_policy_duration):
                    recession_UI_results_AD[key] += recession_UI_all_results_AD[count][key]*recession_prob_array[t_R]*policy_prob_array[t_Policy]
                    count += 1
        saveAsPickleUnderVarName(recession_UI_all_results_AD,figs_dir,locals())
        saveAsPickleUnderVarName(recession_UI_results_AD,figs_dir,locals())
        t1 = time()
        print('Calculating recession and extended UI consumption took (with AD) ' + mystr(t1-t0) + ' seconds.')
    
    #%% 
    
    if Run_Recession:
        # Solving recession under Agg Multiplier   
        t0 = time()
        AggDemandEconomy.solveAD_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession')
        t1 = time()
        print('Solving recession took ' + mystr(t1-t0) + ' seconds.')
        
        if Run_NonAD:
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
            saveAsPickleUnderVarName(recession_all_results,figs_dir,locals())
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
        saveAsPickleUnderVarName(recession_all_results_AD,figs_dir,locals())
        saveAsPickleUnderVarName(recession_results_AD,figs_dir,locals())
        t1 = time()
        print('Calculating recession consumption took ' + mystr(t1-t0) + ' seconds.')
    


    #%% 
    
        
    if Run_TaxCut_Recession:
        # Solving tax cut during recession under Agg Multiplier  
        t0 = time()
        AggDemandEconomy.solveAD_Recession_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_TaxCut')
        t1 = time()
        print('Solving payroll tax cut during recession took ' + mystr(t1-t0) + ' seconds.')
        
        if Run_NonAD:
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
            saveAsPickleUnderVarName(recession_TaxCut_all_results,figs_dir,locals())
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
        saveAsPickleUnderVarName(recession_TaxCut_all_results_AD,figs_dir,locals())
        saveAsPickleUnderVarName(recession_TaxCut_results_AD,figs_dir,locals())
        t1 = time()
        print('Calculating payroll tax cut during recession consumption took ' + mystr(t1-t0) + ' seconds.')
        
        
    #%% Plotting
        

        
    if Make_Plots:
        
        max_T = 20
        x_axis = np.arange(1,21)
        
        #folder1 = './Figures/FullRun_Apr18_AD05_800k/'
        folder2 = './Figures/FullRun_Apr18_AD05_AllStates_800k/'
        folder1 = figs_dir
        
        base_results                = loadPickle('base_results',folder1,locals())

        recession_results               = loadPickle('recession_results',folder1,locals())
        recession_results_AD            = loadPickle('recession_results_AD',folder1,locals())
        recession_results_AD_allStates  = loadPickle('recession_results_AD',folder2,locals())
        
        recession_UI_results                = loadPickle('recession_UI_results',folder1,locals())       
        recession_UI_results_AD             = loadPickle('recession_UI_results_AD',folder1,locals())
        recession_UI_results_AD_allStates   = loadPickle('recession_UI_results_AD',folder2,locals())
        
        recession_TaxCut_results                = loadPickle('recession_TaxCut_results',folder1,locals())
        recession_TaxCut_results_AD             = loadPickle('recession_TaxCut_results_AD',folder1,locals())
        recession_TaxCut_results_AD_allStates   = loadPickle('recession_TaxCut_results_AD',folder2,locals())
        
        
        
       
        
        #%% Multipliers
        
        NPV_AddInc_UI_Rec                       = getSimulationDiff(recession_results,recession_UI_results,'NPV_AggIncome') # Policy expenditure
        NPV_Multiplier_UI_Rec                   = getNPVMultiplier(recession_results,               recession_UI_results,               NPV_AddInc_UI_Rec)
        NPV_Multiplier_UI_Rec_AD                = getNPVMultiplier(recession_results_AD,            recession_UI_results_AD,            NPV_AddInc_UI_Rec)
        NPV_Multiplier_UI_Rec_AD_allStates      = getNPVMultiplier(recession_results_AD_allStates,  recession_UI_results_AD_allStates,  NPV_AddInc_UI_Rec)
        
        
        NPV_AddInc_Rec_TaxCut                   = getSimulationDiff(recession_results,recession_TaxCut_results,'NPV_AggIncome') 
        NPV_Multiplier_Rec_TaxCut               = getNPVMultiplier(recession_results,               recession_TaxCut_results,               NPV_AddInc_Rec_TaxCut)
        NPV_Multiplier_Rec_TaxCut_AD            = getNPVMultiplier(recession_results_AD,            recession_TaxCut_results_AD,            NPV_AddInc_Rec_TaxCut)
        NPV_Multiplier_Rec_TaxCut_AD_allStates  = getNPVMultiplier(recession_results_AD_allStates,  recession_TaxCut_results_AD_allStates,  NPV_AddInc_Rec_TaxCut)
        
        print('NPV_Multiplier_UI_Rec: ',mystr(NPV_Multiplier_UI_Rec[-1]))
        print('NPV_Multiplier_UI_Rec_AD: ',mystr(NPV_Multiplier_UI_Rec_AD[-1]))
        print('NPV_Multiplier_UI_Rec_AD_allStates: ',mystr(NPV_Multiplier_UI_Rec_AD_allStates[-1]))
        
        print('NPV_Multiplier_Rec_TaxCut: ',mystr(NPV_Multiplier_Rec_TaxCut[-1]))
        print('NPV_Multiplier_Rec_TaxCut_AD: ',mystr(NPV_Multiplier_Rec_TaxCut_AD[-1]))
        print('NPV_Multiplier_Rec_TaxCut_AD_allStates: ',mystr(NPV_Multiplier_Rec_TaxCut_AD_allStates[-1]))
        
        #%% Income and Consumption paths UI extension
    
        AddCons_UI_Ext_Rec_RelRec               = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggCons')
        AddInc_UI_Ext_Rec_RelRec                = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggIncome')
        
        AddCons_UI_Ext_Rec_RelRec_AD            = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggCons')
        AddInc_UI_Ext_Rec_RelRec_AD             = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggIncome')
        
        AddCons_UI_Ext_Rec_RelRec_AD_allStates  = getSimulationPercentDiff(recession_results_AD_allStates,    recession_UI_results_AD_allStates,'AggCons')
        AddInc_UI_Ext_Rec_RelRec_AD_allStates   = getSimulationPercentDiff(recession_results_AD_allStates,    recession_UI_results_AD_allStates,'AggIncome')

        
        plt.figure(figsize=(15,10))
        plt.title('Recession + UI extension', size=30)
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec[0:max_T],              color='blue',linestyle='-')
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_AD[0:max_T],           color='blue',linestyle='--')
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_AD_allStates[0:max_T], color='blue',linestyle=':')
        plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec[0:max_T],             color='red',linestyle='-')
        plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_AD[0:max_T],          color='red',linestyle='--')
        plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_AD_allStates[0:max_T],color='red',linestyle=':')  
        plt.legend(['Inc, no AD effects','Inc, AD effects','Inc, AD effects all States',\
                    'Cons, no AD effects','Cons, AD effects','Cons, AD effects all States'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'recession_UI_relrecession.pdf')
        plt.show() 
        
        #%% Income and Consumption paths Tax cut        


        AddCons_Rec_TaxCut_RelRec               = getSimulationPercentDiff(recession_results,               recession_TaxCut_results,'AggCons')
        AddCons_Rec_TaxCut_AD_RelRec            = getSimulationPercentDiff(recession_results_AD,            recession_TaxCut_results_AD,'AggCons')
        AddCons_Rec_TaxCut_AD_RelRec_allStates  = getSimulationPercentDiff(recession_results_AD_allStates,  recession_TaxCut_results_AD_allStates,'AggCons')
        
        AddInc_Rec_TaxCut_RelRec                = getSimulationPercentDiff(recession_results,               recession_TaxCut_results,'AggIncome')
        AddInc_Rec_TaxCut_AD_RelRec             = getSimulationPercentDiff(recession_results_AD,            recession_TaxCut_results_AD,'AggIncome')
        AddInc_Rec_TaxCut_AD_RelRec_allStates   = getSimulationPercentDiff(recession_results_AD_allStates,  recession_TaxCut_results_AD_allStates,'AggIncome')

    
        plt.figure(figsize=(15,10))
        plt.title('Recession + tax cut', size=30)
        plt.plot(x_axis,AddInc_Rec_TaxCut_RelRec[0:max_T],              color='blue',linestyle='-')
        plt.plot(x_axis,AddInc_Rec_TaxCut_AD_RelRec[0:max_T],           color='blue',linestyle='--')
        plt.plot(x_axis,AddInc_Rec_TaxCut_AD_RelRec_allStates[0:max_T], color='blue',linestyle=':')
        plt.plot(x_axis,AddCons_Rec_TaxCut_RelRec[0:max_T],             color='red',linestyle='-')
        plt.plot(x_axis,AddCons_Rec_TaxCut_AD_RelRec[0:max_T],          color='red',linestyle='--')
        plt.plot(x_axis,AddCons_Rec_TaxCut_AD_RelRec_allStates[0:max_T],color='red',linestyle=':')
        plt.legend(['Inc, no AD effects','Inc, AD effects','Inc, AD effects all States',\
                    'Cons, no AD effects','Cons, AD effects','Cons, AD effects all States'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'recession_taxcut_relrecession.pdf')
        plt.show()   
        
        
        
        #%% The following sections look at all results, i.e. not just the weighted sum across all recessions but each single simulation
        
        recession_all_results               = loadPickle('recession_all_results',folder1,locals())
        recession_all_results_AD            = loadPickle('recession_all_results_AD',folder1,locals())
        recession_all_results_AD_allStates  = loadPickle('recession_all_results_AD',folder2,locals())
        
        recession_all_UI_results                = loadPickle('recession_UI_all_results',folder1,locals())       
        recession_all_UI_results_AD             = loadPickle('recession_UI_all_results_AD',folder1,locals())
        recession_all_UI_results_AD_allStates   = loadPickle('recession_UI_all_results_AD',folder2,locals())
    
    
        #%% Function that returns information on a UI experiment with specific RecLength and PolicyLength
        def PlotsforSpecificRecandPolicyLength(RecLength,PolicyLength): 
            NPV_AddInc_UI_Rec                       = getSimulationDiff(recession_all_results[RecLength-1],recession_all_UI_results[(RecLength-1)*6+(PolicyLength-1)],'NPV_AggIncome') # Policy expenditure
            NPV_Multiplier_UI_Rec                   = getNPVMultiplier(recession_all_results[RecLength-1],               recession_all_UI_results[(RecLength-1)*6+(PolicyLength-1)],               NPV_AddInc_UI_Rec)
            NPV_Multiplier_UI_Rec_AD                = getNPVMultiplier(recession_all_results_AD[RecLength-1],            recession_all_UI_results_AD[(RecLength-1)*6+(PolicyLength-1)],            NPV_AddInc_UI_Rec)
            NPV_Multiplier_UI_Rec_AD_allStates      = getNPVMultiplier(recession_all_results_AD_allStates[RecLength-1],  recession_all_UI_results_AD_allStates[(RecLength-1)*6+(PolicyLength-1)],  NPV_AddInc_UI_Rec)
            
            # When does multipiler rise above 2? very early!
            plt.figure(figsize=(15,10))
            plt.plot(x_axis,NPV_Multiplier_UI_Rec_AD_allStates[0:max_T])
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.show() 

            
            Multipliers = [NPV_Multiplier_UI_Rec[-1],NPV_Multiplier_UI_Rec_AD[-1],NPV_Multiplier_UI_Rec_AD_allStates[-1]]
            
            PlotEach = True
            
            if PlotEach:
            
                AddCons_UI_Ext_Rec_RelRec               = getSimulationPercentDiff(recession_all_results[RecLength-1],    recession_all_UI_results[(RecLength-1)*6+(PolicyLength-1)],'AggCons')
                AddInc_UI_Ext_Rec_RelRec                = getSimulationPercentDiff(recession_all_results[RecLength-1],    recession_all_UI_results[(RecLength-1)*6+(PolicyLength-1)],'AggIncome')
                
                AddCons_UI_Ext_Rec_RelRec_AD            = getSimulationPercentDiff(recession_all_results_AD[RecLength-1],    recession_all_UI_results_AD[(RecLength-1)*6+(PolicyLength-1)],'AggCons')
                AddInc_UI_Ext_Rec_RelRec_AD             = getSimulationPercentDiff(recession_all_results_AD[RecLength-1],    recession_all_UI_results_AD[(RecLength-1)*6+(PolicyLength-1)],'AggIncome')
                
                AddCons_UI_Ext_Rec_RelRec_AD_allStates  = getSimulationPercentDiff(recession_all_results_AD_allStates[RecLength-1],    recession_all_UI_results_AD_allStates[(RecLength-1)*6+(PolicyLength-1)],'AggCons')
                AddInc_UI_Ext_Rec_RelRec_AD_allStates   = getSimulationPercentDiff(recession_all_results_AD_allStates[RecLength-1],    recession_all_UI_results_AD_allStates[(RecLength-1)*6+(PolicyLength-1)],'AggIncome')
                
                plt.figure(figsize=(15,10))
                plt.title('Recession lasts ' + str(RecLength) + 'q, UI extension lasts ' + str(PolicyLength) + 'q', size=30)
                plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec[0:max_T],              color='blue',linestyle='-')
                plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_AD[0:max_T],           color='blue',linestyle='--')
                plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_AD_allStates[0:max_T], color='blue',linestyle=':')
                plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec[0:max_T],             color='red',linestyle='-')
                plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_AD[0:max_T],          color='red',linestyle='--')
                plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_AD_allStates[0:max_T],color='red',linestyle=':')  
                plt.legend(['Inc, no AD effects','Inc, AD effects','Inc, AD effects all States',\
                            'Cons, no AD effects','Cons, AD effects','Cons, AD effects all States'], fontsize=14)
                plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
                plt.xlabel('quarter', fontsize=18)
                plt.ylabel('% diff. rel. to no UI extension', fontsize=16)
                plt.savefig(figs_dir +'Rec' + str(RecLength) +'q_UiExt' + str(PolicyLength) + 'q_relrecession.pdf')
                plt.show() 
                
                
                # AddCons_UI_Ext_Rec_RelBase_AD_allStates = getSimulationPercentDiff(base_results,    recession_all_UI_results_AD_allStates[(RecLength-1)*6+(PolicyLength-1)],'AggCons')         
                # AddCons_Rec_RelBase_AD_allStates        = getSimulationPercentDiff(base_results,    recession_all_results_AD_allStates[RecLength-1],'AggCons')
                # AddInc_UI_Ext_Rec_RelBase_AD_allStates  = getSimulationPercentDiff(base_results,    recession_all_UI_results_AD_allStates[(RecLength-1)*6+(PolicyLength-1)],'AggIncome')
                # AddInc_Rec_RelBase_AD_allStates         = getSimulationPercentDiff(base_results,    recession_all_results_AD_allStates[RecLength-1],'AggIncome')
                
                # plt.figure(figsize=(15,10))
                # plt.title('Recession lasts ' + str(RecLength) + 'q, UI extension lasts ' + str(PolicyLength) + 'q', size=30)
                # plt.plot(x_axis,AddInc_Rec_RelBase_AD_allStates[0:max_T]        , color='blue',linestyle='-')
                # plt.plot(x_axis,AddCons_Rec_RelBase_AD_allStates[0:max_T]       , color='red' ,linestyle='-')  
                # plt.plot(x_axis,AddInc_UI_Ext_Rec_RelBase_AD_allStates[0:max_T] , color='blue',linestyle=':')             
                # plt.plot(x_axis,AddCons_UI_Ext_Rec_RelBase_AD_allStates[0:max_T], color='red' ,linestyle=':') 
                # plt.legend(['Inc, Rec','Cons, Rec',\
                #             'Inc, Rec + UI', 'Cons, Rec + UI'], fontsize=14)
                # plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
                # plt.xlabel('quarter', fontsize=18)
                # plt.ylabel('% diff. rel. to baseline', fontsize=16)
                # plt.show() 
            
            return Multipliers
        
        #%% Plotting long-run multiplier as a function of recession and policy duration
        max_recession_duration = 21
        max_policy_duration = 6
        Multipliers = np.zeros((max_recession_duration+1,max_policy_duration+1,3))
        for RecLength in range(1,max_recession_duration+1,1):
            for PolicyLength in range(1,max_policy_duration+1,1):
                Multipliers[RecLength][PolicyLength][0:3] = PlotsforSpecificRecandPolicyLength(RecLength,PolicyLength)

              
        plt.figure(figsize=(15,10))
        plt.title('Multipliers as function of Recession length', size=30)
        plt.plot(x_axis,Multipliers[1:21,1,0], color='black',)
        plt.plot(x_axis,Multipliers[1:21,1,1], color='blue',linestyle='-')
        plt.plot(x_axis,Multipliers[1:21,2,1], color='orange',linestyle='-')
        plt.plot(x_axis,Multipliers[1:21,4,1], color='red',linestyle='-')
        plt.plot(x_axis,Multipliers[1:21,6,1], color='green',linestyle='-')
        
        plt.plot(x_axis,Multipliers[1:21,1,2], color='blue',linestyle='--')
        plt.plot(x_axis,Multipliers[1:21,2,2], color='orange',linestyle='--')
        plt.plot(x_axis,Multipliers[1:21,4,2], color='red',linestyle='--')
        plt.plot(x_axis,Multipliers[1:21,6,2], color='green',linestyle='--') 
        plt.legend(['no AD effects',\
                    'AD effects Rec states, UI ext 1q',\
                    'AD effects Rec states, UI ext 2q',\
                    'AD effects Rec states, UI ext 4q',\
                    'AD effects Rec states, UI ext 6q',\
                    'AD effects all states, UI ext 1q',\
                    'AD effects all states, UI ext 2q',\
                    'AD effects all states, UI ext 4q',\
                    'AD effects all states, UI ext 6q'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('recession lasts quarter', fontsize=18)
        plt.ylabel('Long-run NPV multiplier', fontsize=16)
        plt.savefig(figs_dir +'Multipliers_RecLength_PolicyLength2.pdf')
        plt.show() 
            

          
