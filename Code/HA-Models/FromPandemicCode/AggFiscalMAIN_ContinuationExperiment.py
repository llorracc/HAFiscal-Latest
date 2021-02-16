'''
This is the main script for the paper
'''
#$$$$$$$$$$ represents places in the code that need to be adjusted when the markov state space is changed
from Parameters import init_infhorizon, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
     figs_dir, num_max_iterations_solvingAD, convergence_tol_solvingAD
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from OtherFunctions import loadPickle, saveAsPickleUnderVarName, namestr, getSimulationDiff, getSimulationPercentDiff
mystr = lambda x : '{:.2f}'.format(x)


Run_Recession           = False
Solve_Rec_Taxcut_AD     = True
Run_TaxCut              = True
Run_TaxCut_AD           = True
Run_Continuation        = True
RunPlots                = False


if __name__ == '__main__':
    
    t_start = time()
    
    # Setting up AggDemandEconmy
    from setupEconomy import AggDemandEconomy, base_dict_agg, max_recession_duration, output_keys, recession_prob_array, recession_Cond9q_prob_array
    
    if init_infhorizon['TaxCutContinuationProb_Rec'] == 0.5:
        if init_infhorizon['TaxCutContinuationProb_Bas'] == 0.5:
            save_dir = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_050_Bas_and_Rec/'
        else:
            save_dir = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_050/'    
    elif init_infhorizon['TaxCutContinuationProb_Rec'] == 1:
        save_dir = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_1/'
    elif init_infhorizon['TaxCutContinuationProb_Rec'] == 0:
        save_dir = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_0/'
    print('save_dir',save_dir)
    
      
    # Run the baseline consumption level
    t0 = time()
    base_results = AggDemandEconomy.runExperiment(**base_dict_agg)
    AggDemandEconomy.storeBaseline(base_results['AggCons'])
    t1 = time()
    print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
    
    
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
        recession_Cond9q_results = dict()
        #  running recession with diferent lengths up to 20q then averaging the result
        for t in range(max_recession_duration):
            recession_dict['EconomyMrkv_init'] = [1]*(t+1)
            this_recession_results = AggDemandEconomy.runExperiment(**recession_dict)
            recession_all_results += [this_recession_results]
        for key in output_keys:
            recession_results[key]         = np.sum(np.array([recession_all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
            recession_Cond9q_results[key]  = np.sum(np.array([recession_all_results[t][key]*recession_Cond9q_prob_array[t-8]  for t in range(8,max_recession_duration,1)]), axis=0)
        saveAsPickleUnderVarName(recession_results,save_dir,locals()) 
        saveAsPickleUnderVarName(recession_Cond9q_results,save_dir,locals()) 
        t1 = time()
        print('Calculating recession consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')    
        
          
        # Run the recession consumption level in presence of the Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'Recession')
        recession_dict = base_dict_agg.copy()
        recession_dict.update(**recession_changes)
        recession_all_results_AD = []
        recession_results_AD = dict()
        recession_Cond9q_results_AD = dict()
        for t in range(max_recession_duration):
            recession_dict['EconomyMrkv_init'] = [1]*(t+1)
            this_recession_results_AD = AggDemandEconomy.runExperiment(**recession_dict)
            recession_all_results_AD += [this_recession_results_AD]
        for key in output_keys:
            recession_results_AD[key]           = np.sum(np.array([recession_all_results_AD[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
            recession_Cond9q_results_AD[key]    = np.sum(np.array([recession_all_results_AD[t][key]*recession_Cond9q_prob_array[t-8]  for t in range(8,max_recession_duration,1)]), axis=0)
        saveAsPickleUnderVarName(recession_results_AD,save_dir,locals()) 
        saveAsPickleUnderVarName(recession_Cond9q_results_AD,save_dir,locals())                
        t1 = time()
        print('Calculating recession consumption took ' + mystr(t1-t0) + ' seconds.')

    
    #%% Solving tax cut during recession under Agg Multiplier  
    if Solve_Rec_Taxcut_AD:
        t0 = time()
        AggDemandEconomy.solveAD_Recession_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_TaxCut')
        t1 = time()
        print('Solving payroll tax cut during recession took ' + mystr(t1-t0) + ' seconds.')
    
    #%% Running the payroll tax cut experiments during a recession
    
    if Run_TaxCut:
        
        # Run the payroll tax cut during recession consumption level in absence of Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'baseline')
        recession_TaxCut_dict = base_dict_agg.copy()
        recession_TaxCut_dict.update(**recession_TaxCut_changes)
        recession_TaxCut_all_results = []
        recession_TaxCut_Cond9q_results = dict()
        # construct history of markov states (considering interaction between lenth of recession and payroll tax cuts)
        for t in range(8,max_recession_duration,1):
            if t<7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1
                recession_TaxCut_dict['EconomyMrkv_init'][t+1:8] -= 1
            if t==7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18, -1])+1
            if t>7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.concatenate((np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1, np.array([1]*(t-7))))
            this_recession_results = AggDemandEconomy.runExperiment(**recession_TaxCut_dict)
            recession_TaxCut_all_results += [this_recession_results]
        for key in output_keys:
            recession_TaxCut_Cond9q_results[key] = np.sum(np.array([recession_TaxCut_all_results[t-8][key]*recession_Cond9q_prob_array[t-8]  for t in range(8,max_recession_duration,1)]), axis=0)
        saveAsPickleUnderVarName(recession_TaxCut_Cond9q_results,save_dir,locals()) 
        t1 = time()
        print('Calculating payroll tax cut during recession consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')    
        
        
        if Run_Continuation:
        # Run the payroll tax cut during recession consumption level in absence of Agg Multiplier, Once extended
            t0 = time()
            AggDemandEconomy.restoreADsolution(name = 'baseline')
            recession_TaxCut_dict_OnceExtended = base_dict_agg.copy()
            recession_TaxCut_dict_OnceExtended.update(**recession_TaxCut_changes)
            recession_TaxCut_OnceExtended_all_results = []
            recession_TaxCut_OnceExtended_Cond9q_results = dict()
            # construct history of markov states (considering interaction between lenth of recession and payroll tax cuts)
            for t in range(8,max_recession_duration,1):
                if t<15:
                    recession_TaxCut_dict_OnceExtended['EconomyMrkv_init'] = np.array([ 5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])
                    recession_TaxCut_dict_OnceExtended['EconomyMrkv_init'][t+1:16] -= 1
                elif t==15:
                    recession_TaxCut_dict_OnceExtended['EconomyMrkv_init'] = np.array([ 5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])
                else:
                    recession_TaxCut_dict_OnceExtended['EconomyMrkv_init'] = np.concatenate((np.array([ 5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]), np.array([1]*(t-15))))
                print(recession_TaxCut_dict_OnceExtended['EconomyMrkv_init'])
                this_recession_results = AggDemandEconomy.runExperiment(**recession_TaxCut_dict_OnceExtended)
                recession_TaxCut_OnceExtended_all_results += [this_recession_results]
            for key in output_keys:
                recession_TaxCut_OnceExtended_Cond9q_results[key] = np.sum(np.array([recession_TaxCut_OnceExtended_all_results[t-8][key]*recession_Cond9q_prob_array[t-8]  for t in range(8,max_recession_duration,1)]), axis=0)      
            saveAsPickleUnderVarName(recession_TaxCut_OnceExtended_Cond9q_results,save_dir,locals()) 
            t1 = time()
            print('Calculating payroll tax cut during recession consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.') 
        
        
    
#%%   
    
    if Run_TaxCut_AD:
        
        # Run the payroll tax cut during recession consumption level in presence of the Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'Recession_TaxCut')
        recession_TaxCut_dict = base_dict_agg.copy()
        recession_TaxCut_dict.update(**recession_TaxCut_changes)
        recession_TaxCut_all_results_AD = []
        recession_TaxCut_Cond9q_results_AD = dict()
        for t in range(8,max_recession_duration,1):
            if t<7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1
                recession_TaxCut_dict['EconomyMrkv_init'][t+1:8] -= 1
            if t==7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18, -1])+1
            if t>7:
                recession_TaxCut_dict['EconomyMrkv_init'] = np.concatenate((np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1, np.array([1]*(t-7))))
            print(recession_TaxCut_dict['EconomyMrkv_init'])
            this_recession_results_AD = AggDemandEconomy.runExperiment(**recession_TaxCut_dict)
            recession_TaxCut_all_results_AD += [this_recession_results_AD]
        for key in output_keys:
            recession_TaxCut_Cond9q_results_AD[key] = np.sum(np.array([recession_TaxCut_all_results_AD[t-8][key]*recession_Cond9q_prob_array[t-8]  for t in range(8,max_recession_duration,1)]), axis=0)
        saveAsPickleUnderVarName(recession_TaxCut_Cond9q_results_AD,save_dir,locals()) 
        t1 = time()
        print('Calculating payroll tax cut during recession consumption took ' + mystr(t1-t0) + ' seconds.')
        
    
        if Run_Continuation:
            # Run the payroll tax cut during recession consumption level in presence of Agg Multiplier, Once Extended
            t0 = time()
            AggDemandEconomy.restoreADsolution(name = 'Recession_TaxCut')
            recession_TaxCut_dict_OnceExtended_AD = base_dict_agg.copy()
            recession_TaxCut_dict_OnceExtended_AD.update(**recession_TaxCut_changes)
            recession_TaxCut_OnceExtended_all_results_AD = []
            recession_TaxCut_OnceExtended_Cond9q_results_AD = dict()
            # construct history of markov states (considering interaction between lenth of recession and payroll tax cuts)
            for t in range(8,max_recession_duration,1):
                if t<15:
                    recession_TaxCut_dict_OnceExtended_AD['EconomyMrkv_init'] = np.array([ 5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])
                    recession_TaxCut_dict_OnceExtended_AD['EconomyMrkv_init'][t+1:16] -= 1
                elif t==15:
                    recession_TaxCut_dict_OnceExtended_AD['EconomyMrkv_init'] = np.array([ 5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])
                else:
                    recession_TaxCut_dict_OnceExtended_AD['EconomyMrkv_init'] = np.concatenate((np.array([ 5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]), np.array([1]*(t-15))))
                print(recession_TaxCut_dict_OnceExtended_AD['EconomyMrkv_init'])
                this_recession_results = AggDemandEconomy.runExperiment(**recession_TaxCut_dict_OnceExtended_AD)
                recession_TaxCut_OnceExtended_all_results_AD += [this_recession_results]
            for key in output_keys:
                recession_TaxCut_OnceExtended_Cond9q_results_AD[key] = np.sum(np.array([recession_TaxCut_OnceExtended_all_results_AD[t-8][key]*recession_Cond9q_prob_array[t-8]  for t in range(8,max_recession_duration,1)]), axis=0)      
            saveAsPickleUnderVarName(recession_TaxCut_OnceExtended_Cond9q_results_AD,save_dir,locals()) 
            t1 = time()
            print('Calculating payroll tax cut during recession consumption took (with Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')    
        
            


    
    
    
    
    
    
    #%%
        
    if RunPlots:
        max_T = 20
        x_axis = np.arange(1,21)
        
        def loadAllResultsFromDir(load_dir,namespace,OnceExtendedFlag):
            AllResults = dict()            
            AllResults['recession_Cond9q_results']    = loadPickle('recession_Cond9q_results',load_dir,locals())
            AllResults['recession_Cond9q_results_AD'] = loadPickle('recession_Cond9q_results_AD',load_dir,locals())   
            AllResults['Rec_TaxCut_ContinuationProb_results']                 = loadPickle('recession_TaxCut_Cond9q_results',load_dir,locals())
            AllResults['Rec_TaxCut_ContinuationProb_results_AD']              = loadPickle('recession_TaxCut_Cond9q_results_AD',load_dir,locals())
            if OnceExtendedFlag:
                AllResults['Rec_TaxCut_ContinuationProb_OnceExtended_results']    = loadPickle('recession_TaxCut_OnceExtended_Cond9q_results',load_dir,locals())
                AllResults['Rec_TaxCut_ContinuationProb_OnceExtended_results_AD'] = loadPickle('recession_TaxCut_OnceExtended_Cond9q_results_AD',load_dir,locals())        
            return AllResults
        
        load_dir_Prob_0 = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_0/'
        AllResults_Op_Continuation = loadAllResultsFromDir(load_dir_Prob_0,locals(),False)
        
        load_dir_Prob_050 = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_050/'
        AllResults_O50p_Continuation = loadAllResultsFromDir(load_dir_Prob_050,locals(),True)
        
        load_dir_Prob_1 = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_1/'
        AllResults_1p_Continuation = loadAllResultsFromDir(load_dir_Prob_1,locals(),True)
        
        load_dir_Prob_050_Bas_and_Rec = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_050_Bas_and_Rec/'
        AllResults_050p_Bas_and_Rec_Continuation = loadAllResultsFromDir(load_dir_Prob_050_Bas_and_Rec,locals(),True)
        
        

        #%%
  
        AddCons_NoContinuationProb                  = getSimulationPercentDiff(AllResults_Op_Continuation['recession_Cond9q_results'],AllResults_Op_Continuation['Rec_TaxCut_ContinuationProb_results'],'AggCons')
        AddCons_ContinuationProb                    = getSimulationPercentDiff(AllResults_O50p_Continuation['recession_Cond9q_results'],AllResults_O50p_Continuation['Rec_TaxCut_ContinuationProb_results'],'AggCons')
        AddCons_ContinuationProb_1                  = getSimulationPercentDiff(AllResults_1p_Continuation['recession_Cond9q_results'],AllResults_1p_Continuation['Rec_TaxCut_ContinuationProb_results'],'AggCons')
        AddCons_ContinuationProb_050_Bas_and_Rec    = getSimulationPercentDiff(AllResults_050p_Bas_and_Rec_Continuation['recession_Cond9q_results'],AllResults_050p_Bas_and_Rec_Continuation['Rec_TaxCut_ContinuationProb_results'],'AggCons')
         
        AddInc_NoContinuationProb                   = getSimulationPercentDiff(AllResults_Op_Continuation['recession_Cond9q_results'],AllResults_Op_Continuation['Rec_TaxCut_ContinuationProb_results'],'AggIncome')
        AddInc_ContinuationProb                     = getSimulationPercentDiff(AllResults_O50p_Continuation['recession_Cond9q_results'],AllResults_O50p_Continuation['Rec_TaxCut_ContinuationProb_results'],'AggIncome')
        AddInc_ContinuationProb_1                   = getSimulationPercentDiff(AllResults_1p_Continuation['recession_Cond9q_results'],AllResults_1p_Continuation['Rec_TaxCut_ContinuationProb_results'],'AggIncome')
        AddInc_ContinuationProb_050_Bas_and_Rec     = getSimulationPercentDiff(AllResults_050p_Bas_and_Rec_Continuation['recession_Cond9q_results'],AllResults_050p_Bas_and_Rec_Continuation['Rec_TaxCut_ContinuationProb_results'],'AggIncome')

        AddCons_ContinuationProb_OnceExtended                    = getSimulationPercentDiff(AllResults_O50p_Continuation['recession_Cond9q_results'],AllResults_O50p_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results'],'AggCons')
        AddCons_ContinuationProb_1_OnceExtended                  = getSimulationPercentDiff(AllResults_1p_Continuation['recession_Cond9q_results'],AllResults_1p_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results'],'AggCons')
        AddCons_ContinuationProb_050_Bas_and_Rec_OnceExtended    = getSimulationPercentDiff(AllResults_050p_Bas_and_Rec_Continuation['recession_Cond9q_results'],AllResults_050p_Bas_and_Rec_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results'],'AggCons')
         
        AddInc_ContinuationProb_OnceExtended                     = getSimulationPercentDiff(AllResults_O50p_Continuation['recession_Cond9q_results'],AllResults_O50p_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results'],'AggIncome')
        AddInc_ContinuationProb_1_OnceExtended                   = getSimulationPercentDiff(AllResults_1p_Continuation['recession_Cond9q_results'],AllResults_1p_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results'],'AggIncome')
        AddInc_ContinuationProb_050_Bas_and_Rec_OnceExtended     = getSimulationPercentDiff(AllResults_050p_Bas_and_Rec_Continuation['recession_Cond9q_results'],AllResults_050p_Bas_and_Rec_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results'],'AggIncome')

          
        plt.figure(figsize=(15,10))
        plt.title('Tax Cut, no recession, no AD effects', size=30)
        
        plt.plot(x_axis,AddInc_NoContinuationProb[0:max_T]                              , color='black',linestyle='-')
        plt.plot(x_axis,AddInc_ContinuationProb[0:max_T]                                , color='black',linestyle='-')
        plt.plot(x_axis,AddInc_ContinuationProb_050_Bas_and_Rec[0:max_T]                , color='black',linestyle='-')
        
        plt.plot(x_axis,AddInc_ContinuationProb_OnceExtended[0:max_T]                   , color='black',linestyle='--')         
        plt.plot(x_axis,AddInc_ContinuationProb_1_OnceExtended[0:max_T]                 , color='black',linestyle='--')      
        plt.plot(x_axis,AddInc_ContinuationProb_050_Bas_and_Rec_OnceExtended[0:max_T]   , color='black',linestyle='--')
                
        plt.plot(x_axis,AddCons_NoContinuationProb[0:max_T]                             , color='blue',linestyle='-')
        plt.plot(x_axis,AddCons_ContinuationProb[0:max_T]                               , color='red',linestyle='-')
        plt.plot(x_axis,AddCons_ContinuationProb_050_Bas_and_Rec[0:max_T]               , color='green',linestyle='-')
        
        plt.plot(x_axis,AddCons_ContinuationProb_OnceExtended[0:max_T]                  , color='red',linestyle='--')           
        plt.plot(x_axis,AddCons_ContinuationProb_1_OnceExtended[0:max_T]                , color='orange',linestyle='--')
        plt.plot(x_axis,AddCons_ContinuationProb_050_Bas_and_Rec_OnceExtended[0:max_T]   , color='green',linestyle='--')
        
        
        
        
        plt.legend(['Inc: 8q tax cut, no cont. prob.','Inc: 8q tax cut, 50 % cont. prob.', 'Inc: 8q tax cut, 50 % prob. in baseline and recession', \
                    'Inc: 16q tax cut, 50 % cont. prob.', 'Inc: 16q tax cut, 100 % cont. prob.', 'Inc: 16q tax cut, 100 % cont. prob. in baseline and recession', \
                    'Cons: 8q tax cut, no cont. prob.','Cons: 8q tax cut, 50 % cont. prob.', 'Cons: 8q tax cut, 50 % prob. in baseline and recession', \
                    'Cons: 16q tax cut, 50 % cont. prob.', 'Cons: 16q tax cut, 100 % cont. prob.', 'Cons: 16q tax cut, 100 % cont. prob. in baseline and recession',], fontsize=10)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'/tax_cut_recession_no_AD_effects.pdf')
        plt.show()
        
        #%%
        
          
        AddCons_NoContinuationProb_AD                  = getSimulationPercentDiff(AllResults_Op_Continuation['recession_Cond9q_results_AD'],AllResults_Op_Continuation['Rec_TaxCut_ContinuationProb_results_AD'],'AggCons')
        AddCons_ContinuationProb_AD                    = getSimulationPercentDiff(AllResults_O50p_Continuation['recession_Cond9q_results_AD'],AllResults_O50p_Continuation['Rec_TaxCut_ContinuationProb_results_AD'],'AggCons')
        AddCons_ContinuationProb_1_AD                  = getSimulationPercentDiff(AllResults_1p_Continuation['recession_Cond9q_results_AD'],AllResults_1p_Continuation['Rec_TaxCut_ContinuationProb_results_AD'],'AggCons')
        AddCons_ContinuationProb_050_Bas_and_Rec_AD    = getSimulationPercentDiff(AllResults_050p_Bas_and_Rec_Continuation['recession_Cond9q_results_AD'],AllResults_050p_Bas_and_Rec_Continuation['Rec_TaxCut_ContinuationProb_results_AD'],'AggCons')
         
        AddInc_NoContinuationProb_AD                   = getSimulationPercentDiff(AllResults_Op_Continuation['recession_Cond9q_results_AD'],AllResults_Op_Continuation['Rec_TaxCut_ContinuationProb_results_AD'],'AggIncome')
        AddInc_ContinuationProb_AD                     = getSimulationPercentDiff(AllResults_O50p_Continuation['recession_Cond9q_results_AD'],AllResults_O50p_Continuation['Rec_TaxCut_ContinuationProb_results_AD'],'AggIncome')
        AddInc_ContinuationProb_1_AD                   = getSimulationPercentDiff(AllResults_1p_Continuation['recession_Cond9q_results_AD'],AllResults_1p_Continuation['Rec_TaxCut_ContinuationProb_results_AD'],'AggIncome')
        AddInc_ContinuationProb_050_Bas_and_Rec_AD     = getSimulationPercentDiff(AllResults_050p_Bas_and_Rec_Continuation['recession_Cond9q_results_AD'],AllResults_050p_Bas_and_Rec_Continuation['Rec_TaxCut_ContinuationProb_results_AD'],'AggIncome')

        AddCons_ContinuationProb_OnceExtended_AD                    = getSimulationPercentDiff(AllResults_O50p_Continuation['recession_Cond9q_results_AD'],AllResults_O50p_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results_AD'],'AggCons')
        AddCons_ContinuationProb_1_OnceExtended_AD                  = getSimulationPercentDiff(AllResults_1p_Continuation['recession_Cond9q_results_AD'],AllResults_1p_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results_AD'],'AggCons')
        AddCons_ContinuationProb_050_Bas_and_Rec_OnceExtended_AD    = getSimulationPercentDiff(AllResults_050p_Bas_and_Rec_Continuation['recession_Cond9q_results_AD'],AllResults_050p_Bas_and_Rec_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results_AD'],'AggCons')
         
        AddInc_ContinuationProb_OnceExtended_AD                    = getSimulationPercentDiff(AllResults_O50p_Continuation['recession_Cond9q_results_AD'],AllResults_O50p_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results_AD'],'AggIncome')
        AddInc_ContinuationProb_1_OnceExtended_AD                   = getSimulationPercentDiff(AllResults_1p_Continuation['recession_Cond9q_results_AD'],AllResults_1p_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results_AD'],'AggIncome')
        AddInc_ContinuationProb_050_Bas_and_Rec_OnceExtended_AD     = getSimulationPercentDiff(AllResults_050p_Bas_and_Rec_Continuation['recession_Cond9q_results_AD'],AllResults_050p_Bas_and_Rec_Continuation['Rec_TaxCut_ContinuationProb_OnceExtended_results_AD'],'AggIncome')

          
        
        plt.figure(figsize=(15,10))
        plt.title('Tax Cut, recession, AD effects', size=30)
        
        plt.plot(x_axis,AddInc_NoContinuationProb_AD[0:max_T]                              , color='blue',linestyle=':')
        plt.plot(x_axis,AddInc_ContinuationProb_AD[0:max_T]                                , color='red',linestyle=':')
        plt.plot(x_axis,AddInc_ContinuationProb_050_Bas_and_Rec_AD[0:max_T]                , color='green',linestyle=':')
        
        plt.plot(x_axis,AddInc_ContinuationProb_OnceExtended_AD[0:max_T]                   , color='red',linestyle=':')         
        plt.plot(x_axis,AddInc_ContinuationProb_1_OnceExtended_AD[0:max_T]                 , color='orange',linestyle=':')      
        plt.plot(x_axis,AddInc_ContinuationProb_050_Bas_and_Rec_OnceExtended_AD[0:max_T]   , color='green',linestyle=':')
                
        plt.plot(x_axis,AddCons_NoContinuationProb_AD[0:max_T]                             , color='blue',linestyle='-')
        plt.plot(x_axis,AddCons_ContinuationProb_AD[0:max_T]                               , color='red',linestyle='-')
        plt.plot(x_axis,AddCons_ContinuationProb_050_Bas_and_Rec_AD[0:max_T]               , color='green',linestyle='-')
        
        plt.plot(x_axis,AddCons_ContinuationProb_OnceExtended_AD[0:max_T]                  , color='red',linestyle='--')           
        plt.plot(x_axis,AddCons_ContinuationProb_1_OnceExtended_AD[0:max_T]                , color='orange',linestyle='--')
        plt.plot(x_axis,AddCons_ContinuationProb_050_Bas_and_Rec_OnceExtended_AD[0:max_T]   , color='green',linestyle='--')
        
        
        
        
        plt.legend(['Inc: 8q tax cut, no cont. prob.','Inc: 8q tax cut, 50 % cont. prob.', 'Inc: 8q tax cut, 50 % prob. in baseline and recession', \
                    'Inc: 16q tax cut, 50 % cont. prob.', 'Inc: 16q tax cut, 100 % cont. prob.', 'Inc: 16q tax cut, 100 % cont. prob. in baseline and recession', \
                    'Cons: 8q tax cut, no cont. prob.','Cons: 8q tax cut, 50 % cont. prob.', 'Cons: 8q tax cut, 50 % prob. in baseline and recession', \
                    'Cons: 16q tax cut, 50 % cont. prob.', 'Cons: 16q tax cut, 100 % cont. prob.', 'Cons: 16q tax cut, 100 % cont. prob. in baseline and recession',], fontsize=10)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'/tax_cut_recession_AD_effects.pdf')
        plt.show()
    
    

    
   
    