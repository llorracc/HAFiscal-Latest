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


# solves under AD / no AD
Run_Recession           = True
Run_Check               = False
Run_NonAD_Simulations   = False
Make_Plots              = False

# This runs some investigations into the baseline check experiment
Run_Baseline_Check      = False





# Setting up AggDemandEconmy
t0 = time()
from setupEconomy import AggDemandEconomy, base_dict_agg, max_recession_duration, output_keys, recession_prob_array
t1 = time()
print('Setting up everything took ' + mystr(t1-t0) + ' seconds.')

#%%                        

# Run the baseline consumption level
t0 = time()
base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = False)
t1 = time()
print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
AggDemandEconomy.storeBaseline(base_results['AggCons']) 
saveAsPickleUnderVarName(base_results,figs_dir,locals())


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
    
    
    #%% Check whether AD Func works properly
    def percChange(x,y):
        return 100*abs(y-x)/x
    
    # # To check 0 1, 1 0  and 11
    
    print('Error [0][1] in %:', percChange(AggDemandEconomy.MacroCFunc[0][1](1),recession_all_results_AD[0]['Cratio_hist'][0]))
    
    Error0to0 = np.zeros(1000)
    i=0
    for RecLength in range(1,22):
        for quarter in range(RecLength+1,30):
            Error0to0[i] = percChange(AggDemandEconomy.MacroCFunc[0][0](recession_all_results_AD[RecLength-1]['Cratio_hist'][quarter-1]), \
                                                                        recession_all_results_AD[RecLength-1]['Cratio_hist'][quarter])     
            i +=1
    print('Maximum percentage error for [0][0]: ', np.max(Error0to0))
    
    Error1to1 = np.zeros(1000)
    i=0
    for RecLength in range(2,22): 
        for quarter in range(1,RecLength):
            Error1to1[i] = percChange(AggDemandEconomy.MacroCFunc[1][1](recession_all_results_AD[RecLength-1]['Cratio_hist'][quarter-1]), \
                                                                        recession_all_results_AD[RecLength-1]['Cratio_hist'][quarter])
            i += 1
    print('Maximum percentage error for [1][1]: ', np.max(Error1to1))

    Error1to0 = np.zeros(1000)
    for RecLength in range(1,22): 
        Error1to0[RecLength] = percChange(AggDemandEconomy.MacroCFunc[1][0](recession_all_results_AD[RecLength-1]['Cratio_hist'][RecLength-1]), \
                                                                            recession_all_results_AD[RecLength-1]['Cratio_hist'][RecLength])   
    print('Maximum percentage error for [1][0]: ', np.max(Error1to0))      
    
    percChange(AggDemandEconomy.MacroCFunc[1][0](recession_all_results_AD[1]['Cratio_hist'][1]), \
                                                                            recession_all_results_AD[0]['Cratio_hist'][2]) 
    
    #%%
    
    
    if Run_NonAD_Simulations:
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
            this_sim_results = AggDemandEconomy.runExperiment(**recession_dict, Full_Output = False)
            recession_all_results += [this_sim_results]
        for key in output_keys:
            recession_results[key] = np.sum(np.array([recession_all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
        saveAsPickleUnderVarName(recession_all_results,figs_dir,locals())
        saveAsPickleUnderVarName(recession_results,figs_dir,locals())
        t1 = time()
        print('Calculating recession consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')


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
        this_sim_results = AggDemandEconomy.runExperiment(**recession_Check_dict, Full_Output = False)
        recession_Check_all_results_AD += [this_sim_results]
    for key in output_keys:
        recession_Check_results_AD[key] = np.sum(np.array([recession_Check_all_results_AD[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    saveAsPickleUnderVarName(recession_Check_all_results_AD,figs_dir,locals())
    saveAsPickleUnderVarName(recession_Check_results_AD,figs_dir,locals())
    t1 = time()
    print('Calculating recession + check consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')
    
    
    #%% Check whether AD Func works properly
    def percChange(x,y):
        return 100*abs(y-x)/x
    
    print('Error [0][37] in %:', percChange(AggDemandEconomy.MacroCFunc[0][37](1),recession_Check_all_results_AD[0]['Cratio_hist'][0]))
    
    print('Error [37][1] in %:', percChange(AggDemandEconomy.MacroCFunc[37][1](recession_Check_all_results_AD[0]['Cratio_hist'][0]),recession_Check_all_results_AD[1]['Cratio_hist'][1]))
    
    print('Error [37][0] in %:', percChange(AggDemandEconomy.MacroCFunc[37][0](recession_Check_all_results_AD[0]['Cratio_hist'][0]),recession_Check_all_results_AD[0]['Cratio_hist'][1]))
    
    
    Error1to1 = np.zeros(1000)
    i=0
    for RecLength in range(3,22): #RecLength at least three to get 1 1 jump
        for quarter in range(2,RecLength):
            Error1to1[i] = percChange(AggDemandEconomy.MacroCFunc[1][1](recession_Check_all_results_AD[RecLength-1]['Cratio_hist'][quarter-1]), \
                                                                        recession_Check_all_results_AD[RecLength-1]['Cratio_hist'][quarter])           
            #print('Error [1][1] from q', quarter ,'to next q, and RecLength', RecLength ,'in %:', Error1to1[i])
            i += 1
    print('Maximum percentage error for [1][1]: ', np.max(Error1to1))

    Error1to0 = np.zeros(1000)
    for RecLength in range(2,22): #RecLength 2 first Recession jumping from 1 to 0    
        Error1to0[RecLength] = percChange(AggDemandEconomy.MacroCFunc[1][0](recession_Check_all_results_AD[RecLength-1]['Cratio_hist'][RecLength-1]), \
                                                                 recession_Check_all_results_AD[RecLength-1]['Cratio_hist'][RecLength])   
    print('Maximum percentage error for [1][0]: ', np.max(Error1to0))      
    
    Error0to0 = np.zeros(1000)
    i=0
    for RecLength in range(1,22):
        for quarter in range(RecLength+1,30):
            Error0to0[i] = percChange(AggDemandEconomy.MacroCFunc[0][0](recession_Check_all_results_AD[RecLength-1]['Cratio_hist'][quarter-1]), \
                                                                    recession_Check_all_results_AD[RecLength-1]['Cratio_hist'][quarter])     
            i +=1
    print('Maximum percentage error for [0][0]: ', np.max(Error0to0))            

   

    #%%
    
    if Run_NonAD_Simulations:
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
            this_sim_results = AggDemandEconomy.runExperiment(**recession_Check_dict, Full_Output = False)
            recession_Check_all_results += [this_sim_results]
        for key in output_keys:
            recession_Check_results[key] = np.sum(np.array([recession_Check_all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
        saveAsPickleUnderVarName(recession_Check_all_results,figs_dir,locals())
        saveAsPickleUnderVarName(recession_Check_results,figs_dir,locals())
        t1 = time()
        print('Calculating recession + check consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')
        


#%% Compare Recession and Recession + Check with AD 
    
if Make_Plots:

    max_T = 20
    x_axis = np.arange(1,21)
    
    folder = './Figures/Check_Experiment/'
    folder_allSt = './Figures/Check_Experiment_allStates/'
    
    base_results                    = loadPickle('base_results',folder,locals())
    recession_results               = loadPickle('recession_results',folder,locals())
    recession_results_AD            = loadPickle('recession_results_AD',folder,locals())
    recession_results_AD_allSt      = loadPickle('recession_results_AD',folder_allSt,locals())
    recession_Check_results         = loadPickle('recession_Check_results',folder,locals())
    recession_Check_results_AD      = loadPickle('recession_Check_results_AD',folder,locals())
    recession_Check_results_AD_allSt= loadPickle('recession_Check_results_AD',folder_allSt,locals())
    
    NPV_AddInc_Check_Rec                    = getSimulationDiff(recession_results,      recession_Check_results,    'NPV_AggIncome') # Policy expenditure
    NPV_Multiplier_Check_Rec                = getNPVMultiplier( recession_results,      recession_Check_results,    NPV_AddInc_Check_Rec)
    NPV_Multiplier_Check_Rec_AD             = getNPVMultiplier( recession_results_AD,   recession_Check_results_AD, NPV_AddInc_Check_Rec)
    NPV_Multiplier_Check_Rec_AD_allSt       = getNPVMultiplier( recession_results_AD_allSt, recession_Check_results_AD_allSt, NPV_AddInc_Check_Rec)
    
 
    print('Long-run multiplier for check stimulus during recession, no AD effects: '    ,mystr(NPV_Multiplier_Check_Rec[-1]))
    print('Long-run multiplier for check stimulus during recession, with AD effects: '  ,mystr(NPV_Multiplier_Check_Rec_AD[-1]))
    print('Long-run multiplier for check stimulus during recession, with AD effects in all states: '  ,mystr(NPV_Multiplier_Check_Rec_AD_allSt[-1]))
    
    plt.figure(figsize=(15,10))
    plt.title('NPV multiplier for Recession + Check experiment', size=30)
    plt.plot(np.arange(1,81),NPV_Multiplier_Check_Rec[0:80], color='blue',linestyle='-')
    plt.plot(np.arange(1,81),NPV_Multiplier_Check_Rec_AD[0:80], color='red',linestyle='-')
    plt.plot(np.arange(1,81),NPV_Multiplier_Check_Rec_AD_allSt[0:80], color='green',linestyle='-')
    plt.legend(['no AD',' with AD', 'with AD, all states'], fontsize=14)
    plt.xticks(np.arange(1,81, 1.0))
    plt.xlabel('quarter', fontsize=18)
    plt.savefig(figs_dir +'NPVmulti_recession_Check_relrecession.pdf')
    plt.show() 
    
    
    AddCons_Rec           = getSimulationPercentDiff(recession_results,    recession_Check_results,'AggCons')
    AddInc_Rec            = getSimulationPercentDiff(recession_results,    recession_Check_results,'AggIncome')
    AddCons_Rec_AD        = getSimulationPercentDiff(recession_results_AD, recession_Check_results_AD,'AggCons')
    AddInc_Rec_AD         = getSimulationPercentDiff(recession_results_AD, recession_Check_results_AD,'AggIncome')
    AddCons_Rec_AD_allSt  = getSimulationPercentDiff(recession_results_AD_allSt, recession_Check_results_AD_allSt,'AggCons')
    AddInc_Rec_AD_allSt   = getSimulationPercentDiff(recession_results_AD_allSt, recession_Check_results_AD_allSt,'AggIncome')
    
    max_T = 20
    x_axis = np.arange(1,21)
    
    plt.figure(figsize=(15,10))
    plt.title('Recession + Check experiment', size=30)
    plt.plot(x_axis,AddCons_Rec[0:max_T], color='blue',linestyle='-')
    plt.plot(x_axis,AddInc_Rec[0:max_T],  color='blue',linestyle='--')
    plt.plot(x_axis,AddCons_Rec_AD[0:max_T], color='red',linestyle='-')
    plt.plot(x_axis,AddInc_Rec_AD[0:max_T],  color='red',linestyle='--')
    plt.plot(x_axis,AddCons_Rec_AD_allSt[0:max_T], color='green',linestyle='-')
    plt.plot(x_axis,AddInc_Rec_AD_allSt[0:max_T],  color='green',linestyle='--')
    plt.legend(['Cons, no AD','Inc, no AD','Cons, with AD','Inc, with AD','Cons, with AD all states','Inc, with AD all states'], fontsize=14)
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('quarter', fontsize=18)
    plt.ylabel('% diff. rel. to recession', fontsize=16)
    plt.savefig(figs_dir +'recession_Check_relrecession.pdf')
    plt.show() 

 
#%% Check during baseline 
    
if Run_Baseline_Check: 

    # Run the baseline consumption level
    t0 = time()
    base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = True)
    t1 = time()
    print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
    AggDemandEconomy.storeBaseline(base_results['AggCons']) 
       
    # Run Check experiment during baseline
    t0 = time()
    Check_dict = base_dict_agg.copy()
    Check_dict.update(**Check_changes)
    Check_dict['EconomyMrkv_init'] = [36]
    Check_results = AggDemandEconomy.runExperiment(**Check_dict, Full_Output = True)
    t1 = time()
    print('Calculating check experiment ' + mystr(t1-t0) + ' seconds.')
    
            
    output_keys = ['NPV_AggIncome', 'NPV_AggCons', 'AggIncome', 'AggCons', 'pLvl_all','TranShk_all']
              
    # Recession only
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
    InvestigateResults(recession_results,recession_Check_results)     
    
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


