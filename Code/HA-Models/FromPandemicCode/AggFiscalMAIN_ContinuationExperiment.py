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
mystr = lambda x : '{:.2f}'.format(x)



## Which experiments to run / plots to show
Run_TaxCut = True
Run_Recession = True
Run_TaxCut_Recession = True
Make_Plots = True
Show_TestPlots = True



if __name__ == '__main__':
    
    t_start = time()
    
    # Setting up AggDemandEconmy
    from setupEconomy import AggDemandEconomy, base_dict_agg
    
    
    
    
    
    # Run the baseline consumption level
    t0 = time()
    base_results = AggDemandEconomy.runExperiment(**base_dict_agg)
    AggDemandEconomy.storeBaseline(base_results['AggCons'])
    t1 = time()
    print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')


    
    #%% Solving tax cut under Agg Multiplier  
    t0 = time()
    AggDemandEconomy.solveAD_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'TaxCut')
    t1 = time()
    print('Solving payroll tax cut took ' + mystr(t1-t0) + ' seconds.')
    
   #%% Running the payroll tax cut experiments
    
    save_dir = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/Continuation_Prob_0/'
    
    # Run the payroll tax cut consumption level in absence of Agg Multiplier
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'baseline')
    TaxCut_dict = base_dict_agg.copy()
    TaxCut_dict.update(**TaxCut_changes)
    TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
    TaxCut_results = AggDemandEconomy.runExperiment(**TaxCut_dict)
    SaveResult = open(save_dir +'TaxCut_results.csv', 'wb') 
    pickle.dump(TaxCut_results, SaveResult)  
    t1 = time()
    print('Calculating payroll tax cut consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')
    
    
    # Solutions are stored by solve_AD, this loads it so it can be easily simulated again
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'TaxCut')
    TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
    TaxCut_results_AD = AggDemandEconomy.runExperiment(**TaxCut_dict)
    SaveResult = open(save_dir +'TaxCut_results_AD.csv', 'wb') 
    pickle.dump(TaxCut_results_AD, SaveResult)  
    t1 = time()
    print('Calculating payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
    
    
    if init_infhorizon['TaxCutContinuationProb'] > 0:
        # Run the payroll tax cut consumption level in absence of Agg Multiplier, once extended
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'baseline')
        TaxCut_dict_OnceExtended = base_dict_agg.copy()
        TaxCut_dict_OnceExtended.update(**TaxCut_changes)
        TaxCut_dict_OnceExtended['EconomyMrkv_init'] = np.array(range(16))*2 + 4
        TaxCut_OnceExtended_results = AggDemandEconomy.runExperiment(**TaxCut_dict_OnceExtended)
        SaveResult = open(save_dir +'TaxCut_OnceExtended_results.csv', 'wb') 
        pickle.dump(TaxCut_OnceExtended_results, SaveResult)  
        t1 = time()
        print('Calculating payroll tax cut consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')
        
        
        # Solutions are stored by solve_AD, this loads it so it can be easily simulated again
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'TaxCut')
        TaxCut_dict_OnceExtended['EconomyMrkv_init'] = np.array(range(16))*2 + 4
        TaxCut_OnceExtended_results_AD = AggDemandEconomy.runExperiment(**TaxCut_dict_OnceExtended)
        SaveResult = open(save_dir +'TaxCut_OnceExtended_results_AD.csv', 'wb') 
        pickle.dump(TaxCut_OnceExtended_results_AD, SaveResult)  
        t1 = time()
        print('Calculating payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
    
    
    
    
    
    
    
    #%%     
    max_T = 20
    x_axis = np.arange(1,21)
    
    load_dir = 'C:/Users/ifr/Documents/GitHub/EdmundsFork/SavedPickleFiles/'
    
    # load all results
    SavedFile = open(load_dir + 'Continuation_Prob_0\TaxCut_results.csv', 'rb') 
    TaxCut_NoContinuationProb_results = pickle.load(SavedFile)
    SavedFile = open(load_dir + 'Continuation_Prob_0/TaxCut_results_AD.csv', 'rb') 
    TaxCut_NoContinuationProb_results_AD = pickle.load(SavedFile)
    
    SavedFile = open(load_dir + 'Continuation_Prob_050\TaxCut_results.csv', 'rb') 
    TaxCut_ContinuationProb_results = pickle.load(SavedFile)
    SavedFile = open(load_dir + 'Continuation_Prob_050\TaxCut_results_AD.csv', 'rb') 
    TaxCut_ContinuationProb_results_AD = pickle.load(SavedFile)    
    SavedFile = open(load_dir + 'Continuation_Prob_050\TaxCut_OnceExtended_results.csv', 'rb') 
    TaxCut_ContinuationProb_OnceExtended_results = pickle.load(SavedFile)
    SavedFile = open(load_dir + 'Continuation_Prob_050\TaxCut_OnceExtended_results_AD.csv', 'rb') 
    TaxCut_ContinuationProb_OnceExtended_results_AD = pickle.load(SavedFile)  
  
    #%%
    AddCons_NoContinuationProb                  = (TaxCut_NoContinuationProb_results['AggCons']-base_results['AggCons'])/base_results['AggCons']
    AddCons_ContinuationProb                    = (TaxCut_ContinuationProb_results['AggCons']-base_results['AggCons'])/base_results['AggCons']
    AddCons_ContinuationProb_OnceExtended       = (TaxCut_ContinuationProb_OnceExtended_results['AggCons']-base_results['AggCons'])/base_results['AggCons']
    AddInc_NoContinuationProb                   = (TaxCut_NoContinuationProb_results['AggIncome']-base_results['AggIncome'])/base_results['AggIncome']
    AddInc_ContinuationProb                     = (TaxCut_ContinuationProb_results['AggIncome']-base_results['AggIncome'])/base_results['AggIncome']
    AddInc_ContinuationProb_OnceExtended        = (TaxCut_ContinuationProb_OnceExtended_results['AggIncome']-base_results['AggIncome'])/base_results['AggIncome']
    
    plt.figure(figsize=(15,10))
    plt.title('Tax Cut, no recession, no AD effects', size=30)
    plt.plot(x_axis,AddInc_NoContinuationProb[0:max_T], color='blue',linestyle='-')
    plt.plot(x_axis,AddInc_ContinuationProb[0:max_T], color='blue',linestyle='--')
    plt.plot(x_axis,AddInc_ContinuationProb_OnceExtended[0:max_T], color='blue',linestyle=':')
    plt.plot(x_axis,AddCons_NoContinuationProb[0:max_T], color='red',linestyle='-')
    plt.plot(x_axis,AddCons_ContinuationProb[0:max_T], color='red',linestyle='--')
    plt.plot(x_axis,AddCons_ContinuationProb_OnceExtended[0:max_T], color='red',linestyle=':')
    plt.legend(['Inc: 8q tax cut, no cont. prob.','Inc: 8q tax cut, cont. prob.','Inc: 16q tax cut', \
                'Cons: 8q tax cut, no cont. prob.','Cons: 8q tax cut, cont. prob.','Cons: 16q tax cut'], fontsize=14)
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('quarter', fontsize=18)
    plt.ylabel('% diff. rel. to baseline', fontsize=16)
    plt.savefig(figs_dir +'/tax_cut_no_recession_no_AD_effects.pdf')
    plt.show()
    
    
    AddCons_NoContinuationProb_AD                  = (TaxCut_NoContinuationProb_results_AD['AggCons']-base_results['AggCons'])/base_results['AggCons']
    AddCons_ContinuationProb_AD                    = (TaxCut_ContinuationProb_results_AD['AggCons']-base_results['AggCons'])/base_results['AggCons']
    AddCons_ContinuationProb_OnceExtended_AD       = (TaxCut_ContinuationProb_OnceExtended_results_AD['AggCons']-base_results['AggCons'])/base_results['AggCons']
    AddInc_NoContinuationProb_AD                   = (TaxCut_NoContinuationProb_results_AD['AggIncome']-base_results['AggIncome'])/base_results['AggIncome']
    AddInc_ContinuationProb_AD                     = (TaxCut_ContinuationProb_results_AD['AggIncome']-base_results['AggIncome'])/base_results['AggIncome']
    AddInc_ContinuationProb_OnceExtended_AD        = (TaxCut_ContinuationProb_OnceExtended_results_AD['AggIncome']-base_results['AggIncome'])/base_results['AggIncome']
    
    plt.figure(figsize=(15,10))
    plt.title('Tax Cut, no recession, AD effects', size=30)
    plt.plot(x_axis,AddInc_NoContinuationProb_AD[0:max_T], color='blue',linestyle='-')
    plt.plot(x_axis,AddInc_ContinuationProb_AD[0:max_T], color='blue',linestyle='--')
    plt.plot(x_axis,AddInc_ContinuationProb_OnceExtended_AD[0:max_T], color='blue',linestyle=':')
    plt.plot(x_axis,AddCons_NoContinuationProb_AD[0:max_T], color='red',linestyle='-')
    plt.plot(x_axis,AddCons_ContinuationProb_AD[0:max_T], color='red',linestyle='--')
    plt.plot(x_axis,AddCons_ContinuationProb_OnceExtended_AD[0:max_T], color='red',linestyle=':')
    plt.legend(['Inc: 8q tax cut, no cont. prob.','Inc: 8q tax cut, cont. prob.','Inc: 16q tax cut', \
                'Cons: 8q tax cut, no cont. prob.','Cons: 8q tax cut, cont. prob.','Cons: 16q tax cut'], fontsize=14)
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('quarter', fontsize=18)
    plt.ylabel('% diff. rel. to baseline', fontsize=16)
    plt.savefig(figs_dir +'/tax_cut_no_recession_AD_effects.pdf')
    plt.show()
    
    

    
   
    
    
    
    
    #%% Proof that using AggDemandEconomy.history['AggDemandFac'] in runExperiment is correct:
    
    # Period 8:
    CRatio_q7 = TaxCut_OnceExtended_results_AD['AggIncome'][7]/TaxCut_OnceExtended_results['AggIncome'][7]
    Cratio_q8_Prediction = AggDemandEconomy.CFunc[3*18][3*20](CRatio_q7)
    AggDemandFac_q8 = Cratio_q8_Prediction**0.4
    print('AggDemandFac replicated', AggDemandFac_q8)
    print('AggDemandFac history', AggDemandEconomy.history['AggDemandFac'][8])
    print('AggDemandFac Prev history', AggDemandEconomy.history['AggDemandFacPrev'][8])
    
    # Period 4
    CRatio_q3 = TaxCut_OnceExtended_results_AD['AggIncome'][3]/TaxCut_OnceExtended_results['AggIncome'][3]
    Cratio_q4_Prediction = AggDemandEconomy.CFunc[3*10][3*12](CRatio_q3)
    AggDemandFac_q4 = Cratio_q4_Prediction**0.4
    print('AggDemandFac replicated', AggDemandFac_q4)
    print('AggDemandFac history', AggDemandEconomy.history['AggDemandFac'][4])
  
    # Period 0
    CRatio_qmin1 = 1
    Cratio_q0_Prediction = AggDemandEconomy.CFunc[3*0][3*4](CRatio_qmin1)
    AggDemandFac_q0 = Cratio_q0_Prediction**0.4
    print('AggDemandFac replicated', AggDemandFac_q0)
    print('AggDemandFac history', AggDemandEconomy.history['AggDemandFac'][0])