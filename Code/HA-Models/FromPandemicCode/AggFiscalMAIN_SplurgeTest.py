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
Run_TaxCut              = True



if __name__ == '__main__':
    
    t_start = time()
    
    
    # Setting up AggDemandEconmy
    from setupEconomy import AggDemandEconomy, base_dict_agg, max_recession_duration, output_keys, recession_prob_array, \
                             max_policy_duration, policy_prob_array
        
        
    if Run_Baseline:   
        # Run the baseline consumption level
        t0 = time()
        base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = True)
        #saveAsPickleUnderVarName(base_results,figs_dir,locals())
        AggDemandEconomy.storeBaseline(base_results['AggCons'])     
        t1 = time()
        print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
    
    

    #%% Solving and Simulating
    
    if Run_TaxCut:        
        # Run the payroll tax cut consumption level in absence of Agg Multiplier
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'baseline')
        TaxCut_dict = base_dict_agg.copy()
        TaxCut_dict.update(**TaxCut_changes)
        TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
        TaxCut_results_intermediate = AggDemandEconomy.runExperiment(**TaxCut_dict, Full_Output = True)
        TaxCut_results = TaxCut_results_intermediate
        print('Calculating payroll tax cut consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')
     
    
    

        

            
        if Run_TaxCut:   
            AddCons_TaxCut              = getSimulationPercentDiff(base_results,    TaxCut_results,'AggCons')
            AddInc_TaxCut               = getSimulationPercentDiff(base_results,    TaxCut_results,'AggIncome')
            
            # Value of policy expenditure (need to consider non-AD solution)
            NPV_AddInc_TaxCut           = getSimulationDiff(base_results,TaxCut_results,'NPV_AggIncome')
            AddInc_TaxCut_Abs           = getSimulationDiff(base_results,TaxCut_results,'AggIncome') 
            Stimulus_TaxCut             = getStimulus(base_results,TaxCut_results,NPV_AddInc_TaxCut[-1]) 
            NPV_Multiplier_TaxCut       = getNPVMultiplier(base_results,TaxCut_results,NPV_AddInc_TaxCut)
            
      
        
    
    
    t_end = time()
    print('Doing everything took ' + mystr(t_end-t_start) + ' seconds in total.')
   
    
#%%  
    
def calculate_NPV(X,Periods,R):
    NPV_discount = np.zeros(Periods)
    for t in range(Periods):
        NPV_discount[t] = 1/(R**t)
    NPV = np.zeros(Periods)
    for t in range(Periods):
        NPV[t] = np.sum(X[0:t+1]*NPV_discount[0:t+1])  
    return NPV

diff = np.zeros(2000)  
for t in range(2000):
    consb = base_results['cLvl_all_splurge'][:,t]
    incb  = base_results['pLvl_all'][:,t]*base_results['TranShk_all'][:,t]
    mrkb  = base_results['Mrkv_hist'][:,t]
    cons  = TaxCut_results['cLvl_all_splurge'][:,t]
    inc   = TaxCut_results['pLvl_all'][:,t]*TaxCut_results['TranShk_all'][:,t]
    mrk   = TaxCut_results['Mrkv_hist'][:,t]
    
    #print('Mrk b ',mrkb[0:8])
    #print('Mrk  ',mrk[0:8])
    #print('Inc ',(inc[0:8]-incb[0:8])/incb[0:8])
    #print('Cons ',(cons[0:8]-consb[0:8])/consb[0:8])
    T = 100
    a=(calculate_NPV(inc,T,1.01)-calculate_NPV(incb,T,1.01))
    b=(calculate_NPV(cons,T,1.01)-calculate_NPV(consb,T,1.01))
    print(b[-1]/a[-1])
    diff[t]=(b[-1]/a[-1])<0.99
    #print(100*(calculate_NPV(cons,100,1.01)[-1]-calculate_NPV(inc,100,1.01)[-1])/calculate_NPV(inc,100,1.01)[-1])    
   
Whodies_vs_MultiplierNot1 = diff[0:2000]-np.sum(AggDemandEconomy.agents[0].history['who_dies'],axis=0)[0:2000]>0   

np.sum(Whodies_vs_MultiplierNot1)

print(np.sum(AggDemandEconomy.agents[0].history['who_dies'],axis=0)>0)


