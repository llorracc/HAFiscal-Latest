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


if __name__ == '__main__':
    
    mystr = lambda x : '{:.2f}'.format(x)
    t_start = time()
    base_dict_agg = deepcopy(base_dict)
    
    # Make baseline types - for now only one type, might add more
    num_types = 1 
    # This is not the number of discount factors, but the number of household types; in pandemic paper, there were different education groups
    InfHorizonTypeAgg = AggFiscalType(**init_infhorizon)
    InfHorizonTypeAgg.cycles = 0
    AggDemandEconomy = AggregateDemandEconomy(**init_ADEconomy)
    InfHorizonTypeAgg.getEconomyData(AggDemandEconomy)
    BaseTypeList = [InfHorizonTypeAgg]
  
    # Fill in the Markov income distribution for each base type
    #$$$$$$$$$$
    # NOTE: THIS ASSUMES NO LIFECYCLE
    IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg.IncUnemp])])
    IncomeDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg.IncUnempNoBenefits])])
    IncomeDstn_big = []
    for ThisType in BaseTypeList:
        IncomeDstn_taxcut = deepcopy(ThisType.IncomeDstn[0])
        IncomeDstn_taxcut.X[1] = IncomeDstn_taxcut.X[1]*ThisType.TaxCutIncFactor
        IncomeDstn_big.append([ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, extended UI
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, extended UI
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut   
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp])  # recession, payroll tax cut   

                             
                               
        ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp]
        ThisType.IncomeDstn_big = IncomeDstn_big
        ThisType.AgentCount = AgentCountTotal
        ThisType.DiscFac = 0.96
        ThisType.seed = 0
        
    # The number of discount factors is set in parameters; need to test whether more disc factors work as well
    # Edmund said debugging might be necessary
   #%%     
    # Make the overall list of types; #IF: Not clear yet
    TypeList = []
    n = 0
    for b in range(DiscFacDstns[0].X.size):
        for e in range(num_types):
            DiscFac = DiscFacDstns[e].X[b]
            AgentCount = int(np.floor(AgentCountTotal*TypeShares[e]*DiscFacDstns[e].pmf[b]))
            ThisType = deepcopy(BaseTypeList[e])
            ThisType.AgentCount = AgentCount
            ThisType.DiscFac = DiscFac
            ThisType.seed = n
            TypeList.append(ThisType)
            n += 1
    AggDemandEconomy.agents = TypeList

    AggDemandEconomy.solve()

    AggDemandEconomy.reset()
    for agent in AggDemandEconomy.agents:
        agent.initializeSim()
        agent.AggDemandFac = 1.0
        agent.RfreeNow = 1.0
        agent.CaggNow = 1.0
    
    AggDemandEconomy.makeHistory()   
    AggDemandEconomy.saveState()   
    AggDemandEconomy.switchToCounterfactualMode()
    AggDemandEconomy.makeIdiosyncraticShockHistories()
    
    output_keys = ['cNrm_all', 'TranShk_all', 'cLvl_all', 'pLvl_all', 'mNrm_all', 'aNrm_all', 'cLvl_all_splurge', 
                      'NPV_AggIncome', 'NPV_AggCons', 'AggIncome', 'AggCons']
    
    max_policy_duration = 6
    PolicyUBspell = AggDemandEconomy.agents[0].PolicyUBspell #NOTE - this should come from the market, not the agent
    PolicyUBpersist = 1.-1./PolicyUBspell
    policy_prob_array = np.array([PolicyUBpersist**t*(1-PolicyUBpersist) for t in range(max_policy_duration)])
    policy_prob_array[-1] = 1.0 - np.sum(policy_prob_array[:-1])

    max_recession_duration = 21
    Rspell = AggDemandEconomy.agents[0].Rspell #NOTE - this should come from the market, not the agent
    R_persist = 1.-1./Rspell
    recession_prob_array = np.array([R_persist**t*(1-R_persist) for t in range(max_recession_duration)])
    recession_prob_array[-1] = 1.0 - np.sum(recession_prob_array[:-1])
    
    
    # Run the baseline consumption level
    t0 = time()
    base_results = AggDemandEconomy.runExperiment(**base_dict_agg)
    AggDemandEconomy.storeBaseline(base_results['AggCons'])
    t1 = time()
    print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
    
    #%% Solving recession under Agg Multiplier   
    t0 = time()
    AggDemandEconomy.solveAD_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession')
    t1 = time()
    print('Solving recession took ' + mystr(t1-t0) + ' seconds.')
    
    
    #%% Running the recession experiment
    
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
        this_recession_results = AggDemandEconomy.runExperiment(**recession_dict)
        recession_all_results += [this_recession_results]
    for recession_output in output_keys:
        recession_results[recession_output] = np.sum(np.array([recession_all_results[t][recession_output]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    t1 = time()
    print('Calculating recession consumption took (no Agg Multiplier)' + mystr(t1-t0) + ' seconds.')    

#%%    
    # Run the recession consumption level in presence of the Agg Multiplier
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'Recession')
    recession_dict = base_dict_agg.copy()
    recession_dict.update(**recession_changes)
    recession_all_results_AD = []
    recession_results_AD = dict()
    for t in range(max_recession_duration):
        recession_dict['EconomyMrkv_init'] = [1]*(t+1)
        this_recession_results_AD = AggDemandEconomy.runExperiment(**recession_dict)
        recession_all_results_AD += [this_recession_results_AD]
    for recession_output_AD in output_keys:
        recession_results_AD[recession_output_AD] = np.sum(np.array([recession_all_results_AD[t][recession_output_AD]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    t1 = time()
    print('Calculating recession consumption took ' + mystr(t1-t0) + ' seconds.')
    
    
    
    
    #%% testing
    max_T = 20
    plt.figure(figsize=(15,10))
    plt.plot(recession_results_AD['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results_AD[0]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results_AD[4]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results_AD[8]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results_AD[12]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results_AD[16]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.legend(['Weighted','0','4','8','12','16'], fontsize=20)
    plt.show()
    
    #%% testing
    max_T = 20
    plt.figure(figsize=(15,10))
    plt.plot(recession_results['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results[0]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results[4]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results[8]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results[12]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_all_results[16]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.legend(['Weighted','0','4','8','12','16'], fontsize=20)
    plt.show()
    
  
    #%%     
    to_plot1 = 'NPV_AggCons'
    to_plot2 = 'NPV_AggIncome'
    to_plot3 = 'AggCons'
    to_plot4 = 'AggIncome' 
    max_T = 20
    
        
    AddCons_AD  = recession_results_AD[to_plot3]-base_results[to_plot3]
    AddInc_AD   = recession_results_AD[to_plot4]-base_results[to_plot4] 
    AddCons     = recession_results[to_plot3]-base_results[to_plot3]
    AddInc      = recession_results[to_plot4]-base_results[to_plot4] 
    plt.figure(figsize=(15,10))
    plt.plot(AddInc[0:max_T], color='blue',linestyle='-')
    plt.plot(AddInc_AD[0:max_T], color='blue',linestyle='--')
    plt.plot(AddCons[0:max_T], color='red',linestyle='-')
    plt.plot(AddCons_AD[0:max_T], color='red',linestyle='--')
    plt.title('Recession', size=30)
    plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=20)
    plt.savefig(figs_dir +'recession.pdf')
    plt.show()
    

    

    #%% Line of best fit
    
   
    x = (recession_all_results_AD[-1]['Cratio_hist'][0:19]-1) 
    y = recession_all_results_AD[-1]['Cratio_hist'][1:20]
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
    



