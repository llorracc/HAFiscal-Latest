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
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp])  # recession, payroll tax cut
        ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp]
        ThisType.IncomeDstn_big = IncomeDstn_big
        ThisType.AgentCount = AgentCountTotal
        ThisType.DiscFac = 0.96
        ThisType.seed = 0
 #%%        
    # The number of discount factors is set in parameters; need to test whether more disc factors work as well
    # Edmund said debugging might be necessary
        
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
    
    
    ##WHY DOES THIS TAKE SO LONG???? Because it has to solve the model, and this takes much longer here.
    # Run the baseline consumption level
    t0 = time()
    base_results = AggDemandEconomy.runExperiment(**base_dict_agg)
    AggDemandEconomy.storeBaseline(base_results['AggCons'])
    t1 = time()
    print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # # Run the extended UI consumption level
    # t0 = time()
    # UI_dict = base_dict_agg.copy()
    # UI_dict.update(**UI_changes)
    # UI_all_results = []
    # UI_results = dict()
    # for t in range(max_policy_duration):
    #     UI_dict['EconomyMrkv_init'] = [2]*(t+1)
    #     this_UI_results = AggDemandEconomy.runExperiment(**UI_dict)
    #     UI_all_results += [this_UI_results]
    # for UI_output in output_keys:
    #     UI_results[UI_output] = np.sum(np.array([UI_all_results[t][UI_output]*policy_prob_array[t]  for t in range(max_policy_duration)]), axis=0)
    # t1 = time()
    # print('Calculating extended UI consumption took ' + mystr(t1-t0) + ' seconds.')
    
    #%% Solving recession under Agg Multiplier   
    t0 = time()
    AggDemandEconomy.solveAD_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession')
    t1 = time()
    print('Solving recession took ' + mystr(t1-t0) + ' seconds.')
    #%% Solving tax cut under Agg Multiplier  
    t0 = time()
    AggDemandEconomy.solveAD_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'TaxCut')
    t1 = time()
    print('Solving payroll tax cut took ' + mystr(t1-t0) + ' seconds.')
    #%%  Solving tax cut during recession under Agg Multiplier  
    t0 = time()
    AggDemandEconomy.solveAD_Recession_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_TaxCut')
    t1 = time()
    print('Solving payroll tax cut during recession took ' + mystr(t1-t0) + ' seconds.')
    
    #%% Running the payroll tax cut experiments
    
    # Run the payroll tax cut consumption level in absence of Agg Multiplier
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'baseline')
    TaxCut_dict = base_dict_agg.copy()
    TaxCut_dict.update(**TaxCut_changes)
    TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
    TaxCut_results = AggDemandEconomy.runExperiment(**TaxCut_dict)
    t1 = time()
    print('Calculating payroll tax cut consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')
    
    
    # Solutions are stored by solve_AD, this loads it so it can be easily simulated again
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'TaxCut')
    TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
    TaxCut_results_AD = AggDemandEconomy.runExperiment(**TaxCut_dict)
    t1 = time()
    print('Calculating payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
    
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
    
    
    #%% Running the payroll tax cut during recession experiment
    
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
        this_recession_results = AggDemandEconomy.runExperiment(**recession_TaxCut_dict)
        recession_TaxCut_all_results += [this_recession_results]
    for recession_output in output_keys:
        recession_TaxCut_results[recession_output] = np.sum(np.array([recession_TaxCut_all_results[t][recession_output]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    t1 = time()
    print('Calculating payroll tax cut during recession consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')    
    
    #%%
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
        this_recession_results_AD = AggDemandEconomy.runExperiment(**recession_TaxCut_dict)
        recession_TaxCut_all_results_AD += [this_recession_results_AD]
    for recession_output_AD in output_keys:
        recession_TaxCut_results_AD[recession_output_AD] = np.sum(np.array([recession_TaxCut_all_results_AD[t][recession_output_AD]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    t1 = time()
    print('Calculating payroll tax cut during recession consumption took ' + mystr(t1-t0) + ' seconds.')
    
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
    
    #%% testing
    max_T = 20
    plt.figure(figsize=(15,10))
    plt.plot(recession_TaxCut_results_AD['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results_AD[0]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results_AD[4]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results_AD[8]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results_AD[12]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results_AD[16]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.legend(['Weighted','0','4','8','12','16'], fontsize=20)
    plt.show()
    
    #%% testing
    max_T = 20
    plt.figure(figsize=(15,10))
    plt.plot(recession_TaxCut_results['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[0]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[4]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[8]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[12]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[16]['AggIncome'][0:max_T]-base_results['AggIncome'][0:max_T])
    plt.legend(['Weighted','0','4','8','12','16'], fontsize=20)
    plt.show()
    #%% testing
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
    #%% testing
    max_T = 20
    plt.figure(figsize=(15,10))
    plt.plot(recession_TaxCut_results['AggIncome'][0:max_T]-recession_results['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[0]['AggIncome'][0:max_T]-recession_all_results[0]['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[4]['AggIncome'][0:max_T]-recession_all_results[4]['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[8]['AggIncome'][0:max_T]-recession_all_results[8]['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[12]['AggIncome'][0:max_T]-recession_all_results[12]['AggIncome'][0:max_T])
    plt.plot(recession_TaxCut_all_results[16]['AggIncome'][0:max_T]-recession_all_results[16]['AggIncome'][0:max_T])
    plt.legend(['Weighted','0','4','8','12','16'], fontsize=20)
    plt.show()
    #%%     
    to_plot1 = 'NPV_AggCons'
    to_plot2 = 'NPV_AggIncome'
    to_plot3 = 'AggCons'
    to_plot4 = 'AggIncome' 
    max_T = 20
    
    AddCons_AD      = TaxCut_results_AD[to_plot3]-base_results[to_plot3]
    AddInc_AD       = TaxCut_results_AD[to_plot4]-base_results[to_plot4]
    NPV_AddInc_AD   = TaxCut_results_AD[to_plot2]-base_results[to_plot2]
    AddCons         = TaxCut_results[to_plot3]-base_results[to_plot3]
    AddInc          = TaxCut_results[to_plot4]-base_results[to_plot4]  
    NPV_AddInc      = TaxCut_results[to_plot2]-base_results[to_plot2]  
    plt.figure(figsize=(15,10))
    plt.plot(AddInc[0:max_T], color='blue',linestyle='-')
    plt.plot(AddInc_AD[0:max_T], color='blue',linestyle='--')
    plt.plot(AddCons[0:max_T], color='red',linestyle='-')
    plt.plot(AddCons_AD[0:max_T], color='red',linestyle='--')
    plt.title('Tax Cut, no recession', size=30)
    plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=20)
    plt.savefig(figs_dir +'tax_cut.pdf')
    plt.show()
    
    Stimulus_taxcut    = AddCons/NPV_AddInc[-1]        #divide by total cumulative NPV of the policy
    Stimulus_taxcut_AD = AddCons_AD/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy
        
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
    
    AddCons_AD  = recession_TaxCut_results_AD[to_plot3]-base_results[to_plot3]
    AddInc_AD   = recession_TaxCut_results_AD[to_plot4]-base_results[to_plot4] 
    AddCons     = recession_TaxCut_results[to_plot3]-base_results[to_plot3]
    AddInc      = recession_TaxCut_results[to_plot4]-base_results[to_plot4] 
    plt.figure(figsize=(15,10))
    plt.plot(AddInc[0:max_T], color='blue',linestyle='-')
    plt.plot(AddInc_AD[0:max_T], color='blue',linestyle='--')
    plt.plot(AddCons[0:max_T], color='red',linestyle='-')
    plt.plot(AddCons_AD[0:max_T], color='red',linestyle='--')
    plt.title('Tax Cut during recession (rel to base)', size=30)
    plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=20)
    plt.savefig(figs_dir +'taxcut_recession.pdf')
    plt.show()
    
    AddCons_AD  = recession_TaxCut_results_AD[to_plot3]-recession_results_AD[to_plot3]
    AddInc_AD   = recession_TaxCut_results_AD[to_plot4]-recession_results_AD[to_plot4] 
    AddCons     = recession_TaxCut_results[to_plot3]-recession_results[to_plot3]
    AddInc      = recession_TaxCut_results[to_plot4]-recession_results[to_plot4] 
    plt.figure(figsize=(15,10))
    plt.plot(AddInc[0:max_T], color='blue',linestyle='-')
    plt.plot(AddInc_AD[0:max_T], color='blue',linestyle='--')
    plt.plot(AddCons[0:max_T], color='red',linestyle='-')
    plt.plot(AddCons_AD[0:max_T], color='red',linestyle='--')
    plt.title('Tax Cut during recession (rel to recession)', size=30)
    plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=20)
    plt.savefig(figs_dir +'taxcut_recession2.pdf')
    plt.show()
    

    AddInc_tax_AD   = recession_TaxCut_results_AD[to_plot4]/recession_results_AD[to_plot4] 
    AddInc_tax      = recession_TaxCut_results[to_plot4]/recession_results[to_plot4]
    plt.figure(figsize=(15,10))
    plt.plot(AddInc_tax[0:max_T], color='red',linestyle='-')
    plt.plot(AddInc_tax_AD[0:max_T], color='red',linestyle='--')
    plt.title('', size=30)
    plt.legend(['Income, no AD effects, tax cut plus recession','Income, AD effects, tax cut plus recession'], fontsize=20)
    plt.show()
    
    
    NPV_AddInc                      = recession_TaxCut_results[to_plot2]-recession_results[to_plot2]  
    AddCons                         = recession_TaxCut_results[to_plot3]-recession_results[to_plot3]
    NPV_AddInc_AD                   = recession_TaxCut_results_AD[to_plot2]-recession_results_AD[to_plot2]  
    AddCons_AD                      = recession_TaxCut_results_AD[to_plot3]-recession_results_AD[to_plot3]    
    Stimulus_taxcut_recession       = AddCons/NPV_AddInc[-1]  
    Stimulus_taxcut_recession_AD    = AddCons_AD/NPV_AddInc[-1]

  
    
    # Compare stimulus effects across policy interventions
    plt.figure(figsize=(15,10))
    plt.plot(Stimulus_taxcut[0:max_T], color='blue',linestyle='-')
    plt.plot(Stimulus_taxcut_AD[0:max_T], color='blue',linestyle='--')
    plt.plot(Stimulus_taxcut_recession[0:max_T], color='red',linestyle='-')
    plt.plot(Stimulus_taxcut_recession_AD[0:max_T], color='red',linestyle='--')
    plt.title('Stimulated consumption per period relative to NPV of policy intervention', size=30)
    plt.legend(['Tax cut, no AD effects','Tax cut, AD effects','Tax cut during recession, no AD effects','Tax cut during recession, AD effects'], fontsize=20)
    plt.savefig(figs_dir +'stimulated-consumption.pdf')
    plt.show()
    
    

    
    #%%
    # This should yield a straigt line as the underlying assumption of the numerical algorithm is that future Cratio can be 
    # predicted applying a linear function on current Cratio.
    plt.figure(figsize=(15,10))
    plt.plot(recession_all_results_AD[-1]['Cratio_hist'][0:19],recession_all_results_AD[-1]['Cratio_hist'][1:20])
    plt.title('CRatio[t]/CRatio[t-1]', size=30)
    plt.savefig(figs_dir +'CRatio.pdf')
    plt.show()
    #%% 
    # # Run the recession and extended UI consumption level
    # # This is SUPER SLOW because of the double loop
    # t0 = time()
    # recession_UI_dict = base_dict_agg.copy()
    # recession_UI_dict.update(**recession_UI_changes)
    # recession_UI_all_results = []
    # recession_UI_results = dict()
    # for t_R in range(max_recession_duration):
    #     for t_Policy in range(max_policy_duration):
    #         recession_UI_dict['EconomyMrkv_init'] = np.array([0]*max(max_recession_duration,max_policy_duration))
    #         recession_UI_dict['EconomyMrkv_init'][0:t_R+1] += 1
    #         recession_UI_dict['EconomyMrkv_init'][0:t_Policy+1] +=2
    #         this_recession_UI_results = AggDemandEconomy.runExperiment(**recession_UI_dict)
    #         recession_UI_all_results += [this_recession_UI_results]
    # for recession_UI_output in output_keys:
    #     recession_UI_results[recession_UI_output] = np.zeros_like(recession_UI_all_results[0][recession_UI_output])
    #     count = 0
    #     for t_R in range(max_recession_duration):
    #         for t_Policy in range(max_policy_duration):
    #             recession_UI_results[recession_UI_output] += recession_UI_all_results[count][recession_UI_output]*recession_prob_array[t_R]*policy_prob_array[t_Policy]
    #             count += 1
    # t1 = time()
    # print('Calculating recession and extended UI consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # # Run the recession and payroll tax cut consumption level
    # t0 = time()
    # recession_TaxCut_dict = base_dict_agg.copy()
    # recession_TaxCut_dict.update(**recession_TaxCut_changes)
    # recession_TaxCut_all_results = []
    # recession_TaxCut_results = dict()
    # for t in range(max_recession_duration):
    #     recession_TaxCut_dict['EconomyMrkv_init'] = np.array([0]*max(max_recession_duration,8))
    #     recession_TaxCut_dict['EconomyMrkv_init'][0:8] = np.array(range(8))*2 + 4
    #     recession_TaxCut_dict['EconomyMrkv_init'][0:t+1] += 1
    #     this_recession_TaxCut_results = AggDemandEconomy.runExperiment(**recession_TaxCut_dict)
    #     recession_TaxCut_all_results += [this_recession_TaxCut_results]
    # for recession_TaxCut_output in output_keys:
    #     recession_TaxCut_results[recession_TaxCut_output] = np.sum(np.array([recession_TaxCut_all_results[t][recession_TaxCut_output]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    # t1 = time()
    # print('Calculating recession and payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')

    
 
    # t_end = time()
    # print('Doing everything took ' + mystr(t_end-t_start) + ' seconds in total.')
    


