'''
This is the main script for the paper
'''
#$$$$$$$$$$ represents places in the code that need to be adjusted when the markov state space is changed
from Parameters import T_sim, init_infhorizon, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
     figs_dir
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
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,    # recession, payroll tax cut
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
        
    # Make the overall list of types
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

      # Run the payroll tax cut consumption level
    t0 = time()
    TaxCut_dict = base_dict_agg.copy()
    TaxCut_dict.update(**TaxCut_changes)
    AggDemandEconomy.ADelasticity = 0.0
    TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
    TaxCut_results = AggDemandEconomy.runExperiment(**TaxCut_dict)
    t1 = time()
    print('Calculating payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
    
    num_iterations = 4
    AggDemandEconomy.solveAD_Recession(num_iterations=num_iterations, name = 'Recession')
    AggDemandEconomy.solveAD_TaxCut(num_iterations=num_iterations, name = 'TaxCut')

    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'TaxCut')
    TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
    TaxCut_results_AD = AggDemandEconomy.runExperiment(**TaxCut_dict)
    t1 = time()
    print('Calculating payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # Run the recession consumption level
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'baseline')
    recession_dict = base_dict_agg.copy()
    recession_dict.update(**recession_changes)
    recession_all_results = []
    recession_results = dict()
    for t in range(max_recession_duration):
        recession_dict['EconomyMrkv_init'] = [1]*(t+1)
        this_recession_results = AggDemandEconomy.runExperiment(**recession_dict)
        recession_all_results += [this_recession_results]
    for recession_output in output_keys:
        recession_results[recession_output] = np.sum(np.array([recession_all_results[t][recession_output]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)
    t1 = time()
    print('Calculating recession consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # Run the recession consumption level
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'Recession')
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
    
    to_plot1 = 'NPV_AggCons'
    to_plot2 = 'NPV_AggIncome'
    to_plot3 = 'AggCons'
    to_plot4 = 'AggIncome'
    
    max_T = 20
    AddCons_AD  = TaxCut_results_AD[to_plot3]-base_results[to_plot3]
    AddInc_AD   = TaxCut_results_AD[to_plot4]-base_results[to_plot4] 
    AddCons     = TaxCut_results[to_plot3]-base_results[to_plot3]
    AddInc      = TaxCut_results[to_plot4]-base_results[to_plot4] 
    plt.figure(figsize=(15,10))
    plt.plot(AddInc[0:max_T], color='blue',linestyle='-')
    plt.plot(AddInc_AD[0:max_T], color='blue',linestyle='--')
    plt.plot(AddCons[0:max_T], color='red',linestyle='-')
    plt.plot(AddCons_AD[0:max_T], color='red',linestyle='--')
    plt.title('Tax Cut, no recession', size=30)
    plt.legend(['Income, no AD effects','Income, AD effects','Consumption, no AD effects','Consumption, AD effects'], fontsize=20)
    plt.show()
        
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
    plt.show()
    
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

    
 
    t_end = time()
    print('Doing everything took ' + mystr(t_end-t_start) + ' seconds in total.')
    
    # #%% Fiscal expenditure effectiveness
    
    # to_plot1 = 'NPV_AggCons'
    # to_plot2 = 'NPV_AggIncome'
    # to_plot3 = 'AggCons'
    # to_plot4 = 'AggIncome'
    
    # add_plot_text = ''
    
    
    # NPV_AddCons = UI_results[to_plot1]-base_results[to_plot1]
    # NPV_AddInc  = UI_results[to_plot2]-base_results[to_plot2]  
    # AddCons     = UI_results[to_plot3]-base_results[to_plot3]
    # AddInc      = UI_results[to_plot4]-base_results[to_plot4] 
    # plt.plot(AddInc)
    # plt.plot(AddCons)
    # plt.legend(['Fiscal policy expenditure, UI extension','Additional consumption, UI extension'])
    # plt.savefig(figs_dir +'UI_cut' + add_plot_text +'.pdf')
    # plt.show()
    # Stimulus_UI    = AddCons/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy

    
    # NPV_AddCons = recession_UI_results[to_plot1]-recession_results[to_plot1]
    # NPV_AddInc  = recession_UI_results[to_plot2]-recession_results[to_plot2]  
    # AddCons     = recession_UI_results[to_plot3]-recession_results[to_plot3]
    # AddInc      = recession_UI_results[to_plot4]-recession_results[to_plot4] 
    # plt.plot(AddInc)
    # plt.plot(AddCons)
    # plt.legend(['Fiscal policy expenditure, UI extension during recession','Additional consumption, UI extension during recession'])
    # plt.savefig(figs_dir +'UI_cut_rec' + add_plot_text +'.pdf')
    # plt.show()
    # Stimulus_UI_rec    = AddCons/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy

    
    # NPV_AddCons = TaxCut_results[to_plot1]-base_results[to_plot1]
    # NPV_AddInc  = TaxCut_results[to_plot2]-base_results[to_plot2]  
    # AddCons     = TaxCut_results[to_plot3]-base_results[to_plot3]
    # AddInc      = TaxCut_results[to_plot4]-base_results[to_plot4] 
    # plt.plot(AddInc)
    # plt.plot(AddCons)
    # plt.legend(['Fiscal policy expenditure, tax cut','Additional consumption, tax cut'])
    # plt.savefig(figs_dir +'tax_cut' + add_plot_text +'.pdf')
    # plt.show()
    # Stimulus_taxcut    = AddCons/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy

    
    # NPV_AddCons = recession_TaxCut_results[to_plot1]-recession_results[to_plot1]
    # NPV_AddInc  = recession_TaxCut_results[to_plot2]-recession_results[to_plot2]  
    # AddCons     = recession_TaxCut_results[to_plot3]-recession_results[to_plot3]
    # AddInc      = recession_TaxCut_results[to_plot4]-recession_results[to_plot4] 
    # plt.plot(AddInc)
    # plt.plot(AddCons)
    # plt.legend(['Fiscal policy expenditure, tax cut during recession','Additional consumption, tax cut during recession'])
    # plt.savefig(figs_dir +'tax_cut_rec' + add_plot_text +'.pdf')
    # plt.show()
    # Stimulus_taxcut_rec    = AddCons/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy
  
    
    # # Compare stimulus effects across policy interventions
    # plt.plot(Stimulus_UI)
    # plt.plot(Stimulus_UI_rec)
    # plt.plot(Stimulus_taxcut)
    # plt.plot(Stimulus_taxcut_rec)
    # plt.title('Stimulated consumption per period relative to NPV of policy intervention')
    # plt.legend(['UI','recession_UI','TaxCut','recession_TaxCut'])
    # plt.savefig(figs_dir +'stimulated-consumption' + add_plot_text +'.pdf')
    # plt.show()
 

