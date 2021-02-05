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


    
    #%% Solving tax cut under Agg Multiplier  
    t0 = time()
    AggDemandEconomy.solveAD_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'TaxCut')
    t1 = time()
    print('Solving payroll tax cut took ' + mystr(t1-t0) + ' seconds.')
    
   #%% Running the payroll tax cut experiments
    
    # Run the payroll tax cut consumption level in absence of Agg Multiplier
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'baseline')
    TaxCut_dict = base_dict_agg.copy()
    TaxCut_dict.update(**TaxCut_changes)
    TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
    TaxCut_results = AggDemandEconomy.runExperiment(**TaxCut_dict)
    SaveResult = open(figs_dir +'/TaxCut_results.csv', 'wb') 
    pickle.dump(TaxCut_results, SaveResult)  
    t1 = time()
    print('Calculating payroll tax cut consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')
    
    
    # Solutions are stored by solve_AD, this loads it so it can be easily simulated again
    t0 = time()
    AggDemandEconomy.restoreADsolution(name = 'TaxCut')
    TaxCut_dict['EconomyMrkv_init'] = np.array(range(8))*2 + 4
    TaxCut_results_AD = AggDemandEconomy.runExperiment(**TaxCut_dict)
    SaveResult = open(figs_dir +'/TaxCut_results_AD.csv', 'wb') 
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
        SaveResult = open(figs_dir +'/TaxCut_OnceExtended_results.csv', 'wb') 
        pickle.dump(TaxCut_OnceExtended_results, SaveResult)  
        t1 = time()
        print('Calculating payroll tax cut consumption took (no Agg Multiplier) ' + mystr(t1-t0) + ' seconds.')
        
        
        # Solutions are stored by solve_AD, this loads it so it can be easily simulated again
        t0 = time()
        AggDemandEconomy.restoreADsolution(name = 'TaxCut')
        TaxCut_dict_OnceExtended['EconomyMrkv_init'] = np.array(range(16))*2 + 4
        TaxCut_OnceExtended_results_AD = AggDemandEconomy.runExperiment(**TaxCut_dict_OnceExtended)
        SaveResult = open(figs_dir +'/TaxCut_OnceExtended_results_AD.csv', 'wb') 
        pickle.dump(TaxCut_OnceExtended_results_AD, SaveResult)  
        t1 = time()
        print('Calculating payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
    
    
    
    
    
    
    
    #%%     
    max_T = 20
    x_axis = np.arange(1,21)
    
    
    # load all results
    SavedFile = open('Figures/ContinuationProb0/TaxCut_results.csv', 'rb') 
    TaxCut_NoContinuationProb_results = pickle.load(SavedFile)
    SavedFile = open('Figures/ContinuationProb0/TaxCut_results_AD.csv', 'rb') 
    TaxCut_NoContinuationProb_results_AD = pickle.load(SavedFile)
    
    SavedFile = open('Figures/ContinuationProb50/TaxCut_results.csv', 'rb') 
    TaxCut_ContinuationProb_results = pickle.load(SavedFile)
    SavedFile = open('Figures/ContinuationProb50/TaxCut_results_AD.csv', 'rb') 
    TaxCut_ContinuationProb_results_AD = pickle.load(SavedFile)    
    SavedFile = open('Figures/ContinuationProb50/TaxCut_OnceExtended_results.csv', 'rb') 
    TaxCut_ContinuationProb_OnceExtended_results = pickle.load(SavedFile)
    SavedFile = open('Figures/ContinuationProb50/TaxCut_OnceExtended_results_AD.csv', 'rb') 
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
    plt.plot(x_axis,AddCons_ContinuationProb_OnceExtended[0:max_T], color='red',linestyle='--')
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
    plt.plot(x_axis,AddCons_ContinuationProb_OnceExtended_AD[0:max_T], color='red',linestyle='--')
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