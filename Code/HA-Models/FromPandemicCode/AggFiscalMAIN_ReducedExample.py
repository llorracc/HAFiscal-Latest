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
    
    
    # Run the baseline consumption level
    t0 = time()
    base_results = AggDemandEconomy.runExperiment(**base_dict_agg)
    AggDemandEconomy.storeBaseline(base_results['AggCons'])
    t1 = time()
    print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')


    
    #%% Solving recession under Agg Multiplier   
    t0 = time()
    AggDemandEconomy.solveAD_Recession(num_max_iterations=10,convergence_cutoff=1E-3, name = 'Recession')
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
    
    
    #%% Test 1, Why is AggDemandFac_Init not correctly reflected in AggDemandEconomy.history['AggDemandFac']
    
    # these should be equal
    print(AggDemandEconomy.cLvl_splurgeNow[0][0:5])                  #current simulation
    print(recession_all_results_AD[-1]['cLvl_all_splurge'][-1][0:5]) #last simulation stored
    # They are equal only after fixing l.199 in AggFiscalModel.py
    
        
   
    C_1stQ_Rec_1Q_AD = recession_all_results_AD[-1]['AggCons'][0]
    C_1stQ_Base = base_results['AggCons'][0] # this corresponds perfectly to AggDemandEconomy.base_AggCons
    print('Cratio from stored results: ',C_1stQ_Rec_1Q_AD/C_1stQ_Base)
    
    
    print('The following five values should be equal, but they are not')
    
    AggDemandFac_Init = AggDemandEconomy.ADFunc(AggDemandEconomy.CFunc[0][3](recession_all_results_AD[-1]['Cratio_hist'][0])) #l. 425-428
    print('AggDemandFac_Init: ',AggDemandFac_Init)
    AggDemandFac_Init = AggDemandEconomy.ADFunc(AggDemandEconomy.CFunc[0][3](C_1stQ_Rec_1Q_AD/C_1stQ_Base)) #l. 425-428
    print('AggDemandFac_Init: ',AggDemandFac_Init)
    AggDemandFac_Init = AggDemandEconomy.ADFunc(AggDemandEconomy.CFunc[0][3](AggDemandEconomy.CFunc[0][3].intercept)) #l. 473
    print('AggDemandFac_Init: ',AggDemandFac_Init)
    

    Inc_1stQ_Rec_1Q_AD = recession_all_results_AD[-1]['AggIncome'][0]
    Inc_1stQ_Rec_1Q    = recession_all_results[-1]['AggIncome'][0]
    Inc_Ratio = Inc_1stQ_Rec_1Q_AD/Inc_1stQ_Rec_1Q
    print('Inc_Ratio:',Inc_Ratio)
    print('AggDemandEconomy.history[AggDemandFacPrev][0]',AggDemandEconomy.history['AggDemandFacPrev'][0])
    
    #%% Test 2, Why is AggDemandFac not correctly reflected in AggDemandEconomy.history['AggDemandFac']
    
    period = 10
    
    C_1stQ_Rec_1Q_AD = recession_all_results_AD[-1]['AggCons'][period]
    C_1stQ_Base = base_results['AggCons'][period] # this corresponds perfectly to AggDemandEconomy.base_AggCons
    print('Cratio from stored results: ',C_1stQ_Rec_1Q_AD/C_1stQ_Base)
    
    print('The following four values should be equal, but they are not')
    AggDemandFac = AggDemandEconomy.ADFunc(AggDemandEconomy.CFunc[3][3](recession_all_results_AD[-1]['Cratio_hist'][period]))  #l. 425-428
    print('AggDemandFac: ',AggDemandFac)
    AggDemandFac = AggDemandEconomy.ADFunc(AggDemandEconomy.CFunc[3][3](C_1stQ_Rec_1Q_AD/C_1stQ_Base)) #l. 425-428
    print('AggDemandFac: ',AggDemandFac)

    Inc_1stQ_Rec_1Q_AD = recession_all_results_AD[-1]['AggIncome'][period]
    Inc_1stQ_Rec_1Q    = recession_all_results[-1]['AggIncome'][period]
    Inc_Ratio = Inc_1stQ_Rec_1Q_AD/Inc_1stQ_Rec_1Q
    print('Inc_Ratio:',Inc_Ratio)
    print('AggDemandEconomy.history[AggDemandFac][0]',AggDemandEconomy.history['AggDemandFac'][period])