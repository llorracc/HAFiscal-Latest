'''
This is the main script for the paper
'''
from Parameters import T_sim, init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
     DiscFacCount, AgentCountTotal, base_dict, figs_dir, num_max_iterations_solvingAD,\
     convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
     data_LorenzPts, data_LorenzPtsAll, data_avgLWPI, data_LWoPI, data_EducShares, data_WealthShares,\
     DiscFacInit, DiscFacSpread,\
     max_recession_duration, num_experiment_periods,\
     recession_changes, sticky_e_changes, UI_changes, recession_UI_changes,\
     TaxCut_changes, recession_TaxCut_changes,recession_Check_changes
         
     # init_infhorizon, TypeShares, \
     
from AggFiscalModel import AggFiscalType, AggregateDemandEconomy
from HARK.distribution import DiscreteDistribution
from time import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
from OtherFunctions import getSimulationDiff, getSimulationPercentDiff, getStimulus, getNPVMultiplier, \
                    saveAsPickleUnderVarName, loadPickle, namestr, saveAsPickle
                    
from threading import Thread

mystr = lambda x : '{:.2f}'.format(x)

if __name__ == '__main__':
        
    
    # Setting up AggDemandEconmy
    
    # Make education types
    num_types = 3
    # This is not the number of discount factors, but the number of household types
    
    InfHorizonTypeAgg_d = AggFiscalType(**init_dropout)
    InfHorizonTypeAgg_d.cycles = 0
    InfHorizonTypeAgg_h = AggFiscalType(**init_highschool)
    InfHorizonTypeAgg_h.cycles = 0
    InfHorizonTypeAgg_c = AggFiscalType(**init_college)
    InfHorizonTypeAgg_c.cycles = 0
    AggDemandEconomy = AggregateDemandEconomy(**init_ADEconomy)
    InfHorizonTypeAgg_d.getEconomyData(AggDemandEconomy)
    InfHorizonTypeAgg_h.getEconomyData(AggDemandEconomy)
    InfHorizonTypeAgg_c.getEconomyData(AggDemandEconomy)
    BaseTypeList = [InfHorizonTypeAgg_d, InfHorizonTypeAgg_h, InfHorizonTypeAgg_c ]
          
    # Fill in the Markov income distribution for each base type
    # NOTE: THIS ASSUMES NO LIFECYCLE
    IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg_d.IncUnemp])])
    IncomeDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg_d.IncUnempNoBenefits])])
        
    for ThisType in BaseTypeList:
        EmployedIncomeDstn = deepcopy(ThisType.IncomeDstn[0])
        ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0]] + [IncomeDstn_unemp]*UBspell_normal + [IncomeDstn_unemp_nobenefits] 
        ThisType.IncomeDstn_base = ThisType.IncomeDstn
        
        IncomeDstn_recession = [ThisType.IncomeDstn[0]*(2*(num_experiment_periods+1))] # for normal, rec, recovery  
        ThisType.IncomeDstn_recession = IncomeDstn_recession
        ThisType.IncomeDstn_recessionUI = IncomeDstn_recession
        
        EmployedIncomeDstn.X[1] = EmployedIncomeDstn.X[1]*ThisType.TaxCutIncFactor
        TaxCutStatesIncomeDstn = [EmployedIncomeDstn] + [IncomeDstn_unemp]*UBspell_normal + [IncomeDstn_unemp_nobenefits] 
        IncomeDstn_recessionTaxCut = deepcopy(IncomeDstn_recession)
        # Tax states are 2,3 (q1) 4,5 (q2) ... 16,17 (q8)
        for i in range(2*num_base_MrkvStates,18*num_base_MrkvStates,1):
            IncomeDstn_recessionTaxCut[0][i] =  TaxCutStatesIncomeDstn[np.mod(i,4)]
        ThisType.IncomeDstn_recessionTaxCut = IncomeDstn_recessionTaxCut
        
        ThisType.IncomeDstn_recessionCheck = deepcopy(IncomeDstn_recession)
    

        
    # Make the overall list of types
    TypeList = []
    n = 0
    for e in range(num_types):
        for b in range(DiscFacCount):
            DiscFac = DiscFacDstns[e].X[b]
            AgentCount = int(np.floor(AgentCountTotal*data_EducShares[e]*DiscFacDstns[e].pmf[b]))
            ThisType = deepcopy(BaseTypeList[e])
            ThisType.AgentCount = AgentCount
            ThisType.DiscFac = DiscFac
            ThisType.seed = n
            TypeList.append(ThisType)
            n += 1
    #base_dict['Agents'] = TypeList    
    
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
    AggDemandEconomy.switchToCounterfactualMode("base")
    AggDemandEconomy.makeIdiosyncraticShockHistories()
    
    output_keys = ['NPV_AggIncome', 'NPV_AggCons', 'AggIncome', 'AggCons']
    
    
    base_dict_agg = deepcopy(base_dict)
    
    Rspell = AggDemandEconomy.agents[0].Rspell #NOTE - this should come from the market, not the agent
    R_persist = 1.-1./Rspell
    recession_prob_array = np.array([R_persist**t*(1-R_persist) for t in range(max_recession_duration)])
    recession_prob_array[-1] = 1.0 - np.sum(recession_prob_array[:-1])
   
    x = np.zeros(21)
    for i in range(21):
        x[i] = AggDemandEconomy.agents[i].AgentCount
       
        

    # Run the baseline consumption level
    t0 = time()
    base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = True)
    AggDemandEconomy.storeBaseline(base_results['AggCons'])     
    t1 = time()
    print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
    
            
    def runExperimentsAllRecessions(dict_changes,AggDemandEconomy):
        
        t0 = time()
        dictt = base_dict_agg.copy()
        dictt.update(**dict_changes)
        all_results = []
        avg_results = dict()
        #  running recession with diferent lengths up to max_recession_duration then averaging the result
        for t in [20]:
            dictt['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
            dictt['EconomyMrkv_init'][0:t+1] = np.array(dictt['EconomyMrkv_init'][0:t+1]) +1
            print(dictt['EconomyMrkv_init'])
            this_result = AggDemandEconomy.runExperiment(**dictt, Full_Output = True)
            all_results += [this_result]
        # for key in output_keys:
        #     avg_results[key] = np.sum(np.array([all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)   
        t1 = time()
        print('Calculating took ' + mystr(t1-t0) + ' seconds.') 
        return [this_result,all_results]
    
    

    
    
    #%% Run the payroll tax cut consumption level in absence of Agg Multiplier


    AggDemandEconomy.switch_shock_type('recession')
    AggDemandEconomy.solve()
    [recession_results,recession_all_results] = runExperimentsAllRecessions(recession_changes,AggDemandEconomy)

    AggDemandEconomy.switch_shock_type('recessionTaxCut')
    AggDemandEconomy.solve()
    [results_taxcut,all_results_taxcut] = runExperimentsAllRecessions(recession_TaxCut_changes,AggDemandEconomy)

 #%%

    AddCons_TaxCut              = getSimulationPercentDiff(recession_results,    results_taxcut,'AggCons')
    AddInc_TaxCut               = getSimulationPercentDiff(recession_results,    results_taxcut,'AggIncome')
    
    # Value of policy expenditure (need to consider non-AD solution)
    NPV_AddInc_TaxCut           = getSimulationDiff(recession_results,    results_taxcut,'NPV_AggIncome')
    AddInc_TaxCut_Abs           = getSimulationDiff(recession_results,    results_taxcut,'AggIncome') 
    Stimulus_TaxCut             = getStimulus(recession_results,    results_taxcut,NPV_AddInc_TaxCut[-1]) 
    NPV_Multiplier_TaxCut       = getNPVMultiplier(recession_results,    results_taxcut,NPV_AddInc_TaxCut)
    
    
    
    def calculate_NPV(X,Periods,R):
        NPV_discount = np.zeros(Periods)
        for t in range(Periods):
            NPV_discount[t] = 1/(R**t)
        NPV = np.zeros(Periods)
        for t in range(Periods):
            NPV[t] = np.sum(X[0:t+1]*NPV_discount[0:t+1])  
        return NPV


    N = 900
    MultiplierFarAwayFrom1 = np.zeros(N)  
    for t in range(N):
        consb = recession_results['cLvl_all_splurge'][:,t]
        incb  = recession_results['pLvl_all'][:,t]*recession_results['TranShk_all'][:,t]
        mrkb  = recession_results['Mrkv_hist'][:,t]
        cons  = results_taxcut['cLvl_all_splurge'][:,t]
        inc   = results_taxcut['pLvl_all'][:,t]*results_taxcut['TranShk_all'][:,t]
        mrk   = results_taxcut['Mrkv_hist'][:,t]
        
        #print('Mrk b ',mrkb[0:8])
        #print('Mrk  ',mrk[0:8])
        #print('Inc ',(inc[0:8]-incb[0:8])/incb[0:8])
        #print('Cons ',(cons[0:8]-consb[0:8])/consb[0:8])
        T = 40
        a=(calculate_NPV(inc,T,1.01)-calculate_NPV(incb,T,1.01))
        b=(calculate_NPV(cons,T,1.01)-calculate_NPV(consb,T,1.01))
        if a[-1] > 0:
            MultiplierFarAwayFrom1[t]=(b[-1]/a[-1])<0.98
        else:
            MultiplierFarAwayFrom1[t]=0
        #print(100*(calculate_NPV(cons,100,1.01)[-1]-calculate_NPV(inc,100,1.01)[-1])/calculate_NPV(inc,100,1.01)[-1])    
        
        
    Whodies = [np.sum(AggDemandEconomy.agents[i].history['who_dies'],axis = 0) for i in range(0,21)]
    Whodieslong = [element for sub in Whodies for element in sub][0:N]
    Whodies_vs_MultiplierNot1 = (MultiplierFarAwayFrom1-Whodieslong[0:N])>0
    
        
        
    
    AgentsWhereInconsistency = np.where(Whodies_vs_MultiplierNot1)
    
    
    print('Instances where multiplier is not close to 1 but no death: ', np.sum(Whodies_vs_MultiplierNot1))
    