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
                    saveAsPickleUnderVarName, loadPickle, namestr     
mystr = lambda x : '{:.2f}'.format(x)

## Which experiments to run / plots to show
Run_Baseline            = True
Run_Recession           = True
Run_Check_Recession     = False
Run_UB_Ext_Recession    = False
Run_TaxCut_Recession    = False

Run_AD                  = False
Run_1stRoundAD          = False
Run_NonAD               = True #whether to run nonAD experiments as well


Make_Plots              = False


#%% 

if __name__ == '__main__':
        
    
    # Setting up AggDemandEconmy
    
    #%% 
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
    
       
        
    if Run_Baseline:   
        # Run the baseline consumption level
        t0 = time()
        base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = False)
        saveAsPickleUnderVarName(base_results,figs_dir,locals())
        AggDemandEconomy.storeBaseline(base_results['AggCons'])     
        t1 = time()
        print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
             
        
    def runExperimentsAllRecessions(dict_changes):
        
        t0 = time()
        dictt = base_dict_agg.copy()
        dictt.update(**dict_changes)
        all_results = []
        avg_results = dict()
        #  running recession with diferent lengths up to max_recession_duration then averaging the result
        for t in range(max_recession_duration):
            dictt['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
            dictt['EconomyMrkv_init'][0:t+1] = np.array(dictt['EconomyMrkv_init'][0:t+1]) +1
            print(dictt['EconomyMrkv_init'])
            this_result = AggDemandEconomy.runExperiment(**dictt, Full_Output = False)
            all_results += [this_result]
        for key in output_keys:
            avg_results[key] = np.sum(np.array([all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)   
        t1 = time()
        print('Calculating took ' + mystr(t1-t0) + ' seconds.') 
        return [avg_results,all_results]

    #%% 
    if Run_Recession:
                
        if Run_NonAD:   
            print('Calculating Recession with no AD effects')
            AggDemandEconomy.switch_shock_type("recession")
            AggDemandEconomy.solve()
            [recession_results,recession_all_results] = runExperimentsAllRecessions(recession_changes)
            saveAsPickleUnderVarName(recession_all_results,figs_dir,locals())
            saveAsPickleUnderVarName(recession_results,figs_dir,locals())
        if Run_AD:
            # Solving recession under Agg Multiplier   
            t0 = time()
            AggDemandEconomy.switch_shock_type("recession")
            AggDemandEconomy.solveAD_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession')
            t1 = time()
            print('Solving recession took ' + mystr(t1-t0) + ' seconds.')
            
            print('Calculating Recession with AD effects')
            AggDemandEconomy.switch_shock_type("recession")
            AggDemandEconomy.restoreADsolution(name = 'Recession')
            [recession_results_AD,recession_all_results_AD] = runExperimentsAllRecessions(recession_changes)
            saveAsPickleUnderVarName(recession_all_results_AD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_results_AD,figs_dir,locals())
            
        if Run_1stRoundAD:
            # Solving recession under Agg Multiplier   
            t0 = time()
            AggDemandEconomy.switch_shock_type("recession")
            AggDemandEconomy.solveAD_Recession(num_max_iterations=1,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_1stRoundAD')
            t1 = time()
            print('Solving recession 1st round AD took ' + mystr(t1-t0) + ' seconds.')
           
            print('Calculating Recession with 1st round AD effects')
            AggDemandEconomy.switch_shock_type("recession")
            AggDemandEconomy.restoreADsolution(name = 'Recession_1stRoundAD')
            [recession_results_firstRoundAD,recession_all_results_firstRoundAD] = runExperimentsAllRecessions(recession_changes)
            saveAsPickleUnderVarName(recession_all_results_firstRoundAD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_results_firstRoundAD,figs_dir,locals())
    #%% 
    if Run_Check_Recession:
                
        if Run_NonAD:   
            print('Calculating Check Recession with no AD effects')
            AggDemandEconomy.switch_shock_type("recessionCheck")
            AggDemandEconomy.solve()
            [recession_Check_results,recession_Check_all_results] = runExperimentsAllRecessions(recession_Check_changes)
            saveAsPickleUnderVarName(recession_Check_all_results,figs_dir,locals())
            saveAsPickleUnderVarName(recession_Check_results,figs_dir,locals())
            
        if Run_AD:
            # Solving check under Agg Multiplier   
            t0 = time()
            AggDemandEconomy.switch_shock_type("recessionCheck")
            AggDemandEconomy.solveAD_Check_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_Check')
            t1 = time()
            print('Solving recession check took ' + mystr(t1-t0) + ' seconds.')
            
            print('Calculating Check Recession with AD effects')
            AggDemandEconomy.switch_shock_type("recessionCheck")
            AggDemandEconomy.restoreADsolution(name = 'Recession_Check')
            [recession_Check_results_AD,recession_Check_all_results_AD] = runExperimentsAllRecessions(recession_Check_changes)
            saveAsPickleUnderVarName(recession_Check_all_results_AD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_Check_results_AD,figs_dir,locals())
            
        if Run_1stRoundAD:
            # Solving check under Agg Multiplier   
            t0 = time()
            AggDemandEconomy.switch_shock_type("recessionCheck")
            AggDemandEconomy.solveAD_Check_Recession(num_max_iterations=1,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_Check_1stRoundAD')
            t1 = time()
            print('Solving recession check 1st round AD took ' + mystr(t1-t0) + ' seconds.')
           
            print('Calculating Check Recession with 1st round AD effects')
            AggDemandEconomy.switch_shock_type("recessionCheck")
            AggDemandEconomy.restoreADsolution(name = 'Recession_Check_1stRoundAD')
            [recession_Check_results_firstRoundAD,recession_Check_all_results_firstRoundAD] = runExperimentsAllRecessions(recession_Check_changes)
            saveAsPickleUnderVarName(recession_Check_all_results_firstRoundAD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_Check_results_firstRoundAD,figs_dir,locals())
    #%% 
    if Run_UB_Ext_Recession:
        
        if Run_NonAD:
            print('Calculating UI recession with no AD effects')
            AggDemandEconomy.switch_shock_type("recessionUI")
            AggDemandEconomy.solve()
            [recession_UI_results,recession_UI_all_results] = runExperimentsAllRecessions(recession_UI_changes)
            saveAsPickleUnderVarName(recession_UI_all_results,figs_dir,locals())
            saveAsPickleUnderVarName(recession_UI_results,figs_dir,locals())
            
        if Run_AD:     
            # Solving UI under Agg Multiplier  
            t0 = time()
            AggDemandEconomy.switch_shock_type("recessionUI")
            AggDemandEconomy.solveAD_UIExtension_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'UI_Rec')
            t1 = time()
            print('Solving UI during recession took ' + mystr(t1-t0) + ' seconds.')
            
            print('Calculating UI recession with AD effects')
            AggDemandEconomy.switch_shock_type("recessionUI")
            AggDemandEconomy.restoreADsolution(name = 'UI_Rec')
            [recession_UI_results_AD,recession_UI_all_results_AD] = runExperimentsAllRecessions(recession_UI_changes)
            saveAsPickleUnderVarName(recession_UI_all_results_AD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_UI_results_AD,figs_dir,locals())
        
        if Run_1stRoundAD:
            # Solving UI under Agg Multiplier   
            t0 = time()
            AggDemandEconomy.switch_shock_type("recessionUI")
            AggDemandEconomy.solveAD_UIExtension_Recession(num_max_iterations=1,convergence_cutoff=convergence_tol_solvingAD, name = 'UI_Rec_1stRoundAD')
            t1 = time()
            print('Solving recession UI 1st round AD took ' + mystr(t1-t0) + ' seconds.')
           
            print('Calculating UI Recession with 1st round AD effects')
            AggDemandEconomy.switch_shock_type("recessionUI")
            AggDemandEconomy.restoreADsolution(name = 'UI_Rec_1stRoundAD')
            [recession_UI_results_firstRoundAD,recession_UI_all_results_firstRoundAD] = runExperimentsAllRecessions(recession_UI_changes)
            saveAsPickleUnderVarName(recession_UI_all_results_firstRoundAD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_UI_results_firstRoundAD,figs_dir,locals())
        
    #%% 
    
        
    if Run_TaxCut_Recession:

        
        if Run_NonAD:
            print('Calculating tax cut recession with no AD effects')
            AggDemandEconomy.switch_shock_type("recessionTaxCut")
            AggDemandEconomy.solve()
            [recession_TaxCut_results,recession_TaxCut_all_results] = runExperimentsAllRecessions(recession_TaxCut_changes)
            saveAsPickleUnderVarName(recession_TaxCut_all_results,figs_dir,locals())
            saveAsPickleUnderVarName(recession_TaxCut_results,figs_dir,locals())
        
        if Run_AD:  
            # Solving tax cut during recession under Agg Multiplier  
            t0 = time()
            AggDemandEconomy.switch_shock_type("recessionTaxCut")
            AggDemandEconomy.solveAD_Recession_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_TaxCut')
            t1 = time()
            print('Solving payroll tax cut during recession took ' + mystr(t1-t0) + ' seconds.')
            
            print('Calculating tax cut recession with AD effects')
            AggDemandEconomy.switch_shock_type("recessionTaxCut")
            AggDemandEconomy.restoreADsolution(name = 'Recession_TaxCut')
            [recession_TaxCut_results_AD,recession_TaxCut_all_results_AD] = runExperimentsAllRecessions(recession_TaxCut_changes)
            saveAsPickleUnderVarName(recession_TaxCut_all_results_AD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_TaxCut_results_AD,figs_dir,locals())
        
        if Run_1stRoundAD:
            # Solving recession under Agg Multiplier   
            t0 = time()
            AggDemandEconomy.switch_shock_type("recessionTaxCut")
            AggDemandEconomy.solveAD_Recession_TaxCut(num_max_iterations=1,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_TaxCut_1stRoundAD')
            t1 = time()
            print('Solving payroll tax cut during recession with 1st round AD took ' + mystr(t1-t0) + ' seconds.')
           
            print('Calculating tax cut recession with 1st round AD effects')
            AggDemandEconomy.switch_shock_type("recessionTaxCut")
            AggDemandEconomy.restoreADsolution(name = 'Recession_TaxCut_1stRoundAD')
            [recession_TaxCut_results_firstRoundAD,recession_TaxCut_all_results_firstRoundAD] = runExperimentsAllRecessions(recession_TaxCut_changes)
            saveAsPickleUnderVarName(recession_TaxCut_all_results_firstRoundAD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_TaxCut_results_firstRoundAD,figs_dir,locals())
                    
        
    #%% Plotting
        

        
    if Make_Plots:
        
        max_T = 15
        x_axis = np.arange(1,max_T+1)
        
        folder_AD           = figs_dir #'./Figures/FullRun_AD025/' #
        folder_base         = figs_dir
        folder_noAD         = figs_dir
        folder_firstroundAD = figs_dir
        
        
        base_results                        = loadPickle('base_results',folder_base,locals())

        recession_results                   = loadPickle('recession_results',folder_noAD,locals())
        recession_results_AD                = loadPickle('recession_results_AD',folder_AD,locals())
        recession_results_firstRoundAD      = loadPickle('recession_results_firstRoundAD',folder_firstroundAD,locals())
        
        recession_UI_results                = loadPickle('recession_UI_results',folder_noAD,locals())       
        recession_UI_results_AD             = loadPickle('recession_UI_results_AD',folder_AD,locals())
        recession_UI_results_firstRoundAD   = loadPickle('recession_UI_results_firstRoundAD',folder_firstroundAD,locals())
        
        recession_Check_results                = loadPickle('recession_Check_results',folder_noAD,locals())       
        recession_Check_results_AD             = loadPickle('recession_Check_results_AD',folder_AD,locals())
        recession_Check_results_firstRoundAD   = loadPickle('recession_Check_results_firstRoundAD',folder_firstroundAD,locals())
        
        recession_TaxCut_results                = loadPickle('recession_TaxCut_results',folder_noAD,locals())
        recession_TaxCut_results_AD             = loadPickle('recession_TaxCut_results_AD',folder_AD,locals())
        recession_TaxCut_results_firstRoundAD   = loadPickle('recession_TaxCut_results_firstRoundAD',folder_firstroundAD,locals())
              
        
       
        
        #%% Multipliers
        
        NPV_AddInc_UI_Rec                       = getSimulationDiff(recession_results,recession_UI_results,'NPV_AggIncome') # Policy expenditure
        NPV_Multiplier_UI_Rec                   = getNPVMultiplier(recession_results,               recession_UI_results,               NPV_AddInc_UI_Rec)
        NPV_Multiplier_UI_Rec_AD                = getNPVMultiplier(recession_results_AD,            recession_UI_results_AD,            NPV_AddInc_UI_Rec)
        NPV_Multiplier_UI_Rec_firstRoundAD      = getNPVMultiplier(recession_results_firstRoundAD,  recession_UI_results_firstRoundAD,  NPV_AddInc_UI_Rec)
        
        
        NPV_AddInc_Rec_TaxCut                   = getSimulationDiff(recession_results,recession_TaxCut_results,'NPV_AggIncome')
        NPV_Multiplier_Rec_TaxCut               = getNPVMultiplier(recession_results,               recession_TaxCut_results,               NPV_AddInc_Rec_TaxCut)
        NPV_Multiplier_Rec_TaxCut_AD            = getNPVMultiplier(recession_results_AD,            recession_TaxCut_results_AD,            NPV_AddInc_Rec_TaxCut)
        NPV_Multiplier_Rec_TaxCut_firstRoundAD  = getNPVMultiplier(recession_results_firstRoundAD,  recession_TaxCut_results_firstRoundAD,  NPV_AddInc_Rec_TaxCut)
       
        NPV_AddInc_Rec_Check                    = getSimulationDiff(recession_results,recession_Check_results,'NPV_AggIncome') 
        NPV_Multiplier_Rec_Check                = getNPVMultiplier(recession_results,               recession_Check_results,               NPV_AddInc_Rec_Check)
        NPV_Multiplier_Rec_Check_AD             = getNPVMultiplier(recession_results_AD,            recession_Check_results_AD,            NPV_AddInc_Rec_Check)
        NPV_Multiplier_Rec_Check_firstRoundAD   = getNPVMultiplier(recession_results_firstRoundAD,  recession_Check_results_firstRoundAD,  NPV_AddInc_Rec_Check)
                
        print('NPV Multiplier UI recession no AD: \t\t',mystr(NPV_Multiplier_UI_Rec[-1]))
        print('NPV Multiplier UI recession with AD: \t\t',mystr(NPV_Multiplier_UI_Rec_AD[-1]))
        print('NPV Multiplier UI recession 1st round AD: \t',mystr(NPV_Multiplier_UI_Rec_firstRoundAD[-1]))
        print('')
        
        print('NPV Multiplier tax cut recession no AD: \t',mystr(NPV_Multiplier_Rec_TaxCut[-1]))
        print('NPV Multiplier tax cut recession with AD: \t',mystr(NPV_Multiplier_Rec_TaxCut_AD[-1]))
        print('NPV Multiplier tax cut recession 1st round AD:  ',mystr(NPV_Multiplier_Rec_TaxCut_firstRoundAD[-1]))
        print('')
        
        print('NPV Multiplier check recession no AD: \t\t',mystr(NPV_Multiplier_Rec_Check[-1]))
        print('NPV Multiplier check recession with AD: \t',mystr(NPV_Multiplier_Rec_Check_AD[-1]))
        print('NPV Multiplier check recession 1st round AD: \t',mystr(NPV_Multiplier_Rec_Check_firstRoundAD[-1]))
        print('')
        
        # Multipliers in non-AD are less than 1 -> this is because of deaths!
        
        
        
        # Multiplier plots
        
        #Period
        AddInc_UI_Rec       = getSimulationDiff(recession_results,recession_UI_results,'AggIncome')
        AddInc_Rec_TaxCut   = getSimulationDiff(recession_results,recession_TaxCut_results,'AggIncome')
        AddInc_Rec_Check    = getSimulationDiff(recession_results,recession_Check_results,'AggIncome')
        
        PM_UI_Rec = 1/100*getStimulus(recession_results_AD, recession_UI_results_AD, AddInc_UI_Rec)
        PM_TaxCut_Rec = 1/100*getStimulus(recession_results_AD, recession_TaxCut_results_AD, AddInc_Rec_TaxCut)
        PM_Check_Rec = 1/100*getStimulus(recession_results_AD, recession_Check_results_AD, AddInc_Rec_Check)
        # values of inf nonsensical
        PM_UI_Rec[PM_UI_Rec>1000] = 0
        PM_TaxCut_Rec[PM_TaxCut_Rec>1000] = 0
        PM_Check_Rec[PM_Check_Rec>1000] = 0
        
        max_T = 30
        x_axis = np.arange(1,max_T+1)
        plt.figure(figsize=(15,10))
        plt.title('Period multipliers with AD effects', size=30)
        plt.plot(x_axis,PM_UI_Rec[0:max_T],                  color='blue',linestyle='-')
        plt.plot(x_axis,PM_TaxCut_Rec[0:max_T],              color='red',linestyle='-')
        plt.plot(x_axis,PM_Check_Rec[0:max_T],               color='green',linestyle='-')
        plt.legend(['UI','Tax Cut','Check'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.savefig(figs_dir +'P_multipliers.pdf')
        plt.show()     
        
        #Cumulative
        C_Multiplier_UI_Rec_AD                = getNPVMultiplier(recession_results_AD,            recession_UI_results_AD,            NPV_AddInc_UI_Rec[-1])
        C_Multiplier_Rec_TaxCut_AD            = getNPVMultiplier(recession_results_AD,            recession_TaxCut_results_AD,        NPV_AddInc_Rec_TaxCut[-1])
        C_Multiplier_Rec_Check_AD             = getNPVMultiplier(recession_results_AD,            recession_Check_results_AD,         NPV_AddInc_Rec_Check[-1])
        max_T = 30
        x_axis = np.arange(1,max_T+1)
        plt.figure(figsize=(15,10))
        plt.title('Cummulative multipliers at different horizons with AD effects', size=30)
        plt.plot(x_axis,C_Multiplier_UI_Rec_AD[0:max_T],                  color='blue',linestyle='-')
        plt.plot(x_axis,C_Multiplier_Rec_TaxCut_AD[0:max_T],              color='red',linestyle='-')
        plt.plot(x_axis,C_Multiplier_Rec_Check_AD[0:max_T],               color='green',linestyle='-')
        plt.legend(['UI','Tax Cut','Check'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.savefig(figs_dir +'C_multipliers.pdf')
        plt.show()
            
        #NPV
        max_T = 30
        x_axis = np.arange(1,max_T+1)
        plt.figure(figsize=(15,10))
        plt.title('NPV multipliers at different horizons with AD effects', size=30)
        plt.plot(x_axis,NPV_Multiplier_UI_Rec_AD[0:max_T],                  color='blue',linestyle='-')
        plt.plot(x_axis,NPV_Multiplier_Rec_TaxCut_AD[0:max_T],              color='red',linestyle='-')
        plt.plot(x_axis,NPV_Multiplier_Rec_Check_AD[0:max_T],               color='green',linestyle='-')
        plt.legend(['UI','Tax Cut','Check'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.savefig(figs_dir +'NPV_multipliers.pdf')
        plt.show()
        
        #%% Income and Consumption paths UI extension
    
        AddCons_UI_Ext_Rec_RelRec               = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggCons')
        AddInc_UI_Ext_Rec_RelRec                = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggIncome')
        
        AddCons_UI_Ext_Rec_RelRec_AD            = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggCons')
        AddInc_UI_Ext_Rec_RelRec_AD             = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggIncome')
 
        AddCons_UI_Ext_Rec_RelRec_firstRoundAD  = getSimulationPercentDiff(recession_results_firstRoundAD,    recession_UI_results_firstRoundAD,'AggCons')
        AddInc_UI_Ext_Rec_RelRec_firstRoundAD   = getSimulationPercentDiff(recession_results_firstRoundAD,    recession_UI_results_firstRoundAD,'AggIncome')       
        
        plt.figure(figsize=(15,10))
        plt.title('Recession + UI extension', size=30)
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec[0:max_T],              color='blue',linestyle='-')
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_AD[0:max_T],           color='blue',linestyle='--')
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_firstRoundAD[0:max_T], color='blue',linestyle=':')
        plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec[0:max_T],             color='red',linestyle='-')
        plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_AD[0:max_T],          color='red',linestyle='--') 
        plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_firstRoundAD[0:max_T],color='red',linestyle=':')
        plt.legend(['Inc, no AD effects','Inc, AD effects','Inc, 1st round AD effects', \
                    'Cons, no AD effects','Cons, AD effects','Cons, 1st round AD effects'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'recession_UI_relrecession.pdf')
        plt.show() 
        
        #%% Income and Consumption paths Tax cut        


        AddCons_Rec_TaxCut_RelRec               = getSimulationPercentDiff(recession_results,               recession_TaxCut_results,'AggCons')
        AddCons_Rec_TaxCut_AD_RelRec            = getSimulationPercentDiff(recession_results_AD,            recession_TaxCut_results_AD,'AggCons')
        AddCons_Rec_TaxCut_firstRoundAD_RelRec  = getSimulationPercentDiff(recession_results_firstRoundAD,  recession_TaxCut_results_firstRoundAD,'AggCons')
        
        AddInc_Rec_TaxCut_RelRec                = getSimulationPercentDiff(recession_results,               recession_TaxCut_results,'AggIncome')
        AddInc_Rec_TaxCut_AD_RelRec             = getSimulationPercentDiff(recession_results_AD,            recession_TaxCut_results_AD,'AggIncome')
        AddInc_Rec_TaxCut_firstRoundAD_RelRec  = getSimulationPercentDiff(recession_results_firstRoundAD,   recession_TaxCut_results_firstRoundAD,'AggIncome')

    
        plt.figure(figsize=(15,10))
        plt.title('Recession + tax cut', size=30)
        plt.plot(x_axis,AddInc_Rec_TaxCut_RelRec[0:max_T],              color='blue',linestyle='-')
        plt.plot(x_axis,AddInc_Rec_TaxCut_AD_RelRec[0:max_T],           color='blue',linestyle='--')
        plt.plot(x_axis,AddInc_Rec_TaxCut_firstRoundAD_RelRec[0:max_T], color='blue',linestyle=':')
        plt.plot(x_axis,AddCons_Rec_TaxCut_RelRec[0:max_T],             color='red',linestyle='-')
        plt.plot(x_axis,AddCons_Rec_TaxCut_AD_RelRec[0:max_T],          color='red',linestyle='--')
        plt.plot(x_axis,AddCons_Rec_TaxCut_firstRoundAD_RelRec[0:max_T],color='red',linestyle=':')
        plt.legend(['Inc, no AD effects','Inc, AD effects','Inc, 1st round AD effects', \
                    'Cons, no AD effects','Cons, AD effects','Cons, 1st round AD effects'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'recession_taxcut_relrecession.pdf')
        plt.show()   
        
        #%% Income and Consumption paths Check experiment        


        AddCons_Rec_Check_RelRec               = getSimulationPercentDiff(recession_results,               recession_Check_results,'AggCons')
        AddInc_Rec_Check_RelRec                = getSimulationPercentDiff(recession_results,               recession_Check_results,'AggIncome')
        
        AddCons_Rec_Check_AD_RelRec            = getSimulationPercentDiff(recession_results_AD,            recession_Check_results_AD,'AggCons')
        AddInc_Rec_Check_AD_RelRec             = getSimulationPercentDiff(recession_results_AD,            recession_Check_results_AD,'AggIncome')

        AddCons_Rec_Check_firstRoundAD_RelRec  = getSimulationPercentDiff(recession_results_firstRoundAD,  recession_Check_results_firstRoundAD,'AggCons')
        AddInc_Rec_Check_firstRoundAD_RelRec   = getSimulationPercentDiff(recession_results_firstRoundAD,  recession_Check_results_firstRoundAD,'AggIncome')

    
        plt.figure(figsize=(15,10))
        plt.title('Recession + Check', size=30)
        plt.plot(x_axis,AddInc_Rec_Check_RelRec[0:max_T],              color='blue',linestyle='-')
        plt.plot(x_axis,AddInc_Rec_Check_AD_RelRec[0:max_T],           color='blue',linestyle='--')
        plt.plot(x_axis,AddInc_Rec_Check_firstRoundAD_RelRec[0:max_T], color='blue',linestyle=':')
        plt.plot(x_axis,AddCons_Rec_Check_RelRec[0:max_T],             color='red',linestyle='-')
        plt.plot(x_axis,AddCons_Rec_Check_AD_RelRec[0:max_T],          color='red',linestyle='--')
        plt.plot(x_axis,AddCons_Rec_Check_firstRoundAD_RelRec[0:max_T],color='red',linestyle=':')
        plt.legend(['Inc, no AD effects','Inc, AD effects','Inc, 1st round AD effects', \
                    'Cons, no AD effects','Cons, AD effects','Cons, 1st round AD effects'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'recession_Check_relrecession.pdf')
        plt.show()        

    
        #%% Function that returns information on a UI experiment with specific RecLength and PolicyLength
        def PlotsforSpecificRecLength(RecLength,Policy): 
            
            # Policy options 'recession_UI' / 'recession_TaxCut' / 'recession_Check'
            
            recession_all_results               = loadPickle('recession_all_results',folder_noAD,locals())
            recession_all_results_AD            = loadPickle('recession_all_results_AD',folder_AD,locals())
            recession_all_results_firstRoundAD  = loadPickle('recession_all_results_firstRoundAD',folder_firstroundAD,locals())
            
            recession_all_policy_results        = loadPickle( Policy + '_all_results',folder_noAD,locals())       
            recession_all_policy_results_AD     = loadPickle(Policy + '_all_results_AD',folder_AD,locals())
            recession_all_policy_results_firstRoundAD= loadPickle(Policy + '_all_results_firstRoundAD',folder_firstroundAD,locals())
            
            
            NPV_AddInc                  = getSimulationDiff(recession_all_results[RecLength-1],recession_all_policy_results[RecLength-1],'NPV_AggIncome') # Policy expenditure
            NPV_Multiplier              = getNPVMultiplier(recession_all_results[RecLength-1],               recession_all_policy_results[RecLength-1],               NPV_AddInc)
            NPV_Multiplier_AD           = getNPVMultiplier(recession_all_results_AD[RecLength-1],            recession_all_policy_results_AD[RecLength-1],            NPV_AddInc)
            NPV_Multiplier_firstRoundAD = getNPVMultiplier(recession_all_results_firstRoundAD[RecLength-1],  recession_all_policy_results_firstRoundAD[RecLength-1],  NPV_AddInc)
            
 
            Multipliers = [NPV_Multiplier,NPV_Multiplier_AD,NPV_Multiplier_firstRoundAD]
            
            PlotEach = False
            
            if PlotEach:
            
                AddCons_RelRec               = getSimulationPercentDiff(recession_all_results[RecLength-1],    recession_all_policy_results[RecLength-1],'AggCons')
                AddInc_RelRec                = getSimulationPercentDiff(recession_all_results[RecLength-1],    recession_all_policy_results[RecLength-1],'AggIncome')
                
                AddCons_RelRec_AD            = getSimulationPercentDiff(recession_all_results_AD[RecLength-1],    recession_all_policy_results_AD[RecLength-1],'AggCons')
                AddInc_RelRec_AD             = getSimulationPercentDiff(recession_all_results_AD[RecLength-1],    recession_all_policy_results_AD[RecLength-1],'AggIncome')
                
   
                plt.figure(figsize=(15,10))
                plt.title('Recession lasts ' + str(RecLength) + 'q', size=30)
                plt.plot(x_axis,AddInc_RelRec[0:max_T],              color='blue',linestyle='-')
                plt.plot(x_axis,AddInc_RelRec_AD[0:max_T],           color='blue',linestyle='--')
                plt.plot(x_axis,AddCons_RelRec[0:max_T],             color='red',linestyle='-')
                plt.plot(x_axis,AddCons_RelRec_AD[0:max_T],          color='red',linestyle='--') 
                plt.legend(['Inc, no AD effects','Inc, AD effects',\
                            'Cons, no AD effects','Cons, AD effects'], fontsize=14)
                plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
                plt.xlabel('quarter', fontsize=18)
                plt.ylabel('% diff. rel. to recession', fontsize=16)
                plt.show() 
                
            
            return Multipliers
        
    #%%
        RecLengthInspect = 21
        Multiplier21qRecession_UI = PlotsforSpecificRecLength(RecLengthInspect,'recession_UI')
        #print('NPV_Multiplier_UI_Rec for 21q recession: ',mystr(Multiplier21qRecession_UI[0]))
        print('NPV_Multiplier_UI_Rec_AD for 21q recession: ',mystr(Multiplier21qRecession_UI[1][-1]))
     #%%    
        Multiplier21qRecession_TaxCut = PlotsforSpecificRecLength(RecLengthInspect,'recession_TaxCut')
        #print('NPV_Multiplier_Rec_TaxCut for 21q recession: ',mystr(Multiplier21qRecession_TaxCut[0]))
        print('NPV_Multiplier_Rec_TaxCut_AD for 21q recession: ',mystr(Multiplier21qRecession_TaxCut[1][-1]))
     #%%   
        Multiplier21qRecession_Check = PlotsforSpecificRecLength(RecLengthInspect,'recession_Check')
        #print('NPV_Multiplier_Rec_Check for 21q recession: ',mystr(Multiplier21qRecession_Check[0]))
        print('NPV_Multiplier_Rec_Check_AD for 21q recession: ',mystr(Multiplier21qRecession_Check[1][-1]))


              
