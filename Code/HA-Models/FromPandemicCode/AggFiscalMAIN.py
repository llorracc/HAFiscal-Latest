'''
This is the main script for the paper
'''
from Parameters import T_sim, init_infhorizon, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
     recession_Check_changes, figs_dir, num_max_iterations_solvingAD, convergence_tol_solvingAD
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
Run_Check_Recession     = True
Run_UB_Ext_Recession    = True
Run_TaxCut_Recession    = True

Run_AD                  = True
Run_NonAD               = True #whether to run nonAD experiments as well

Make_Plots              = True


#%% 

if __name__ == '__main__':
        
    
    # Setting up AggDemandEconmy
    from setupEconomy import AggDemandEconomy, base_dict_agg, max_recession_duration, output_keys, recession_prob_array, \
                             max_policy_duration, policy_prob_array
        
        
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
        #  running recession with diferent lengths up to 20q then averaging the result
        for t in range(max_recession_duration):
            dictt['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
            dictt['EconomyMrkv_init'][0:t+1] = np.array(dictt['EconomyMrkv_init'][0:t+1]) +1
            #print(dictt['EconomyMrkv_init'])
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
            # Solving recession under Agg Multiplier   
            t0 = time()
            AggDemandEconomy.switch_shock_type("recessionCheck")
            AggDemandEconomy.solveAD_Check_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = 'Recession_Check')
            t1 = time()
            print('Solving recession took ' + mystr(t1-t0) + ' seconds.')
            
            print('Calculating Check Recession with AD effects')
            AggDemandEconomy.switch_shock_type("recessionCheck")
            AggDemandEconomy.restoreADsolution(name = 'Recession_Check')
            [recession_Check_results_AD,recession_Check_all_results_AD] = runExperimentsAllRecessions(recession_Check_changes)
            saveAsPickleUnderVarName(recession_Check_all_results_AD,figs_dir,locals())
            saveAsPickleUnderVarName(recession_Check_results_AD,figs_dir,locals())
    
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
            
                    
        
    #%% Plotting
        

        
    if Make_Plots:
        
        max_T = 20
        x_axis = np.arange(1,21)
        
        folder = figs_dir
        
        
        base_results                = loadPickle('base_results',folder,locals())

        recession_results               = loadPickle('recession_results',folder,locals())
        recession_results_AD            = loadPickle('recession_results_AD',folder,locals())
        
        recession_UI_results                = loadPickle('recession_UI_results',folder,locals())       
        recession_UI_results_AD             = loadPickle('recession_UI_results_AD',folder,locals())
        
        recession_Check_results                = loadPickle('recession_Check_results',folder,locals())       
        recession_Check_results_AD             = loadPickle('recession_Check_results_AD',folder,locals())
        
        recession_TaxCut_results                = loadPickle('recession_TaxCut_results',folder,locals())
        recession_TaxCut_results_AD             = loadPickle('recession_TaxCut_results_AD',folder,locals())
        
        
        
       
        
        #%% Multipliers
        
        NPV_AddInc_UI_Rec                       = getSimulationDiff(recession_results,recession_UI_results,'NPV_AggIncome') # Policy expenditure
        NPV_Multiplier_UI_Rec                   = getNPVMultiplier(recession_results,               recession_UI_results,               NPV_AddInc_UI_Rec)
        NPV_Multiplier_UI_Rec_AD                = getNPVMultiplier(recession_results_AD,            recession_UI_results_AD,            NPV_AddInc_UI_Rec)
         
        NPV_AddInc_Rec_TaxCut                   = getSimulationDiff(recession_results,recession_TaxCut_results,'NPV_AggIncome') 
        NPV_Multiplier_Rec_TaxCut               = getNPVMultiplier(recession_results,               recession_TaxCut_results,               NPV_AddInc_Rec_TaxCut)
        NPV_Multiplier_Rec_TaxCut_AD            = getNPVMultiplier(recession_results_AD,            recession_TaxCut_results_AD,            NPV_AddInc_Rec_TaxCut)
       
        NPV_AddInc_Rec_Check                    = getSimulationDiff(recession_results,recession_Check_results,'NPV_AggIncome') 
        NPV_Multiplier_Rec_Check                = getNPVMultiplier(recession_results,               recession_Check_results,               NPV_AddInc_Rec_Check)
        NPV_Multiplier_Rec_Check_AD             = getNPVMultiplier(recession_results_AD,            recession_Check_results_AD,            NPV_AddInc_Rec_Check)
                
        print('NPV_Multiplier_UI_Rec: ',mystr(NPV_Multiplier_UI_Rec[-1]))
        print('NPV_Multiplier_UI_Rec_AD: ',mystr(NPV_Multiplier_UI_Rec_AD[-1]))
        
        print('NPV_Multiplier_Rec_TaxCut: ',mystr(NPV_Multiplier_Rec_TaxCut[-1]))
        print('NPV_Multiplier_Rec_TaxCut_AD: ',mystr(NPV_Multiplier_Rec_TaxCut_AD[-1]))

        print('NPV_Multiplier_Rec_Check: ',mystr(NPV_Multiplier_Rec_Check[-1]))
        print('NPV_Multiplier_Rec_Check_AD: ',mystr(NPV_Multiplier_Rec_Check_AD[-1]))
        
        #%% Income and Consumption paths UI extension
    
        AddCons_UI_Ext_Rec_RelRec               = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggCons')
        AddInc_UI_Ext_Rec_RelRec                = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggIncome')
        
        AddCons_UI_Ext_Rec_RelRec_AD            = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggCons')
        AddInc_UI_Ext_Rec_RelRec_AD             = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggIncome')
        
        
        plt.figure(figsize=(15,10))
        plt.title('Recession + UI extension', size=30)
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec[0:max_T],              color='blue',linestyle='-')
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_AD[0:max_T],           color='blue',linestyle='--')
        plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec[0:max_T],             color='red',linestyle='-')
        plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_AD[0:max_T],          color='red',linestyle='--') 
        plt.legend(['Inc, no AD effects','Inc, AD effects',\
                    'Cons, no AD effects','Cons, AD effects'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'recession_UI_relrecession.pdf')
        plt.show() 
        
        #%% Income and Consumption paths Tax cut        


        AddCons_Rec_TaxCut_RelRec               = getSimulationPercentDiff(recession_results,               recession_TaxCut_results,'AggCons')
        AddCons_Rec_TaxCut_AD_RelRec            = getSimulationPercentDiff(recession_results_AD,            recession_TaxCut_results_AD,'AggCons')
        
        AddInc_Rec_TaxCut_RelRec                = getSimulationPercentDiff(recession_results,               recession_TaxCut_results,'AggIncome')
        AddInc_Rec_TaxCut_AD_RelRec             = getSimulationPercentDiff(recession_results_AD,            recession_TaxCut_results_AD,'AggIncome')

    
        plt.figure(figsize=(15,10))
        plt.title('Recession + tax cut', size=30)
        plt.plot(x_axis,AddInc_Rec_TaxCut_RelRec[0:max_T],              color='blue',linestyle='-')
        plt.plot(x_axis,AddInc_Rec_TaxCut_AD_RelRec[0:max_T],           color='blue',linestyle='--')
        plt.plot(x_axis,AddCons_Rec_TaxCut_RelRec[0:max_T],             color='red',linestyle='-')
        plt.plot(x_axis,AddCons_Rec_TaxCut_AD_RelRec[0:max_T],          color='red',linestyle='--')
        plt.legend(['Inc, no AD effects','Inc, AD effects',\
                    'Cons, no AD effects','Cons, AD effects'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'recession_taxcut_relrecession.pdf')
        plt.show()   
        
        #%% Income and Consumption paths Tax cut        


        AddCons_Rec_Check_RelRec               = getSimulationPercentDiff(recession_results,               recession_Check_results,'AggCons')
        AddCons_Rec_Check_AD_RelRec            = getSimulationPercentDiff(recession_results_AD,            recession_Check_results_AD,'AggCons')
        
        AddInc_Rec_Check_RelRec                = getSimulationPercentDiff(recession_results,               recession_Check_results,'AggIncome')
        AddInc_Rec_Check_AD_RelRec             = getSimulationPercentDiff(recession_results_AD,            recession_Check_results_AD,'AggIncome')

    
        plt.figure(figsize=(15,10))
        plt.title('Recession + Check', size=30)
        plt.plot(x_axis,AddInc_Rec_Check_RelRec[0:max_T],              color='blue',linestyle='-')
        plt.plot(x_axis,AddInc_Rec_Check_AD_RelRec[0:max_T],           color='blue',linestyle='--')
        plt.plot(x_axis,AddCons_Rec_Check_RelRec[0:max_T],             color='red',linestyle='-')
        plt.plot(x_axis,AddCons_Rec_Check_AD_RelRec[0:max_T],          color='red',linestyle='--')
        plt.legend(['Inc, no AD effects','Inc, AD effects',\
                    'Cons, no AD effects','Cons, AD effects'], fontsize=14)
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.xlabel('quarter', fontsize=18)
        plt.ylabel('% diff. rel. to recession', fontsize=16)
        plt.savefig(figs_dir +'recession_Check_relrecession.pdf')
        plt.show()        


             
    
        #%% Function that returns information on a UI experiment with specific RecLength and PolicyLength
        def PlotsforSpecificRecLength(RecLength,Policy): 
            
            # Policy options 'recession_UI' / 'recession_TaxCut' / 'recession_Check'
            
            recession_all_results               = loadPickle('recession_all_results',folder,locals())
            recession_all_results_AD            = loadPickle('recession_all_results_AD',folder,locals())      
            recession_all_policy_results            = loadPickle( Policy + '_all_results',folder,locals())       
            recession_all_policy_results_AD         = loadPickle(Policy + '_all_results_AD',folder,locals())
            
            
            NPV_AddInc           = getSimulationDiff(recession_all_results[RecLength-1],recession_all_policy_results[RecLength-1],'NPV_AggIncome') # Policy expenditure
            NPV_Multiplier       = getNPVMultiplier(recession_all_results[RecLength-1],               recession_all_policy_results[RecLength-1],               NPV_AddInc)
            NPV_Multiplier_AD    = getNPVMultiplier(recession_all_results_AD[RecLength-1],            recession_all_policy_results_AD[RecLength-1],            NPV_AddInc)
            
 
            Multipliers = [NPV_Multiplier[-1],NPV_Multiplier_AD[-1]]
            
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
        
        
        Multiplier21qRecession_UI = PlotsforSpecificRecLength(21,'recession_UI')
        Multiplier21qRecession_TaxCut = PlotsforSpecificRecLength(21,'recession_TaxCut')
        Multiplier21qRecession_Check = PlotsforSpecificRecLength(21,'recession_Check')
        
   
        #print('NPV_Multiplier_UI_Rec for 21q recession: ',mystr(Multiplier21qRecession_UI[0]))
        print('NPV_Multiplier_UI_Rec_AD for 21q recession: ',mystr(Multiplier21qRecession_UI[1]))
        
        #print('NPV_Multiplier_Rec_TaxCut for 21q recession: ',mystr(Multiplier21qRecession_TaxCut[0]))
        print('NPV_Multiplier_Rec_TaxCut_AD for 21q recession: ',mystr(Multiplier21qRecession_TaxCut[1]))

        #print('NPV_Multiplier_Rec_Check for 21q recession: ',mystr(Multiplier21qRecession_Check[0]))
        print('NPV_Multiplier_Rec_Check_AD for 21q recession: ',mystr(Multiplier21qRecession_Check[1]))


     
        # #%% Plotting long-run multiplier as a function of recession and policy duration
        # max_recession_duration = 21
        # Multipliers = np.zeros((max_recession_duration+1,2))
        # for RecLength in range(1,max_recession_duration+1,1):
        #     Multipliers[RecLength][0:2] = PlotsforSpecificRecandPolicyLength(RecLength)


        # print('NPV_Multiplier_UI_Rec when recession lasts 20q: ',Multipliers[-1][0])
        # print('NPV_Multiplier_UI_Rec_AD: when recession lasts 20q:',Multipliers[-1][1])
              
        # x_axis = np.arange(2,21)
        
        # plt.figure(figsize=(15,10))
        # plt.title('Multipliers as function of Recession length', size=30)
        # plt.plot(x_axis,Multipliers[2:21,0], color='black',)
        # plt.plot(x_axis,Multipliers[2:21,1], color='blue',linestyle='-')
        # plt.legend(['no AD effects',\
        #             'AD effects Rec states'], fontsize=14)
        # plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        # plt.xlabel('recession lasts quarter', fontsize=18)
        # plt.ylabel('Long-run NPV multiplier', fontsize=16)
        # plt.savefig(figs_dir +'Multipliers_RecLength_PolicyLength2.pdf')
        # plt.show() 
            


          
