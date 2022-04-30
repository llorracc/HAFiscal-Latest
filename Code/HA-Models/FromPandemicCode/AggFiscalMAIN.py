'''
This is the main script for the paper
'''


#%% load base parameter file

from Parameters import init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
     DiscFacCount, AgentCountTotal, base_dict, num_max_iterations_solvingAD,\
     convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
     data_EducShares, max_recession_duration, num_experiment_periods,\
     recession_changes, UI_changes, recession_UI_changes,\
     TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes  
     
     
from Simulate import Simulate

#%% Execute main Simulation

Run_Dict = dict()
Run_Dict['Run_Baseline']            = True
Run_Dict['Run_Recession ']          = True
Run_Dict['Run_Check_Recession']     = True
Run_Dict['Run_UB_Ext_Recession']    = True
Run_Dict['Run_TaxCut_Recession']    = True
Run_Dict['Run_Check']               = True
Run_Dict['Run_UB_Ext']              = True
Run_Dict['Run_TaxCut']              = True
Run_Dict['Run_AD ']                 = True
Run_Dict['Run_1stRoundAD']          = True
Run_Dict['Run_NonAD']               = True

figs_dir = './Figures/FullRun/'

Simulate(Run_Dict,init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
     DiscFacCount, AgentCountTotal, base_dict, figs_dir, num_max_iterations_solvingAD,\
     convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
     data_EducShares, max_recession_duration, num_experiment_periods,\
     recession_changes, UI_changes, recession_UI_changes,\
     TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes )

    
#%% Execute PV Same run
    
Run_Dict = dict()
Run_Dict['Run_Baseline']            = True
Run_Dict['Run_Recession ']          = True
Run_Dict['Run_Check_Recession']     = True
Run_Dict['Run_UB_Ext_Recession']    = True
Run_Dict['Run_TaxCut_Recession']    = True
Run_Dict['Run_Check']               = True
Run_Dict['Run_UB_Ext']              = True
Run_Dict['Run_TaxCut']              = True
Run_Dict['Run_AD ']                 = True
Run_Dict['Run_1stRoundAD']          = False
Run_Dict['Run_NonAD']               = True

figs_dir = './Figures/FullRun_PVSame/'

init_dropout['TaxCutIncFactor']     = 1 + 0.02*838/28693;
init_highschool['TaxCutIncFactor']  = 1 + 0.02*838/28693;
init_college['TaxCutIncFactor']     = 1 + 0.02*838/28693;

init_dropout['CheckStimLvl']    = 1200/1000 * 838/10178
init_highschool['CheckStimLvl'] = 1200/1000 * 838/10178
init_college['CheckStimLvl']    = 1200/1000 * 838/10178 


Simulate(Run_Dict,init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
     DiscFacCount, AgentCountTotal, base_dict, figs_dir, num_max_iterations_solvingAD,\
     convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
     data_EducShares, max_recession_duration, num_experiment_periods,\
     recession_changes, UI_changes, recession_UI_changes,\
     TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes )