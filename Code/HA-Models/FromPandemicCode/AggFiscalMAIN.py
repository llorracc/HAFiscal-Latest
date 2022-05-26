'''
This is the main script for the paper
'''




from Simulate import Simulate


Run_Main            = True
Run_EqualPVs        = True
Run_CRRA_robustness = True



#%% Execute main Simulation

if Run_Main:

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
    Simulate(Run_Dict,figs_dir,Parametrization='Baseline')

    
#%% Execute PV Same run
    
if Run_EqualPVs:
        
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
    Simulate(Run_Dict,figs_dir,Parametrization='Baseline_PVSame')

    
#%% Execute robustness run
        
if Run_CRRA_robustness:

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
    
    figs_dir = './Figures/CRRA2.0_Robustnes/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA2')
            
    figs_dir = './Figures/CRRA2.0_Robustnes_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA2_PVSame')
    
    
    

    
