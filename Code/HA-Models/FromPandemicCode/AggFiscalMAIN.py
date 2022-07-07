'''
This is the main script for the paper
'''

#from Parameters import returnParameters
from Simulate import Simulate
from Output_Results import Output_Results


#%%


Run_Main                = False
Run_EqualPVs            = False
Run_ADElas_robustness   = False
Run_CRRA_robustness     = True
Run_Rfree_robustness    = False
Run_Rspell_robustness   = False
Run_LowerUBnoB          = False


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

#%% Execute main Simulation

if Run_Main:

    figs_dir = './Figures/FullRun/'    
    Simulate(Run_Dict,figs_dir,Parametrization='Baseline')
    
    Output_Results('./Figures/FullRun/','./Figures/','./Tables/',Parametrization='Baseline')
    
if Run_EqualPVs:
        
    Run_Dict['Run_1stRoundAD']          = False
    
    figs_dir = './Figures/FullRun_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Baseline_PVSame')
    Output_Results('./Figures/FullRun_PVSame/','./Figures/FullRun_PVSame/','./Tables/',Parametrization='Baseline_PVSame')

#%% 
if Run_ADElas_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/ADElas/'
    Simulate(Run_Dict,figs_dir,Parametrization='ADElas')
    Output_Results('./Figures/ADElas/','./Figures/Robustness_Figs/ADElas/','./Tables/ADElas/',Parametrization='ADElas')
     
    figs_dir = './Figures/ADElas_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='ADElas_PVSame')
    Output_Results('./Figures/ADElas_PVSame/','./Figures/Robustness_Figs/ADElas_PVSame/','./Tables/ADElas_PVSame/',Parametrization='ADElas_PVSame')

    
    
#%% Execute robustness run
        
if Run_CRRA_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/CRRA2.0_Robustnes/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA2')
    Output_Results('./Figures/CRRA2.0_Robustnes/','./Figures/Robustness_Figs/CRRA2/','./Tables/CRRA2/',Parametrization='CRRA2')
     
    figs_dir = './Figures/CRRA2.0_Robustnes_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA2_PVSame')
    Output_Results('./Figures/CRRA2.0_Robustnes_PVSame/','./Figures/Robustness_Figs/CRRA2_PVsame/','./Tables/CRRA2_PVsame/',Parametrization='CRRA2_PVSame')

    #figs_dir = './Figures/CRRA3.0_Robustnes/'
    #imulate(Run_Dict,figs_dir,Parametrization='CRRA3')
    #Output_Results('./Figures/CRRA3.0_Robustnes/','./Figures/Robustness_Figs/CRRA3/','./Tables/CRRA3/',Parametrization='CRRA3')
     
    #figs_dir = './Figures/CRRA3.0_Robustnes_PVSame/'
    #Simulate(Run_Dict,figs_dir,Parametrization='CRRA3_PVSame')
    #Output_Results('./Figures/CRRA3.0_Robustnes_PVSame/','./Figures/Robustness_Figs/CRRA3_PVsame/','./Tables/CRRA3_PVsame/',Parametrization='CRRA3_PVSame')

    
if Run_Rfree_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/Rfree_1005/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1005')
    Output_Results('./Figures/Rfree_1005/','./Figures/Robustness_Figs/Rfree_1005/','./Tables/Rfree_1005/',Parametrization='Rfree_1005')
     
    figs_dir = './Figures/Rfree_1005_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1005_PVSame')
    Output_Results('./Figures/Rfree_1005_PVSame/','./Figures/Robustness_Figs/Rfree_1005_PVsame/','./Tables/Rfree_1005_PVsame/',Parametrization='Rfree_1005_PVSame')

    figs_dir = './Figures/Rfree_1015/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1015')
    Output_Results('./Figures/Rfree_1015/','./Figures/Robustness_Figs/Rfree_1015/','./Tables/Rfree_1015/',Parametrization='Rfree_1015')
     
    figs_dir = './Figures/Rfree_1015_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1015_PVSame')
    Output_Results('./Figures/Rfree_1015_PVSame/','./Figures/Robustness_Figs/Rfree_1015_PVSame/','./Tables/Rfree_1015_PVSame/',Parametrization='Rfree_1015_PVSame')


if Run_Rspell_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/Rspell_4/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rspell_4')
    Output_Results('./Figures/Rspell_4/','./Figures/Robustness_Figs/Rspell_4/','./Tables/Rspell_4/',Parametrization='Rspell_4')
     
    figs_dir = './Figures/Rspell_4_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rspell_4_PVSame')
    Output_Results('./Figures/Rspell_4_PVSame/','./Figures/Robustness_Figs/Rspell_4_PVSame/','./Tables/Rspell_4_PVSame/',Parametrization='Rspell_4_PVSame')


if Run_LowerUBnoB:

    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = './Figures/LowerUBnoB/'
    Simulate(Run_Dict,figs_dir,Parametrization='LowerUBnoB')
    Output_Results('./Figures/LowerUBnoB/','./Figures/Robustness_Figs/LowerUBnoB/','./Tables/LowerUBnoB/',Parametrization='LowerUBnoB')
   