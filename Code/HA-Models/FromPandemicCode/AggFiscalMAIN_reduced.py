'''
This is the main script for the paper
'''
#from Parameters import returnParameters

import os
import sys

# for output
cwd             = os.getcwd()
folders         = cwd.split(os.path.sep)
top_most_folder = folders[-1]
if top_most_folder == 'FromPandemicCode':
    Abs_Path = cwd
else:
    Abs_Path = cwd + '\\FromPandemicCode'

sys.path.append(Abs_Path)
from Simulate import Simulate
from Output_Results import Output_Results

#%%



Run_Dict = dict()
Run_Dict['Run_Baseline']            = True
Run_Dict['Run_Recession ']          = True
Run_Dict['Run_Check_Recession']     = False
Run_Dict['Run_UB_Ext_Recession']    = True
Run_Dict['Run_TaxCut_Recession']    = False
Run_Dict['Run_Check']               = False
Run_Dict['Run_UB_Ext']              = True
Run_Dict['Run_TaxCut']              = False
Run_Dict['Run_AD ']                 = False
Run_Dict['Run_1stRoundAD']          = False
Run_Dict['Run_NonAD']               = True


    
figs_dir = Abs_Path+'/Figures/Reduced_Run/'    
Simulate(Run_Dict,figs_dir,Parametrization='Reduced_Run')    
Output_Results(Abs_Path+'/Figures/Reduced_Run/',Abs_Path+'/Figures/Reduced_Run/',Abs_Path+'/Tables/Reduced_Run/',Parametrization='Reduced_Run')

