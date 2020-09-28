import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from HARK.distribution import Uniform
from importlib import reload
figs_dir = '../../Figures/'

# Import configurable parameters, and keep them updated through reload
import parameter_config
reload(parameter_config)
from parameter_config import *

###############################################################################

# Size of simulations
AgentCountTotal = 1000000 # Total simulated population
T_sim = 13              # Number of quarters to simulate in counterfactuals

# Basic lifecycle length parameters (don't touch these)
T_cycle = 1

# Define the distribution of the discount factor for each eduation level
DiscFacCount = 7
DiscFacDstn = Uniform(DiscFacMean-DiscFacSpread, DiscFacMean+DiscFacSpread).approx(DiscFacCount)
DiscFacDstns = [DiscFacDstn]

#$$$$$$$$$$
def makeMrkvArray(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, Rspell):
    '''
    Make a Markov transition matrix
    
    Parameters
    ----------
    Urate_normal: float
        Erogodic unemployment rate in normal times
    Uspell_normal: float
        Expected length of unemployment spell in normal times
    UBspell_normal: float
        Expected length of unemployment benefits without extension
    Urate_recession: float
        Erogodic unemployment rate in a recession
    Uspell_recession: float
        Expected length of unemployment spell in a recession
    Rspell: float
        Expected length of a recession
    '''
    U_persist_normal = 1.-1./Uspell_normal
    E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
    UB_persist_normal = 1.-1./UBspell_normal
    U_persist_recession = 1.-1./Uspell_recession
    E_persist_recession = 1.-Urate_recession*(1.-U_persist_recession)/(1.-Urate_recession)
    R_persist = 1.-1./Rspell
    u_n = U_persist_normal
    e_n = E_persist_normal
    ub_n = UB_persist_normal
    u_r = U_persist_recession
    e_r = E_persist_recession
    r = R_persist
    
#    MrkvArray = [np.array([[e_n,           1-e_n,         0.0,       0.0          ],    # Start state: employed,   no recession
#                           [1-u_n,         u_n,           0.0,       0.0          ],    # Start state: enemployed, no recession
#                           [e_n*(1-r),     (1-e_n)*(1-r), e_r*r,    (1-e_r)*r     ],    # Start state: employed,   recession
#                           [(1-u_n)*(1-r), u_n*(1-r),     (1-u_r)*r, u_r*r        ]     # Start state: unemployed, recession
#                           ])]
    
    MrkvArray_normal    = np.array([[e_n,           0.0,           1-e_n     ],    # Start state: employed,   no recession
                                     [1-u_n,         u_n,           0.0       ],    # Start state: enemployed, no recession, no benefits
                                     [1-u_n,         u_n*(1-ub_n),  u_n*ub_n  ]])  # Start state: enemployed, no recession, benefits

    MrkvArray_recession = np.array([[e_r,           0.0,           1-e_r     ],    # Start state: employed,   recession
                                     [1-u_r,         u_r,           0.0       ],    # Start state: enemployed, recession, no benefits
                                     [1-u_r,         u_r*(1-ub_n),  u_r*ub_n  ]])  # Start state: enemployed, recession, benefits

    MrkvArray_1 = np.concatenate((MrkvArray_normal, np.zeros((3,3))),axis=1)      
    MrkvArray_2 = np.concatenate((MrkvArray_recession*(1-r), MrkvArray_recession*r),axis=1) 
    
    MrkvArray = [np.concatenate((MrkvArray_1, MrkvArray_2),axis = 0)]
                            
    return MrkvArray


# Make Markov transition arrays among discrete states in each period of the lifecycle (ACTUAL / SIMULATION)
MrkvArray_real = makeMrkvArray(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession_real, Uspell_recession_real, Rspell_real)
# Make Markov transition arrays among discrete states in each period of the lifecycle (PERCEIVED / SIMULATION)
MrkvArray_pcvd = makeMrkvArray(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession_pcvd, Uspell_recession_pcvd, Rspell_pcvd)
num_MrkvStates = MrkvArray_real[0].shape[0]
num_normal_MrkvStates =3

# Define permanent income growth rates
PermGroFac =       [np.array([1.0]*num_MrkvStates)]
PermGroFac_small = [np.array([1.0]*num_normal_MrkvStates)]

# Define the permanent and transitory shocks 
TranShkStd = [0.1]
PermShkStd = [0.05]

LivPrb       = [1.0-np.array([1.0/240.0]*num_MrkvStates)]
LivPrb_small = [1.0-np.array([1.0/240.0]*num_normal_MrkvStates)]
# Make a small state Markov array that is only used when generating the initial distribution of states
MrkvArray_small = list(MrkvArray_real_i[0:num_normal_MrkvStates,0:num_normal_MrkvStates] for MrkvArray_real_i in MrkvArray_real)
# find intial distribution of states
vals, vecs = np.linalg.eig(np.transpose(MrkvArray_small[0]))
dist = np.abs(np.abs(vals) - 1.)
idx = np.argmin(dist)
init_mrkv_dist = vecs[:,idx].astype(float)/np.sum(vecs[:,idx].astype(float))


# Define a parameter dictionary
init_infhorizon = {"T_cycle": T_cycle,
                'T_sim': 400, #Simulate up to age 400
                'T_age': None,
                'AgentCount': 10000,
                "PermGroFacAgg": PermGroFacAgg,
                "PopGroFac": PopGroFac,
                "CRRA": CRRA,
                "DiscFac": 0.98, # This will be overwritten at type construction
                "Rfree_big" : np.array(num_MrkvStates*[1.01]),
                "PermGroFac_big": PermGroFac,
                "LivPrb_big": LivPrb,
                "MrkvArray_big" : MrkvArray_pcvd,
                "Rfree" : np.array(num_normal_MrkvStates*[1.01]),
                "PermGroFac": PermGroFac_small,
                "LivPrb": LivPrb_small,
                "MrkvArray" : MrkvArray_small, # Yes, this is intentional
                "MrkvArray_pcvd" : MrkvArray_small, # Yes, this is intentional
                "MrkvArray_sim" : MrkvArray_real,
                "BoroCnstArt": 0.0,
                "PermShkStd": PermShkStd,
                "PermShkCount": PermShkCount,
                "TranShkStd": TranShkStd,
                "TranShkCount": TranShkCount,
                "UnempPrb": 0.0, # Unemployment is modeled as a Markov state
                "IncUnemp": IncUnemp,
                "IncUnempNoBenefits": IncUnempNoBenefits,
                "aXtraMin": aXtraMin,
                "aXtraMax": aXtraMax,
                "aXtraCount": aXtraCount,
                "aXtraExtra": aXtraExtra,
                "aXtraNestFac": aXtraNestFac,
                "CubicBool": False,
                "vFuncBool": False,
                'aNrmInitMean': np.log(0.00001), # Initial assets are zero
                'aNrmInitStd': 0.0,
                'pLvlInitMean': pLvlInitMean,
                'pLvlInitStd': pLvlInitStd,
                "MrkvPrbsInit" : np.array(list(init_mrkv_dist) + (num_MrkvStates-num_normal_MrkvStates)*[0.0]),
                'Urate_normal' : Urate_normal,
                'Uspell_normal' : Uspell_normal,
                'UBspell_normal' : UBspell_normal,
                'Urate_recession_real' : Urate_recession_real,
                'Uspell_recession_real' : Uspell_recession_real,
                'Rspell_real' : Rspell_real,
                'Urate_recession_pcvd' : Urate_recession_pcvd,
                'Uspell_recession_pcvd' : Uspell_recession_pcvd,
                'Rspell_pcvd' : Rspell_pcvd,
                'R_shared' : R_shared,
                'track_vars' : []
                }

if R_shared:
    init_infhorizon['T_recession'] = int(Rspell_real)
    
# Population share of each type (at present only one type)    
TypeShares = [1.0]

# Define a dictionary to represent the baseline scenario
base_dict = {'RecessionShock' : False
             }

# Define a dictionary to mutate baseline for the recession
recession_changes = {
        'RecessionShock' : True
        }

quick_test = True
if quick_test:
    AgentCountTotal = 10000
    DiscFacCount = 2
    DiscFacDstn = Uniform(DiscFacMean-DiscFacSpread, DiscFacMean+DiscFacSpread).approx(DiscFacCount)
    DiscFacDstns = [DiscFacDstn]
    init_infhorizon['T_sim'] = 20