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

def makeMrkvArray(Urate_normal, Uspell_normal, Urate_recession, Uspell_recession, Rspell):
    '''
    Make a Markov transition matrix
    
    Parameters
    ----------
    Urate_normal: float
        Erogodic unemployment rate in normal times
    Uspell_normal: float
        Expected length of unemployment spell in normal times
    Urate_recession: float
        Erogodic unemployment rate in a recession
    Uspell_recession: float
        Expected length of unemployment spell in a recession
    Rspell: float
        Expected length of a recession
    '''
    U_persist_normal = 1.-1./Uspell_normal
    E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
    U_persist_recession = 1.-1./Uspell_recession
    E_persist_recession = 1.-Urate_recession*(1.-U_persist_recession)/(1.-Urate_recession)
    R_persist = 1.-1./Rspell
    u_n = U_persist_normal
    e_n = E_persist_normal
    u_r = U_persist_recession
    e_r = E_persist_recession
    r = R_persist
    
    MrkvArray = [np.array([[e_n,           1-e_n,         0.0,       0.0          ],    # Start state: employed,   no recession
                           [1-u_n,         u_n,           0.0,       0.0          ],    # Start state: enemployed, no recession
                           [e_n*(1-r),     (1-e_n)*(1-r), e_r*r,    (1-e_r)*r     ],    # Start state: employed,   recession
                           [(1-u_n)*(1-r), u_n*(1-r),     (1-u_r)*r, u_r*r        ]     # Start state: unemployed, recession
                           ])]

    return MrkvArray


# Make Markov transition arrays among discrete states in each period of the lifecycle (ACTUAL / SIMULATION)
MrkvArray_real = makeMrkvArray(Urate_normal, Uspell_normal, Urate_recession_real, Uspell_recession_real, Rspell_real)
# Make Markov transition arrays among discrete states in each period of the lifecycle (PERCEIVED / SIMULATION)
MrkvArray_pcvd = makeMrkvArray(Urate_normal, Uspell_normal, Urate_recession_pcvd, Uspell_recession_pcvd, Rspell_pcvd)
num_MrkvStates = MrkvArray_real[0].shape[0]

# Define permanent income growth rates
PermGroFac =       [np.array([1.0]*num_MrkvStates)]
PermGroFac_small = [np.array([1.0]*2)]

# Define the permanent and transitory shocks 
TranShkStd = [0.1]
PermShkStd = [0.05]

LivPrb       = [np.array([1.0/240.0]*num_MrkvStates)]
LivPrb_small = [np.array([1.0/240.0]*2)]
# Make a two state Markov array ("small") that is only used when generating the initial distribution of states
MrkvArray_small = list(MrkvArray_real_i[0:2,0:2] for MrkvArray_real_i in MrkvArray_real)

# Define a parameter dictionary
init_infhorizon = {"cycles" : 1,
                "T_cycle": 1,
                'T_sim': 13,
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
                "Rfree" : np.array(2*[1.01]),
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
                "MrkvPrbsInit" : np.array([1-Urate_normal, Urate_normal] + (num_MrkvStates-2)*[0.0]),
                'Urate_normal' : Urate_normal,
                'Uspell_normal' : Uspell_normal,
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
base_dict = {       }


