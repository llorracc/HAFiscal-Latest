import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from HARK.distribution import Uniform
from importlib import reload


#figs_dir = './Figures/FullRun/'
figs_dir = './Figures/FullRun_PVsame/'


Equalize_PVs = True

try:
    os.mkdir(figs_dir)
except OSError:
    print ("Creation of the directory %s failed" % figs_dir)
else:
    print ("Successfully created the directory %s " % figs_dir)

# Targets in the estimation of the discount factor distributions for each 
# education level. 
# From SCF 2004: [20,40,60,80]-percentiles of the Lorenz curve for liquid wealth
data_LorenzPts_d = np.array([0, 0.01, 0.60, 3.58])    # \
data_LorenzPts_h = np.array([0.06, 0.63, 2.98, 11.6]) # -> units: % 
data_LorenzPts_c = np.array([0.15, 0.92, 3.27, 10.3]) # /
data_LorenzPts = [data_LorenzPts_d, data_LorenzPts_h, data_LorenzPts_c]
data_LorenzPtsAll = np.array([0.03, 0.35, 1.84, 7.42])
# From SCF 2004: Average liquid wealth to permanent income ratio 
data_avgLWPI = np.array([15.7, 47.7, 111])*4 # weighted average of fractions in percent
# From SCF 2004: Total LW over total PI by education group
data_LWoPI = np.array([28.1, 59.6, 162])*4 # units: %
# From SCF 2004: Weighted median of liquid wealth to permanent income ratio
data_medianLWPI = np.array([1.16, 7.55, 28.2])*4 # weighted median of fractions in percent

# Population share of each type
data_EducShares = [0.093, 0.527, 0.38] # Proportion of dropouts, HS grads, college types, SCF 2004 
# Wealth share of each type 
data_WealthShares = np.array([0.008, 0.179, 0.812])*100 # Percentage of total wealth of dropouts, HS grads, college types, SCF 2004 

# Parameters concerning the distribution of discount factors
# Initial values for estimation, taken from pandemic paperCondMrkvArrays_base
DiscFacMeanD = 0.79939338  # Mean intertemporal discount factor for dropout types
DiscFacMeanH = 0.93744293  # Mean intertemporal discount factor for high school types
DiscFacMeanC = 0.98525333  # Mean intertemporal discount factor for college types
DiscFacInit = [DiscFacMeanD, DiscFacMeanH, DiscFacMeanC]
DiscFacSpreadD = 0.2279055
DiscFacSpreadH = 0.06607694
DiscFacSpreadC = 0.01241598 # Half-width of uniform distribution of discount factors

# Define the distribution of the discount factor for each eduation level
DiscFacCount = 7
DiscFacDstnD = Uniform(DiscFacMeanD-DiscFacSpreadD, DiscFacMeanD+DiscFacSpreadD).approx(DiscFacCount)
DiscFacDstnH = Uniform(DiscFacMeanH-DiscFacSpreadH, DiscFacMeanH+DiscFacSpreadH).approx(DiscFacCount)
DiscFacDstnC = Uniform(DiscFacMeanC-DiscFacSpreadC, DiscFacMeanC+DiscFacSpreadC).approx(DiscFacCount)
DiscFacDstns = [DiscFacDstnD, DiscFacDstnH, DiscFacDstnC]

# Parameters concerning Markov transition matrix
#https://www.statista.com/statistics/232942/unemployment-rate-by-level-of-education-in-the-us/
Urate_normal_d = 0.085       # Unemployment rate in normal times, dropouts 2004
Urate_normal_h = 0.044       # Unemployment rate in normal times, highschooler+some college 2004
Urate_normal_c = 0.027       # Unemployment rate in normal times, college 2004

Uspell_normal = 1.5          # Average duration of unemployment spell in normal times, in quarters
UBspell_normal = 2           # Average duration of unemployment benefits in normal times, in quarters

# Recession
Urate_recession_d = 2 * Urate_normal_d # Unemployment rate in recession
Urate_recession_h = 2 * Urate_normal_h 
Urate_recession_c = 2 * Urate_normal_c

Uspell_recession = 4         # Average duration of unemployment spell in recession, in quarters
Rspell = 6                   # Expected length of recession, in quarters. If R_shared = True, must be an integer
R_shared = False             # Indicator for whether the recession shared (True) or idiosyncratic (False)
# UI extension
UBspell_extended = 4         # Average duration of unemployment benefits when extended and assuming policy remains in place, in quarters
PolicyUBspell = 2            # Average duration that policy of extended unemployment benefits is in place
# Tax Cut parameter
PolicyTaxCutspell = 2        # Average duration that policy of payroll tax cuts
if Equalize_PVs:
    TaxCutIncFactor = 1 + 0.02*625/28693       # Amount by which the payroll tax cut increases after-tax income
else:
    TaxCutIncFactor = 1 + 0.02
TaxCutPeriods = 8            # Deterministic duTrueration of tax cut 
TaxCutContinuationProb_Rec = 0.5   # Probability that tax cut is continued after tax cut periods run out, when recession in q8
TaxCutContinuationProb_Bas = 0.0   # Probability that tax cut is continued after tax cut periods run out, when baseline in q8
#Check experiment parameter
if Equalize_PVs:
    CheckStimLvl = 1200/1000 * 625/10178 #1 = 1k, multiplication to get PVs equal
else:
    CheckStimLvl = 1200/1000
CheckStimLvl_PLvl_Cutoff_start = 100/4/1 #100 k yearly income #At this Level of permanent inc, Stimulus beings to fall linearly
CheckStimLvl_PLvl_Cutoff_end = 150/4/1 #150k yearly income #At this Level of permanent inc, Stimulus is zero


UpdatePrb = 0.25    # probability of updating macro state (when sticky expectations is on)
Splurge = 0.32      # amount of income that is splurged

# Basic model parameters: CRRA, growth factors, unemployment parameters (for normal times)
CRRA = 1.0              # Coefficient of relative risk aversion
PopGroFac = 1.0         #1.01**0.25  # Population growth factor
PermGroFacAgg = 1.0     #1.01**0.25 # Technological growth rate or aggregate productivity growth factor
IncUnemp = 0.3              # Unemployment benefits replacement rate (proportion of permanent income)
IncUnempNoBenefits = 0.05   # Unemployment income when benefits run out (proportion of permanent income)

# Parameters concerning the initial distribution of permanent income 
pLvlInitMean_d = np.log(6.2)   # Average quarterly permanent income of "newborn" HS dropout ($1000)
pLvlInitMean_h = np.log(11.1)  # Average quarterly permanent income of "newborn" HS graduate ($1000)
pLvlInitMean_c = np.log(14.5)  # Average quarterly permanent income of "newborn" HS  ($1000)
pLvlInitStd_d  = 0.32          # Standard deviation of initial log permanent income 
pLvlInitStd_h  = 0.42          # Standard deviation of initial log permanent income 
pLvlInitStd_c  = 0.53          # Standard deviation of initial log permanent income 

# Parameters concerning grid sizes: assets, permanent income shocks, transitory income shocks
aXtraMin = 0.001        # Lowest non-zero end-of-period assets above minimum gridpoint
aXtraMax = 40           # Highest non-zero end-of-period assets above minimum gridpoint
aXtraCount = 48         # Base number of end-of-period assets above minimum gridpoints
aXtraExtra = [0.002,0.003] # Additional gridpoints to "force" into the grid
aXtraNestFac = 3        # Exponential nesting factor for aXtraGrid (how dense is grid near zero)
PermShkCount = 7        # Number of points in equiprobable discrete approximation to permanent shock distribution
TranShkCount = 7        # Number of points in equiprobable discrete approximation to transitory shock distribution
T_age = 200             # Kill off agents who have worked for 50 years


# Size of simulations
AgentCountTotal = 10000     # Total simulated population
T_sim = 40                  # Number of quarters to simulate in counterfactuals

# Basic lifecycle length parameters (don't touch these)
T_cycle = 1

# Define grid of aggregate assets to labor
CgridBase = np.array([0.8, 1.0, 1.2])  

num_base_MrkvStates = 2 + UBspell_normal #employed, unemployed with 2 quarters benefits, unemployed with 1 quarter benefit, unemployed no benefits
num_experiment_periods = 20
max_recession_duration = 21

def makeMacroMrkvArray_recession(Rspell, num_experiment_periods):
    R_persist = 1.-1./Rspell
    recession_transitions = np.array([[1.0, 0.0],[1-R_persist, R_persist]])
    MacroMrkvArray = np.zeros((2*(num_experiment_periods+1), 2*(num_experiment_periods+1)))
    MacroMrkvArray[0:2,0:2] = recession_transitions
    for i in np.array(range(num_experiment_periods-1))+1:
        MacroMrkvArray[2*i:2*i+2, 2*i+2:2*i+4] = recession_transitions
    MacroMrkvArray[2*num_experiment_periods:2*num_experiment_periods+2, 0:2] = recession_transitions 
    return MacroMrkvArray

def small_MrkvArray(e,u,ub,transition_ub=True):
    small_MrkvArray = np.zeros((ub+2, ub+2))
    small_MrkvArray[0,0] = e
    small_MrkvArray[0,1] = 1-e
    for i in np.array(range(ub))+1:
        if transition_ub:
            small_MrkvArray[i,i+1] = u
        else:
            small_MrkvArray[i,i] = u
        small_MrkvArray[i,0] = 1-u
    small_MrkvArray[ub+1,ub+1] = u
    small_MrkvArray[ub+1,0] = 1-u
    return small_MrkvArray 

def makeCondMrkvArrays_base(Urate_normal, Uspell_normal, UBspell_normal):
    U_persist_normal = 1.-1./Uspell_normal
    E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
    MrkvArray_normal         = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBspell_normal)
    CondMrkvArrays = [MrkvArray_normal]
    return CondMrkvArrays

def makeCondMrkvArrays_recession(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, num_experiment_periods):
    U_persist_normal = 1.-1./Uspell_normal
    E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
    U_persist_recession = 1.-1./Uspell_recession
    E_persist_recession = 1.-Urate_recession*(1.-U_persist_recession)/(1.-Urate_recession)
    MrkvArray_normal         = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBspell_normal)
    MrkvArray_recession      = small_MrkvArray(E_persist_recession, U_persist_recession, UBspell_normal)
    CondMrkvArrays = [MrkvArray_normal, MrkvArray_recession]*(num_experiment_periods+1)
    return CondMrkvArrays

def makeCondMrkvArrays_recessionUI(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, num_experiment_periods, ExtraUBperiods):
    U_persist_normal = 1.-1./Uspell_normal
    E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
    U_persist_recession = 1.-1./Uspell_recession
    E_persist_recession = 1.-Urate_recession*(1.-U_persist_recession)/(1.-Urate_recession)
    MrkvArray_normal         = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBspell_normal)
    MrkvArray_recession      = small_MrkvArray(E_persist_recession, U_persist_recession, UBspell_normal)
    MrkvArray_normalUI       = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBspell_normal, transition_ub=False)
    MrkvArray_recessionUI    = small_MrkvArray(E_persist_recession, U_persist_recession, UBspell_normal, transition_ub=False)
    CondMrkvArrays = [MrkvArray_normal, MrkvArray_recession] + [MrkvArray_normalUI, MrkvArray_recessionUI]*ExtraUBperiods + [MrkvArray_normal, MrkvArray_recession]*(num_experiment_periods-ExtraUBperiods)
    return CondMrkvArrays


def makeFullMrkvArray(MacroMrkvArray, CondMrkvArrays):
    for i in range(MacroMrkvArray.shape[0]):
        this_row = MacroMrkvArray[i,0]*CondMrkvArrays[0]
        for j in range(MacroMrkvArray.shape[0]-1):
            this_row = np.concatenate((this_row, MacroMrkvArray[i,j+1]*CondMrkvArrays[j+1]),axis=1)
        if i==0:
            FullMrkv = this_row
        else:
            FullMrkv = np.concatenate((FullMrkv, this_row), axis=0)
    return [FullMrkv]

MacroMrkvArray_base = np.array([[1.0]])
CondMrkvArrays_base_d = makeCondMrkvArrays_base(Urate_normal_d, Uspell_normal, UBspell_normal)
CondMrkvArrays_base_h = makeCondMrkvArrays_base(Urate_normal_h, Uspell_normal, UBspell_normal)
CondMrkvArrays_base_c = makeCondMrkvArrays_base(Urate_normal_c, Uspell_normal, UBspell_normal)
MrkvArray_base_d = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_d)
MrkvArray_base_h = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_h)
MrkvArray_base_c = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_c)

MacroMrkvArray_recession = makeMacroMrkvArray_recession(Rspell, num_experiment_periods)
CondMrkvArrays_recession_d = makeCondMrkvArrays_recession(Urate_normal_d, Uspell_normal, UBspell_normal, Urate_recession_d, Uspell_recession, num_experiment_periods)
CondMrkvArrays_recession_h = makeCondMrkvArrays_recession(Urate_normal_h, Uspell_normal, UBspell_normal, Urate_recession_h, Uspell_recession, num_experiment_periods)
CondMrkvArrays_recession_c = makeCondMrkvArrays_recession(Urate_normal_c, Uspell_normal, UBspell_normal, Urate_recession_c, Uspell_recession, num_experiment_periods)
MrkvArray_recession_d = makeFullMrkvArray(MacroMrkvArray_recession, CondMrkvArrays_recession_d)
MrkvArray_recession_h = makeFullMrkvArray(MacroMrkvArray_recession, CondMrkvArrays_recession_h)
MrkvArray_recession_c = makeFullMrkvArray(MacroMrkvArray_recession, CondMrkvArrays_recession_c)

MacroMrkvArray_recessionCheck = MacroMrkvArray_recession
CondMrkvArrays_recessionCheck_d = CondMrkvArrays_recession_d
CondMrkvArrays_recessionCheck_h = CondMrkvArrays_recession_h
CondMrkvArrays_recessionCheck_c = CondMrkvArrays_recession_c
MrkvArray_recessionCheck_d = makeFullMrkvArray(MacroMrkvArray_recessionCheck, CondMrkvArrays_recessionCheck_d)
MrkvArray_recessionCheck_h = makeFullMrkvArray(MacroMrkvArray_recessionCheck, CondMrkvArrays_recessionCheck_h)
MrkvArray_recessionCheck_c = makeFullMrkvArray(MacroMrkvArray_recessionCheck, CondMrkvArrays_recessionCheck_c)

MacroMrkvArray_recessionTaxCut = MacroMrkvArray_recession
CondMrkvArrays_recessionTaxCut_d = CondMrkvArrays_recession_d
CondMrkvArrays_recessionTaxCut_h = CondMrkvArrays_recession_h
CondMrkvArrays_recessionTaxCut_c = CondMrkvArrays_recession_c
MrkvArray_recessionTaxCut_d = makeFullMrkvArray(MacroMrkvArray_recessionTaxCut, CondMrkvArrays_recessionTaxCut_d)
MrkvArray_recessionTaxCut_h = makeFullMrkvArray(MacroMrkvArray_recessionTaxCut, CondMrkvArrays_recessionTaxCut_h)
MrkvArray_recessionTaxCut_c = makeFullMrkvArray(MacroMrkvArray_recessionTaxCut, CondMrkvArrays_recessionTaxCut_c)

MacroMrkvArray_recessionUI = MacroMrkvArray_recession
CondMrkvArrays_recessionUI_d = makeCondMrkvArrays_recessionUI(Urate_normal_d, Uspell_normal, UBspell_normal, Urate_recession_d, Uspell_recession, num_experiment_periods, UBspell_extended-UBspell_normal)
CondMrkvArrays_recessionUI_h = makeCondMrkvArrays_recessionUI(Urate_normal_h, Uspell_normal, UBspell_normal, Urate_recession_h, Uspell_recession, num_experiment_periods, UBspell_extended-UBspell_normal)
CondMrkvArrays_recessionUI_c = makeCondMrkvArrays_recessionUI(Urate_normal_c, Uspell_normal, UBspell_normal, Urate_recession_c, Uspell_recession, num_experiment_periods, UBspell_extended-UBspell_normal)
MrkvArray_recessionUI_d    = makeFullMrkvArray(MacroMrkvArray_recessionUI, CondMrkvArrays_recessionUI_d)
MrkvArray_recessionUI_h    = makeFullMrkvArray(MacroMrkvArray_recessionUI, CondMrkvArrays_recessionUI_h)
MrkvArray_recessionUI_c    = makeFullMrkvArray(MacroMrkvArray_recessionUI, CondMrkvArrays_recessionUI_c)

# Define permanent income growth rates
PermGroFac_base =   [1.0]
PermGroFac_base_d = [1.0 + 0.01421/4]  # From Pandemic paper: avg growth rates during 
PermGroFac_base_h = [1.0 + 0.01812/4]  # working life for each education group 
PermGroFac_base_c = [1.0 + 0.01958/4]

# # Define the permanent and transitory shocks 
# TranShkStd = [0.1]
# PermShkStd = [0.05]
# Variances from Sticky expectations paper: 
TranShkStd = [np.sqrt(0.12)]
PermShkStd = [np.sqrt(0.003)]

Rfree_base = [1.01]
LivPrb_base = [1.0-1/160.0]     # 40 years (160 quarters) working life 
# find intial distribution of states for each education type
vals_d, vecs_d = np.linalg.eig(np.transpose(MrkvArray_base_d[0])) 
dist_d = np.abs(np.abs(vals_d) - 1.)
idx_d = np.argmin(dist_d)
init_mrkv_dist_d = vecs_d[:,idx_d].real/np.sum(vecs_d[:,idx_d].real)

vals_h, vecs_h = np.linalg.eig(np.transpose(MrkvArray_base_h[0])) 
dist_h = np.abs(np.abs(vals_h) - 1.)
idx_h = np.argmin(dist_h)
init_mrkv_dist_h = vecs_h[:,idx_h].real/np.sum(vecs_h[:,idx_h].real)

vals_c, vecs_c = np.linalg.eig(np.transpose(MrkvArray_base_c[0])) 
dist_c = np.abs(np.abs(vals_c) - 1.)
idx_c = np.argmin(dist_c)
init_mrkv_dist_c = vecs_c[:,idx_c].real/np.sum(vecs_c[:,idx_c].real)



# Define a parameter dictionary for dropout type
init_dropout = {"cycles": 0, # 00This will be overwritten at type construction
                "T_cycle": T_cycle,
                'T_sim': 400, #Simhulate up to age 400
                'T_age': T_age,
                'AgentCount': 1000, #number overwritten later
                "PermGroFacAgg": PermGroFacAgg,
                "PopGroFac": PopGroFac,
                "CRRA": CRRA,
                "DiscFac": 0.98, # This will be overwritten at type construction
                "Rfree_base" : Rfree_base,
                "PermGroFac_base": PermGroFac_base_d,
                "LivPrb_base": LivPrb_base,
                "MrkvArray_recession" : MrkvArray_recession_d,
                "MacroMrkvArray_recession" : MacroMrkvArray_recession,
                "CondMrkvArrays_recession" : CondMrkvArrays_recession_d,
                "MrkvArray_recessionUI" : MrkvArray_recessionUI_d,
                "MacroMrkvArray_recessionUI" : MacroMrkvArray_recessionUI,
                "CondMrkvArrays_recessionUI" : CondMrkvArrays_recessionUI_d,
                "MrkvArray_recessionTaxCut" : MrkvArray_recessionTaxCut_d,
                "MacroMrkvArray_recessionTaxCut" : MacroMrkvArray_recessionTaxCut,
                "CondMrkvArrays_recessionTaxCut" : CondMrkvArrays_recessionTaxCut_d,
                "MrkvArray_recessionCheck" : MrkvArray_recessionCheck_d,
                "MacroMrkvArray_recessionCheck" : MacroMrkvArray_recessionCheck,
                "CondMrkvArrays_recessionCheck" : CondMrkvArrays_recessionCheck_d,
                "Rfree" : np.array(num_base_MrkvStates*Rfree_base),
                "PermGroFac": [np.array(PermGroFac_base_d*num_base_MrkvStates)],
                "LivPrb": [np.array(LivPrb_base*num_base_MrkvStates)],
                "MrkvArray_base" : MrkvArray_base_d, 
                "MacroMrkvArray_base" : MacroMrkvArray_base,
                "CondMrkvArrays_base" : CondMrkvArrays_base_d,
                "MrkvArray" : MrkvArray_base_d, 
                "MacroMrkvArray" : MacroMrkvArray_base,
                "CondMrkvArrays" : CondMrkvArrays_base_d,
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
                'pLvlInitMean': pLvlInitMean_d,
                'pLvlInitStd': pLvlInitStd_d,
                "MrkvPrbsInit" : np.array(list(init_mrkv_dist_d)),
                'Urate_normal' : Urate_normal_d,
                'Uspell_normal' : Uspell_normal,
                'UBspell_normal' : UBspell_normal,
                'num_base_MrkvStates' : num_base_MrkvStates,
                'Urate_recession' : Urate_recession_d,
                'Uspell_recession' : Uspell_recession,
                'num_experiment_periods' : num_experiment_periods,
                'Rspell' : Rspell,
                'R_shared' : R_shared,
                'UBspell_extended' : UBspell_extended,
                'PolicyUBspell' : PolicyUBspell,
                'PolicyTaxCutspell' : PolicyTaxCutspell,
                'TaxCutIncFactor' : TaxCutIncFactor,
                'TaxCutPeriods' : TaxCutPeriods,
                'TaxCutContinuationProb_Rec' : TaxCutContinuationProb_Rec,
                'TaxCutContinuationProb_Bas' : TaxCutContinuationProb_Bas,
                'CheckStimLvl' : CheckStimLvl,
                'CheckStimLvl_PLvl_Cutoff_start' : CheckStimLvl_PLvl_Cutoff_start,
                'CheckStimLvl_PLvl_Cutoff_end' : CheckStimLvl_PLvl_Cutoff_end,
                'UpdatePrb' : 1.0,
                'Splurge' : Splurge,
                'track_vars' : [],
                'EducType': 0
                }

adj_highschool = {
    "PermGroFac_base": PermGroFac_base_h,
    "PermGroFac": [np.array(PermGroFac_base_h*num_base_MrkvStates)],
    "MrkvArray_base" : MrkvArray_base_h, 
    "CondMrkvArrays_base" : CondMrkvArrays_base_h,
    "MrkvArray" : MrkvArray_base_h, 
    "CondMrkvArrays" : CondMrkvArrays_base_h,
    "MrkvArray_recession" : MrkvArray_recession_h,
    "CondMrkvArrays_recession" : CondMrkvArrays_recession_h,
    "MrkvArray_recessionUI" : MrkvArray_recessionUI_h,
    "CondMrkvArrays_recessionUI" : CondMrkvArrays_recessionUI_h,
    "MrkvArray_recessionTaxCut" : MrkvArray_recessionTaxCut_h,
    "CondMrkvArrays_recessionTaxCut" : CondMrkvArrays_recessionTaxCut_h,
    "MrkvArray_recessionCheck" : MrkvArray_recessionCheck_h,
    "CondMrkvArrays_recessionCheck" : CondMrkvArrays_recessionCheck_h,  
    'pLvlInitMean': pLvlInitMean_h,
    'pLvlInitStd': pLvlInitStd_h,
    "MrkvPrbsInit" : np.array(list(init_mrkv_dist_h)),
    'Urate_normal' : Urate_normal_h,
    'Urate_recession' : Urate_recession_h,
    'EducType' : 1}
init_highschool = init_dropout.copy()
init_highschool.update(adj_highschool)

adj_college = {
    "PermGroFac_base": PermGroFac_base_c,
    "PermGroFac": [np.array(PermGroFac_base_c*num_base_MrkvStates)],
    "MrkvArray_base" : MrkvArray_base_c, 
    "CondMrkvArrays_base" : CondMrkvArrays_base_c,
    "MrkvArray" : MrkvArray_base_c, 
    "CondMrkvArrays" : CondMrkvArrays_base_c,
    "MrkvArray_recession" : MrkvArray_recession_c,
    "CondMrkvArrays_recession" : CondMrkvArrays_recession_c,
    "MrkvArray_recessionUI" : MrkvArray_recessionUI_c,
    "CondMrkvArrays_recessionUI" : CondMrkvArrays_recessionUI_c,
    "MrkvArray_recessionTaxCut" : MrkvArray_recessionTaxCut_c,
    "CondMrkvArrays_recessionTaxCut" : CondMrkvArrays_recessionTaxCut_c,
    "MrkvArray_recessionCheck" : MrkvArray_recessionCheck_c,
    "CondMrkvArrays_recessionCheck" : CondMrkvArrays_recessionCheck_c, 
    'pLvlInitMean': pLvlInitMean_c,
    'pLvlInitStd': pLvlInitStd_c,
    "MrkvPrbsInit" : np.array(list(init_mrkv_dist_c)),
    'Urate_normal' : Urate_normal_c,
    'Urate_recession' : Urate_recession_c,
    'EducType' : 2}
init_college = init_dropout.copy()
init_college.update(adj_college)

    
# Population share of each type (at present only one type)    
TypeShares = [1.0]

# Define a dictionary to represent the baseline scenario
base_dict = {'shock_type' : "base",
             'UpdatePrb' : 1.0,
             'Splurge' : Splurge
             }
# Define a dictionary to mutate baseline for the recession
recession_changes = {
             'shock_type' : "recession",
             }
UI_changes = {
             'shock_type' : "UI",
             }
recession_UI_changes = {
             'shock_type' : "recessionUI",
             }
TaxCut_changes = {
             'shock_type' : "TaxCut",
             }
recession_TaxCut_changes = {
             'shock_type' : "recessionTaxCut",
             }
Check_changes = {
             'shock_type' : "Check",
             }
recession_Check_changes = {
             'shock_type' : "recessionCheck",
             }
sticky_e_changes = {
             'UpdatePrb' : UpdatePrb
             }
frictionless_changes = {
             'UpdatePrb' : 1.0
             }


    
# Parameters for AggregateDemandEconomy economy
intercept_prev = np.ones((num_base_MrkvStates,num_base_MrkvStates ))    # Intercept of aggregate savings function
slope_prev = np.zeros((num_base_MrkvStates,num_base_MrkvStates ))       # Slope of aggregate savings function
ADelasticity = 0.5                                                      # Elasticity of productivity to consumption

num_max_iterations_solvingAD = 15
convergence_tol_solvingAD = 1E-4
Cfunc_iter_stepsize       = 1

# Make a dictionary to specify a Cobb-Douglas economy
init_ADEconomy = {'intercept_prev': intercept_prev,
                     'slope_prev': slope_prev,
                     'ADelasticity' : 0.0,
                     'demand_ADelasticity' : ADelasticity,
                     'Cfunc_iter_stepsize' : Cfunc_iter_stepsize,
                     'MrkvArray' : MrkvArray_base_h,
                     'MrkvArray_recession' : MrkvArray_recession_h,
                     'MrkvArray_recessionUI' : MrkvArray_recessionUI_h,
                     'MrkvArray_recessionTaxCut' : MrkvArray_recessionTaxCut_h,
                     'MrkvArray_recessionCheck' : MrkvArray_recessionCheck_h,
                     'num_base_MrkvStates' : num_base_MrkvStates,
                     'num_experiment_periods' : num_experiment_periods,
                     "MrkvArray_base" : MrkvArray_base_h, 
                     'CgridBase' : CgridBase,
                     'EconomyMrkvNow_init': 0,
                     'act_T' : 400,
                     'TaxCutContinuationProb_Rec' : TaxCutContinuationProb_Rec,
                     'TaxCutContinuationProb_Bas' : TaxCutContinuationProb_Bas
                     }