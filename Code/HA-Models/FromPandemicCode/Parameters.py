import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from HARK.distribution import Uniform
from importlib import reload


figs_dir = './Figures/Test/'

try:
    os.mkdir(figs_dir)
except OSError:
    print ("Creation of the directory %s failed" % figs_dir)
else:
    print ("Successfully created the directory %s " % figs_dir)



# Parameters concerning the distribution of discount factors
DiscFacMean = 0.986     # Mean intertemporal discount factor 
DiscFacSpread = 0.0183  # Half-width of uniform distribution of discount factors

# Parameters concerning Markov transition matrix

    # Normal
Urate_normal = 0.05          # Unemployment rate in normal times
Uspell_normal = 1.5          # Average duration of unemployment spell in normal times, in quarters
UBspell_normal = 2           # Average duration of unemployment benefits in normal times, in quarters
    # Recession
Urate_recession = 0.1        # Unemployment rate in recession
Uspell_recession = 4         # Average duration of unemployment spell in recession, in quarters
Rspell = 6                   # Expected length of recession, in quarters. If R_shared = True, must be an integer
R_shared = False             # Indicator for whether the recession shared (True) or idiosyncratic (False)
    # UI extension
UBspell_extended = 4         # Average duration of unemployment benefits when extended and assuming policy remains in place, in quarters
PolicyUBspell = 2            # Average duration that policy of extended unemployment benefits is in place
    # Tax Cut parameter
PolicyTaxCutspell = 2        # Average duration that policy of payroll tax cuts
TaxCutIncFactor = 1.02       # Amount by which the payroll tax cut increases after-tax income
TaxCutPeriods = 8            # Deterministic duration of tax cut 
TaxCutContinuationProb_Rec = 0.5   # Probability that tax cut is continued after tax cut periods run out, when recession in q8
TaxCutContinuationProb_Bas = 0.0   # Probability that tax cut is continued after tax cut periods run out, when baseline in q8
    #Check experiment parameter
CheckStimLvl = 1200/15000 #1 = 15k
CheckStimLvl_PLvl_Cutoff_start = 100/4/15 #100 k yearly income #At this Level of permanent inc, Stimulus beings to fall linearly
CheckStimLvl_PLvl_Cutoff_end = 150/4/15 #150k yearly income #At this Level of permanent inc, Stimulus is zero


UpdatePrb = 0.25    # probability of updating macro state (when sticky expectations is on)
Splurge = 0.32      # amount of income that is splurged


# Basic model parameters: CRRA, growth factors, unemployment parameters (for normal times)
CRRA = 1.0              # Coefficient of relative risk aversion
PopGroFac = 1.0 #1.01**0.25  # Population growth factor
PermGroFacAgg = 1.0 #1.01**0.25 # Technological growth rate or aggregate productivity growth factor

IncUnemp = 0.3          # Unemployment benefits replacement rate (proportion of permanent income)
IncUnempNoBenefits = 0.05          # Unemployment income when benefits run out (proportion of permanent income)

# Parameters concerning the initial distribution of permanent income 
pLvlInitMean = 0 
pLvlInitStd = 0.4           # Standard deviation of initial log permanent income 

# Parameters concerning grid sizes: assets, permanent income shocks, transitory income shocks
aXtraMin = 0.001        # Lowest non-zero end-of-period assets above minimum gridpoint
aXtraMax = 40           # Highest non-zero end-of-period assets above minimum gridpoint
aXtraCount = 48         # Base number of end-of-period assets above minimum gridpoints
aXtraExtra = [0.002,0.003] # Additional gridpoints to "force" into the grid
aXtraNestFac = 3        # Exponential nesting factor for aXtraGrid (how dense is grid near zero)
PermShkCount = 7        # Number of points in equiprobable discrete approximation to permanent shock distribution
TranShkCount = 7        # Number of points in equiprobable discrete approximation to transitory shock distribution


# Size of simulations
AgentCountTotal = 50000 # Total simulated population
T_sim = 80              # Number of quarters to simulate in counterfactuals

# Basic lifecycle length parameters (don't touch these)
T_cycle = 1

# Define the distribution of the discount factor for each eduation level
DiscFacCount = 7
DiscFacDstn = Uniform(DiscFacMean-DiscFacSpread, DiscFacMean+DiscFacSpread).approx(DiscFacCount)
DiscFacDstns = [DiscFacDstn]

# Define grid of aggregate assets to labor
CgridBase = np.array([0.8,0.9,0.98,1.0,1.02,1.1,1.2])  

num_base_MrkvStates = 2+ UBspell_normal #employed, unemployed with 2 quarters benefits, unemployed with 1 quarter benefit, unemployed no benefits
num_experiment_periods = 20

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

# def makeMacroMrkvArray(Rspell, PolicyUBspell, TaxCutPeriods, TaxCutContinuationProb_Rec, TaxCutContinuationProb_Bas):
#     '''
#     Make a Markov transition matrix for the macro states
    
#     Parameters
#     ----------
#     Rspell: float
#         Expected length of a recession
#     PolicyUBspell: float
#         Expected length of time the POLICY of entending unemployment benefits remains in place
#     TaxCutPeriods : int
#         Number of periods for which the tax cut is in place
#     '''
#     R_persist = 1.-1./Rspell
#     PolicyUBpersist = 1.-1./PolicyUBspell # persistence of the extended unemployment POLICY (not the benefits themselves)
    
#     MacroMrkvArray = np.array([[1.0,                               0.0,                           0.0,                           0.0],
#                                [1-R_persist,                       R_persist,                     0.0,                           0.0],
#                                [1-PolicyUBpersist,                 0.0,                           PolicyUBpersist,               0.0],
#                                [(1-PolicyUBpersist)*(1-R_persist), (1-PolicyUBpersist)*R_persist, PolicyUBpersist*(1-R_persist), PolicyUBpersist*R_persist]])
    
    
#     MacroMrkStates_TaxCut = TaxCutPeriods * 4  # recession and normal, first and 2nd cycle
#     MacroTaxCutArray = np.zeros((MacroMrkStates_TaxCut,MacroMrkStates_TaxCut))
#     for i in range(2*TaxCutPeriods-1):
#         # after the initial cycle of Tax Cut, there is a TaxCutContinuationProb_Rec chance of jumping into the second cycle, but only if recession is active in q8
#         if i==TaxCutPeriods-1: 
#             MacroTaxCutArray[2*i:2*i+2,2*i+2:2*i+4] = np.array([[TaxCutContinuationProb_Bas * 1,         0.0],
#                                                                 [TaxCutContinuationProb_Rec *(1-R_persist), TaxCutContinuationProb_Rec *R_persist]])
#         else:
#             MacroTaxCutArray[2*i:2*i+2,2*i+2:2*i+4] =np.array([[1.0,         0.0],
#                                                                [1-R_persist, R_persist]])
                
#     MacroMrkvArray = np.concatenate((np.concatenate((MacroMrkvArray,np.zeros((MacroMrkvArray.shape[0],MacroMrkStates_TaxCut))),axis=1),np.concatenate((np.zeros((MacroMrkStates_TaxCut,MacroMrkvArray.shape[0])),MacroTaxCutArray),axis=1)),axis=0)
    
#     # From the first Tax cut state, there is a (1-TaxCutContinuationProb_Rec) chance of jumpting back into baseline/recession
#     MacroMrkvArray[4+2*(TaxCutPeriods-1):4+2*(TaxCutPeriods-1)+2,0:2] =   np.array([[(1-TaxCutContinuationProb_Bas)*1.0,         0.0],
#                                                                                     [(1-TaxCutContinuationProb_Rec)*(1-R_persist), (1-TaxCutContinuationProb_Rec)*R_persist]])
    
#     # From the last TaxCut state in the 2nd cycle, one jumps back into baseline / recession
#     MacroMrkvArray[-2:,0:2] =  np.array([[1.0,         0.0],
#                                         [1-R_persist, R_persist]])
    
#     # Add Check Experiment states
#     NewDim = len(MacroMrkvArray)+2
#     MacroCheckArray = np.zeros((2,NewDim))
#     MacroCheckArray[0,0] = 1            #Transisition from check in normal to normal is 100% because check only lasts 1 q
#     MacroCheckArray[1,0] = 1-R_persist  #Transisition from check in rec to normal
#     MacroCheckArray[1,1] = R_persist    #Transisition from check in rec to rec
#     MacroMrkvArray = np.concatenate((MacroMrkvArray,np.zeros((MacroMrkvArray.shape[0],2))),axis=1) #extend to dim 38 in width
#     MacroMrkvArray = np.concatenate((MacroMrkvArray,MacroCheckArray),axis = 0) #append check transitions
    
 
#     return MacroMrkvArray
    
# def makeCondMrkvArrays(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, UBspell_extended, TaxCutPeriods):
#     '''
#     Make a Markov transition matrix for the micro state conditional on 
#     the macro state
    
#     Parameters
#     ----------
#     Urate_normal: float
#         Erogodic unemployment rate in normal times
#     Uspell_normal: float
#         Expected length of unemployment spell in normal times
#     UBspell_normal: float
#         Expected length of unemployment benefits without extension
#     Urate_recession: float
#         Erogodic unemployment rate in a recession
#     Uspell_recession: float
#         Expected length of unemployment spell in a recession
#     UBspell_extended: float
#         Expected length of unemployment benefits WITH extension (if policy remains in place)
#     TaxCutPeriods : int
#         Number of periods for which the tax cut is in place
#     '''
#     U_persist_normal = 1.-1./Uspell_normal
#     E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
#     UB_persist_normal = 1.-1./UBspell_normal
#     U_persist_recession = 1.-1./Uspell_recession
#     E_persist_recession = 1.-Urate_recession*(1.-U_persist_recession)/(1.-Urate_recession)
#     UBpersist_extended = 1.-1./UBspell_extended # persistence of unemployment benefits when they have been extended

#     def small_MrkvArray(e,u,ub):
#         small_MrkvArray = np.array([[e,           0.0,       1-e     ],    # Start state: employed
#                                      [1-u,        u,         0.0     ],    # Start state: unemployed, no benefits
#                                      [1-u,        u*(1-ub),  u*ub   ]])  # Start state: unemployed, benefits
#         return small_MrkvArray
    
#     # 3x3 lists (rows x colums)
#     MrkvArray_normal         = small_MrkvArray(E_persist_normal,    U_persist_normal,    UB_persist_normal)
#     MrkvArray_recession      = small_MrkvArray(E_persist_recession, U_persist_recession, UB_persist_normal)
#     MrkvArray_normal_exUB    = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBpersist_extended)
#     MrkvArray_recession_exUB = small_MrkvArray(E_persist_recession, U_persist_recession, UBpersist_extended)
    
#     CondMrkvArrays = [MrkvArray_normal, MrkvArray_recession, MrkvArray_normal_exUB, MrkvArray_recession_exUB]
#     CondMrkvArrays += [MrkvArray_normal, MrkvArray_recession]*TaxCutPeriods*2
#     CondMrkvArrays += [MrkvArray_normal, MrkvArray_recession]*2 #CheckStates
#     return CondMrkvArrays

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
CondMrkvArrays_base = makeCondMrkvArrays_base(Urate_normal, Uspell_normal, UBspell_normal)
MrkvArray_base = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base)

MacroMrkvArray_recession = makeMacroMrkvArray_recession(Rspell, num_experiment_periods)
CondMrkvArrays_recession = makeCondMrkvArrays_recession(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, num_experiment_periods)
MrkvArray_recession = makeFullMrkvArray(MacroMrkvArray_recession, CondMrkvArrays_recession)

MacroMrkvArray_recessionUI = makeMacroMrkvArray_recession(Rspell, num_experiment_periods)
CondMrkvArrays_recessionUI = makeCondMrkvArrays_recessionUI(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, num_experiment_periods, UBspell_extended-UBspell_normal)
MrkvArray_recessionUI = makeFullMrkvArray(MacroMrkvArray_recessionUI, CondMrkvArrays_recessionUI)


# Define permanent income growth rates
PermGroFac_base =   [1.0]

# Define the permanent and transitory shocks 
TranShkStd = [0.1]
PermShkStd = [0.05]
Rfree_base = [1.01]

LivPrb_base     = [1.0-1/240.0]
# find intial distribution of states
vals, vecs = np.linalg.eig(np.transpose(MrkvArray_base[0]))
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
                "Rfree_base" : Rfree_base,
                "PermGroFac_base": PermGroFac_base,
                "LivPrb_base": LivPrb_base,
                "MrkvArray_recession" : MrkvArray_recession,
                "MacroMrkvArray_recession" : MacroMrkvArray_recession,
                "CondMrkvArrays_recession" : CondMrkvArrays_recession,
                "MrkvArray_recessionUI" : MrkvArray_recessionUI,
                "MacroMrkvArray_recessionUI" : MacroMrkvArray_recessionUI,
                "CondMrkvArrays_recessionUI" : CondMrkvArrays_recessionUI,
                "Rfree" : np.array(num_base_MrkvStates*Rfree_base),
                "PermGroFac": [np.array(PermGroFac_base*num_base_MrkvStates)],
                "LivPrb": [np.array(LivPrb_base*num_base_MrkvStates)],
                "MrkvArray_base" : MrkvArray_base, 
                "MacroMrkvArray_base" : MacroMrkvArray_base,
                "CondMrkvArrays_base" : CondMrkvArrays_base,
                "MrkvArray" : MrkvArray_base, 
                "MacroMrkvArray" : MacroMrkvArray_base,
                "CondMrkvArrays" : CondMrkvArrays_base,
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
                "MrkvPrbsInit" : np.array(list(init_mrkv_dist)),
                'Urate_normal' : Urate_normal,
                'Uspell_normal' : Uspell_normal,
                'UBspell_normal' : UBspell_normal,
                'num_base_MrkvStates' : num_base_MrkvStates,
                'Urate_recession' : Urate_recession,
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
                'track_vars' : []
                }

if R_shared:
    init_infhorizon['T_recession'] = int(Rspell)
    
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


quick_test = True
if quick_test:
    AgentCountTotal = 40000
    DiscFacCount = 1
    DiscFacDstn = Uniform(DiscFacMean-DiscFacSpread, DiscFacMean+DiscFacSpread).approx(DiscFacCount)
    DiscFacDstns = [DiscFacDstn]
    
# Parameters for AggregateDemandEconomy economy
intercept_prev = np.ones((num_base_MrkvStates,num_base_MrkvStates ))    # Intercept of aggregate savings function
slope_prev = np.zeros((num_base_MrkvStates,num_base_MrkvStates ))       # Slope of aggregate savings function
ADelasticity = 0.50                                                         # Elasticity of productivity to consumption

num_max_iterations_solvingAD = 15
convergence_tol_solvingAD = 1E-5
Cfunc_iter_stepsize       = 0.75

# Make a dictionary to specify a Cobb-Douglas economy
init_ADEconomy = {'intercept_prev': intercept_prev,
                     'slope_prev': slope_prev,
                     'ADelasticity' : 0.0,
                     'demand_ADelasticity' : ADelasticity,
                     'Cfunc_iter_stepsize' : Cfunc_iter_stepsize,
                     'MrkvArray' : MrkvArray_base,
                     'MrkvArray_recession' : MrkvArray_recession,
                     'MrkvArray_recessionUI' : MrkvArray_recessionUI,
                     'num_base_MrkvStates' : num_base_MrkvStates,
                     'num_experiment_periods' : num_experiment_periods,
                     "MrkvArray_base" : MrkvArray_base, 
                     'CgridBase' : CgridBase,
                     'EconomyMrkvNow_init': 0,
                     'act_T' : 400,
                     'TaxCutContinuationProb_Rec' : TaxCutContinuationProb_Rec,
                     'TaxCutContinuationProb_Bas' : TaxCutContinuationProb_Bas
                     }