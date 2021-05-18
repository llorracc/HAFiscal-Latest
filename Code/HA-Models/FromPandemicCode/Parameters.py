import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from HARK.distribution import Uniform
from importlib import reload


figs_dir = './Figures/Check_Experiment/'

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

#$$$$$$$$$$
def makeMacroMrkvArray(Rspell, PolicyUBspell, TaxCutPeriods, TaxCutContinuationProb_Rec, TaxCutContinuationProb_Bas):
    '''
    Make a Markov transition matrix for the macro states
    
    Parameters
    ----------
    Rspell: float
        Expected length of a recession
    PolicyUBspell: float
        Expected length of time the POLICY of entending unemployment benefits remains in place
    TaxCutPeriods : int
        Number of periods for which the tax cut is in place
    '''
    R_persist = 1.-1./Rspell
    PolicyUBpersist = 1.-1./PolicyUBspell # persistence of the extended unemployment POLICY (not the benefits themselves)
    
    MacroMrkvArray = np.array([[1.0,                               0.0,                           0.0,                           0.0],
                               [1-R_persist,                       R_persist,                     0.0,                           0.0],
                               [1-PolicyUBpersist,                 0.0,                           PolicyUBpersist,               0.0],
                               [(1-PolicyUBpersist)*(1-R_persist), (1-PolicyUBpersist)*R_persist, PolicyUBpersist*(1-R_persist), PolicyUBpersist*R_persist]])
    
    
    MacroMrkStates_TaxCut = TaxCutPeriods * 4  # recession and normal, first and 2nd cycle
    MacroTaxCutArray = np.zeros((MacroMrkStates_TaxCut,MacroMrkStates_TaxCut))
    for i in range(2*TaxCutPeriods-1):
        # after the initial cycle of Tax Cut, there is a TaxCutContinuationProb_Rec chance of jumping into the second cycle, but only if recession is active in q8
        if i==TaxCutPeriods-1: 
            MacroTaxCutArray[2*i:2*i+2,2*i+2:2*i+4] = np.array([[TaxCutContinuationProb_Bas * 1,         0.0],
                                                                [TaxCutContinuationProb_Rec *(1-R_persist), TaxCutContinuationProb_Rec *R_persist]])
        else:
            MacroTaxCutArray[2*i:2*i+2,2*i+2:2*i+4] =np.array([[1.0,         0.0],
                                                               [1-R_persist, R_persist]])
                
    MacroMrkvArray = np.concatenate((np.concatenate((MacroMrkvArray,np.zeros((MacroMrkvArray.shape[0],MacroMrkStates_TaxCut))),axis=1),np.concatenate((np.zeros((MacroMrkStates_TaxCut,MacroMrkvArray.shape[0])),MacroTaxCutArray),axis=1)),axis=0)
    
    # From the first Tax cut state, there is a (1-TaxCutContinuationProb_Rec) chance of jumpting back into baseline/recession
    MacroMrkvArray[4+2*(TaxCutPeriods-1):4+2*(TaxCutPeriods-1)+2,0:2] =   np.array([[(1-TaxCutContinuationProb_Bas)*1.0,         0.0],
                                                                                    [(1-TaxCutContinuationProb_Rec)*(1-R_persist), (1-TaxCutContinuationProb_Rec)*R_persist]])
    
    # From the last TaxCut state in the 2nd cycle, one jumps back into baseline / recession
    MacroMrkvArray[-2:,0:2] =  np.array([[1.0,         0.0],
                                        [1-R_persist, R_persist]])
    
    # Add Check Experiment states
    NewDim = len(MacroMrkvArray)+2
    MacroCheckArray = np.zeros((2,NewDim))
    MacroCheckArray[0,0] = 1            #Transisition from check in normal to normal is 100% because check only lasts 1 q
    MacroCheckArray[1,0] = 1-R_persist  #Transisition from check in rec to normal
    MacroCheckArray[1,1] = R_persist    #Transisition from check in rec to rec
    MacroMrkvArray = np.concatenate((MacroMrkvArray,np.zeros((MacroMrkvArray.shape[0],2))),axis=1) #extend to dim 38 in width
    MacroMrkvArray = np.concatenate((MacroMrkvArray,MacroCheckArray),axis = 0) #append check transitions
    
 
    return MacroMrkvArray
    
def makeCondMrkvArrays(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, UBspell_extended, TaxCutPeriods):
    '''
    Make a Markov transition matrix for the micro state conditional on 
    the macro state
    
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
    UBspell_extended: float
        Expected length of unemployment benefits WITH extension (if policy remains in place)
    TaxCutPeriods : int
        Number of periods for which the tax cut is in place
    '''
    U_persist_normal = 1.-1./Uspell_normal
    E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
    UB_persist_normal = 1.-1./UBspell_normal
    U_persist_recession = 1.-1./Uspell_recession
    E_persist_recession = 1.-Urate_recession*(1.-U_persist_recession)/(1.-Urate_recession)
    UBpersist_extended = 1.-1./UBspell_extended # persistence of unemployment benefits when they have been extended

    def small_MrkvArray(e,u,ub):
        small_MrkvArray = np.array([[e,           0.0,       1-e     ],    # Start state: employed
                                     [1-u,        u,         0.0     ],    # Start state: unemployed, no benefits
                                     [1-u,        u*(1-ub),  u*ub   ]])  # Start state: unemployed, benefits
        return small_MrkvArray
    
    # 3x3 lists (rows x colums)
    MrkvArray_normal         = small_MrkvArray(E_persist_normal,    U_persist_normal,    UB_persist_normal)
    MrkvArray_recession      = small_MrkvArray(E_persist_recession, U_persist_recession, UB_persist_normal)
    MrkvArray_normal_exUB    = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBpersist_extended)
    MrkvArray_recession_exUB = small_MrkvArray(E_persist_recession, U_persist_recession, UBpersist_extended)
    
    CondMrkvArrays = [MrkvArray_normal, MrkvArray_recession, MrkvArray_normal_exUB, MrkvArray_recession_exUB]
    CondMrkvArrays += [MrkvArray_normal, MrkvArray_recession]*TaxCutPeriods*2
    CondMrkvArrays += [MrkvArray_normal, MrkvArray_recession]*2 #CheckStates
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

MacroMrkvArray = makeMacroMrkvArray(Rspell, PolicyUBspell, TaxCutPeriods, TaxCutContinuationProb_Rec, TaxCutContinuationProb_Bas)
CondMrkvArrays = makeCondMrkvArrays(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, UBspell_extended, TaxCutPeriods)
MrkvArray = makeFullMrkvArray(MacroMrkvArray, CondMrkvArrays)

num_MrkvStates = MrkvArray[0].shape[0]
num_normal_MrkvStates =3

# Define permanent income growth rates
PermGroFac =       [np.array([1.0]*num_MrkvStates)]
PermGroFac_small = [np.array([1.0]*num_normal_MrkvStates)]

# Define the permanent and transitory shocks 
TranShkStd = [0.1]
PermShkStd = [0.05]

LivPrb       = [1.0-np.array([1/240.0]*num_MrkvStates)]
LivPrb_small = [1.0-np.array([1/240.0]*num_normal_MrkvStates)]
# Make a small state Markov array that is only used when generating the initial distribution of states
MrkvArray_small = list(MrkvArray_i[0:num_normal_MrkvStates,0:num_normal_MrkvStates] for MrkvArray_i in MrkvArray)
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
                "MrkvArray_big" : MrkvArray,
                "MacroMrkvArray" : MacroMrkvArray,
                "CondMrkvArrays" : CondMrkvArrays,
                "Rfree" : np.array(num_normal_MrkvStates*[1.01]),
                "PermGroFac": PermGroFac_small,
                "LivPrb": LivPrb_small,
                "MrkvArray" : MrkvArray_small, # Yes, this is intentional
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
                'Urate_recession' : Urate_recession,
                'Uspell_recession' : Uspell_recession,
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
base_dict = {'RecessionShock' : False,
             'ExtendedUIShock' : False,
             'TaxCutShock' : False,
             'UpdatePrb' : 1.0,
             'Splurge' : Splurge
             }
# Define a dictionary to mutate baseline for the recession
recession_changes = {
             'RecessionShock' : True,
             'ExtendedUIShock' : False,
             'TaxCutShock' : False,
             }
UI_changes = {
             'RecessionShock' : False,
             'ExtendedUIShock' : True,
             'TaxCutShock' : False,
             #EconomyMrkv_init' : [2,2]
             }
recession_UI_changes = {
             'RecessionShock' : True,
             'ExtendedUIShock' : True,
             'TaxCutShock' : False,
             }
TaxCut_changes = {
             'RecessionShock' : False,
             'ExtendedUIShock' : False,
             'TaxCutShock' : True,
             }
recession_TaxCut_changes = {
             'RecessionShock' : True,
             'ExtendedUIShock' : False,
             'TaxCutShock' : True,
             }
Check_changes = {
             'RecessionShock' : False,
             'CheckShock' : True,
             }
recession_Check_changes = {
             'RecessionShock' : True,
             'CheckShock' : True,
             }
sticky_e_changes = {
             'UpdatePrb' : UpdatePrb
             }
frictionless_changes = {
             'UpdatePrb' : 1.0
             }


quick_test = True
if quick_test:
    AgentCountTotal = 200000
    DiscFacCount = 1
    DiscFacDstn = Uniform(DiscFacMean-DiscFacSpread, DiscFacMean+DiscFacSpread).approx(DiscFacCount)
    DiscFacDstns = [DiscFacDstn]
    
# Parameters for AggregateDemandEconomy economy
intercept_prev = np.ones((num_normal_MrkvStates,num_normal_MrkvStates ))    # Intercept of aggregate savings function
slope_prev = np.zeros((num_normal_MrkvStates,num_normal_MrkvStates ))       # Slope of aggregate savings function
intercept_prev_big = np.ones((num_MrkvStates, num_MrkvStates))              # Intercept of aggregate savings function
slope_prev_big = np.zeros((num_MrkvStates, num_MrkvStates))                 # Slope of aggregate savings function
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
                     'MrkvArray' : MrkvArray_small,
                     'MrkvArray_big' : MrkvArray,
                     'intercept_prev_big' : intercept_prev_big,
                     'slope_prev_big' : slope_prev_big,
                     'CgridBase' : CgridBase,
                     'EconomyMrkvNow_init': 0,
                     'act_T' : 400,
                     'TaxCutContinuationProb_Rec' : TaxCutContinuationProb_Rec,
                     'TaxCutContinuationProb_Bas' : TaxCutContinuationProb_Bas
                     }