"""
This file is MNW's version of HA-Fiscal-HANK-SAM.py, which produces a pickled object
with heterogeneous agent SSJs for the HA-Fiscal project. To run it, you should install
a local version of the HARK repo (`pip install . -e` when in the root directory) and
pull down the branch called MonteCarloAttempt. This branch will hopefully soon be merged
into main, and then there will be a HARK release.
"""

import os
import sys
from copy import deepcopy
import numpy as np
from time import time
import pickle
from HARK.distributions import DiscreteDistributionLabeled
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType, markov_constructor_dict
from HARK.Calibration.Income.IncomeProcesses import construct_HANK_lognormal_income_process_unemployment, get_PermShkDstn_from_IncShkDstn_markov, get_TranShkDstn_from_IncShkDstn_markov
from Parameters import returnParameters

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
###############################################################################

"""
Specify the output to be produced, and where to write it.
"""
    
# Name the shocks parameters, their labels for the output file, and whether they are offset in time
shock_params = ["transfers", "Rfree_base", "wage", "tax_rate", "job_find", "DiscFac", "IncUnempExt", "IncUnemp"]
shock_labels = ["transfers", "r", "w", "tau", "eta", "DiscFac", "UI_extend", "UI_rr"]  # used in pickled dictionary
offset_list = [True, True, True, True, True, False, True, True]  # Most are offset in time
    
output_filename = "HA_Fiscal_Jacs_MNW.obj"
    
###############################################################################

"""
Set parameters and make AgentType dictionaries by education level.
"""

# Set some fundamental parameters 
bigT = 300  # Maximum time horizon for the SSJ analysis
dx = 1e-4  # Size of perturbation of parameters in fake news algorithm
states = 4 + 2  # Number of discrete Markov states
job_find = 2/3  # Job-finding probability
EU_prob = 0.0306834  # Calibrated probability of switching to unemployment
job_sep = EU_prob / (1. - job_find)  # Implied job separation probability
tax_rate_SS = 0.3  # Calibrated steady state tax rate
wage_SS = 1.0  # Calibrated steady state wage rate
U_rr = 0.4 * (1.0 - tax_rate_SS) * wage_SS  # Ordinary unemployment benefits
weights_of_educ_types = [0.093, 0.527, 0.38]  # Dropout, high school, college
educ_names = ["dropout", "high school", "college"]

# Load in the parameters from the main parameters file
[init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
DiscFacCount, AgentCountTotal, base_dict, num_max_iterations_solvingAD,\
convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
data_EducShares, max_recession_duration, num_experiment_periods,\
recession_changes, UI_changes, recession_UI_changes,\
TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes] = \
    returnParameters(Parametrization='Baseline',OutputFor='_Main.py')

# Set some parameters to overwrite and extend the ones loaded in above
swap_dict = {
    'PermGroFac': [np.ones(states)],  # No permanent income growth
    'LivPrb': [np.ones(states)*0.99375],  # Constant mortality probability
    'transfers': 0.0,  # Direct government transfer payments
    'tax_rate': [0.3],  # Calibrated steady-state tax rate
    'wage': [1.0],  # Basic default
    'labor': [1.0],  # Basic default
    'IncUnemp': U_rr,  # Unemployment benefit replacement rate for *one* of the couple
    'IncUnempExt': 0.0,  # Extended unemployment benefits for quarters 3 and 4
    'UnempPrb': 0.0,  # Turn off ordinary "unemployment" in transitory shocks
    'job_find': job_find,  # Probability of finding a job if unemployed
    'job_sep': job_sep,  # Probability of separating from a job
    'aXtraMax': 1e4,  # Very high top of assets grid
    'aXtraCount': 200,  # With a bunch more asset points
    'tolerance': 1e-14,  # Set a tight solution tolerance because of very patient agents
    'cycles': 0,  # Specify as infinite horizon
    }

# Give those parameters to each education type
init_dropout.update(swap_dict)
init_highschool.update(swap_dict)
init_college.update(swap_dict)

###############################################################################

"""
Define custom constructor functions for the HANK-SAM model.
"""

# Define a constructor to make a constant Rfree across Markov states
def make_flat_Rfree(T_cycle, Rfree_base):
    return [Rfree_base[t] * np.ones(states) for t in range(T_cycle)]
    

# Define a constructor that makes the six-state Markov matrix for this model
def make_hank_sam_markov_array(job_sep, job_find, T_cycle):
    """
    Construct a constant list of 6x6 Markov arrays, representing transitions among
    (un)employment states. Index z=0 represents employment, and z>0 represents
    being unemployed for z periods (capped at 5). There is a constant probability
    of re-employment, and the probabilities allow agents to become re-employed in
    the same period that they separated from their prior job.
    
    This function is used as the constructor for MrkvArray.

    Parameters
    ----------
    job_sep : float
        Probability of separating from the current job.
    job_find : float
        Probability of finding a new job conditional on being unemployed.
    T_cycle : int
        Number of periods in the sequence or cycle.

    Returns
    -------
    MrkvArray : [np.array]
        Repeated list of T arrays, each representing the same transition probabilities
        among (un)employment states.
    """
    EU = job_sep * (1.0 - job_find)
    EE = 1.0 - EU
    UE = job_find
    UU = 1.0 - UE
    
    MrkvArray_t = np.array([[EE, EU, 0., 0., 0., 0.],
                            [UE, 0., UU, 0., 0., 0.],
                            [UE, 0., 0., UU, 0., 0.],
                            [UE, 0., 0., 0., UU, 0.],
                            [UE, 0., 0., 0., 0., UU],
                            [UE, 0., 0., 0., 0., UU],
                            ])
    MrkvArray = T_cycle * [MrkvArray_t]  # Repeat it T times
    return MrkvArray


# Define a constructor that makes the six-state income process for this model
def make_hank_sam_income_dstn(IncShkDstnEmp, IncUnemp, IncUnempExt, transfers, T_cycle):
    """
    Construct a nested list of income shock distributions. The outer list is over
    periods of the cycle, and the inner list is over discrete Markov states. Index
    z=0 corresponds to ordinary employment, while z>0 represents being unemployed
    for z quarters (capped at 5). When unemployed, the agent gets a fraction of
    their permanent income as benefits in the first two quarters, and "extended"
    unemployment benefits in the third and fourth quarters. Because the agent is
    one member of a couple, the effective replacement rate is averaged with 1.0. 

    Parameters
    ----------
    IncShkDstnEmp : [DiscreteDistributionLabeled]
        Income shock distribution when employed, generated by the standard HANK
        income process constructor. Time-varying list.
    IncUnemp : float
        Unemployment benefit replacement rate for standard benefits.
    IncUnempExt : float
        Unemployment benefit replacement rate for extended benefits.
    transfers : float
        Direct transfer payment in all employment states.
    T_cycle : int
        Number of periods in the sequence or cycle.

    Returns
    -------
    IncShkDstn : [[DiscreteDistributionLabeled]]
        Nested list of income shock distributions by period and discrete Markov state.
    """
    rr_0 = (1. + IncUnemp) / 2. + transfers     # Ordinary unemployment benefits
    rr_1 = (1. + IncUnempExt) / 2. + transfers  # Extended unemployment benefits
    rr_2 = (1. + 0.) / 2. + transfers           # No employment benefits at all
    
    # Make degenerate income shock distributions for unemployment states
    IncShkDstnUnemp_ordinary = DiscreteDistributionLabeled(np.array([1.0]),
                                                           np.array([[1.0], [rr_0]]),
                                                           name="ordinary benefits",
                                                           var_names=["PermShk","TranShk"],
                                                           )
    IncShkDstnUnemp_extended = DiscreteDistributionLabeled(np.array([1.0]),
                                                           np.array([[1.0], [rr_1]]),
                                                           name="extended benefits",
                                                           var_names=["PermShk","TranShk"],
                                                           )
    IncShkDstnUnemp_longterm = DiscreteDistributionLabeled(np.array([1.0]),
                                                           np.array([[1.0], [rr_2]]),
                                                           name="long term unemp",
                                                           var_names=["PermShk","TranShk"],
                                                           )
    
    # Relabel those distributions for shorter typing below
    U0 = IncShkDstnUnemp_ordinary
    U1 = IncShkDstnUnemp_extended
    U2 = IncShkDstnUnemp_longterm
    
    # Construct a nested list of income shock distributions by model period
    IncShkDstn = []
    for t in range(T_cycle):
        # Take this period's employed income distribution and add transfers
        IncShkDstnEmp_t = deepcopy(IncShkDstnEmp[t])
        IncShkDstnEmp_t.atoms[1] += transfers
        
        # Make a list of this period's income distributions by discrete state
        IncShkDstn_t = [IncShkDstnEmp_t, U0, U0, U1, U1, U2]
        
        # Add this period's income distribution to the time-varying list
        IncShkDstn.append(IncShkDstn_t)
        
    return IncShkDstn

###############################################################################

"""
Make baseline AgentType instances for the three education levels.
"""
    
# Set some custom constructor functions for this model
new_constructors = {
    "Rfree" : make_flat_Rfree,
    "IncShkDstnEmp" : construct_HANK_lognormal_income_process_unemployment,
    "IncShkDstn" : make_hank_sam_income_dstn,
    "PermShkDstn" : get_PermShkDstn_from_IncShkDstn_markov,
    "TranShkDstn" : get_TranShkDstn_from_IncShkDstn_markov,
    "MrkvArray" : make_hank_sam_markov_array,
    }
HAF_constructors = markov_constructor_dict.copy()
HAF_constructors.update(new_constructors)

# Give those constructors to each base education type
init_dropout["constructors"] = HAF_constructors
init_highschool["constructors"] = HAF_constructors
init_college["constructors"] = HAF_constructors

# Make a list of three education types
agent_DO = MarkovConsumerType(**init_dropout)
agent_HS = MarkovConsumerType(**init_highschool)
agent_CG = MarkovConsumerType(**init_college)
BaseTypeList = [agent_DO, agent_HS, agent_CG]

###############################################################################

"""
Compute HA-SSJs for consumption and assets for each agent type and each shock,
saving them to (rather large) arrays.
"""
          
# Specify type dimensionality of the model
num_educ_types = len(BaseTypeList)
num_discfacs = len(DiscFacDstns[0].atoms[0])
num_shocks = len(shock_params)

# Initialize arrays to hold HA-SSJs for consumption and assets
Cjac_all = np.zeros((num_educ_types, num_discfacs, num_shocks, bigT, bigT))
Ajac_all = np.zeros((num_educ_types, num_discfacs, num_shocks, bigT, bigT))

# Initialize arrays to hold steady state averages by type
C_ss_all = np.zeros((num_educ_types, num_discfacs))
A_ss_all = np.zeros((num_educ_types, num_discfacs))

# Make grid specifications
cons_grid_spec = {"min": 0.0, "max": 5.0, "N": 201}
asset_grid_spec = {"min": 0.0, "max": 100.0, "N": 301}
z_grid_spec = {"N": states}
hank_sam_grid_specs = {"kNrm": asset_grid_spec, "cNrm": cons_grid_spec, "zPrev": z_grid_spec}

overall_start = time()
print("")  # Skip a line
print("Time to compute some HA-SSJs!")

# Loop over each type of agent (education X discount factor)
for e in range(num_educ_types):
    betas = DiscFacDstns[e].atoms[0]
    ThisType = BaseTypeList[e]
    for d,beta in enumerate(betas):
        type_start = time()
        print("")  # skip a line
        print("Beginning work on agents with " + educ_names[e] + " education and beta={:.4f}".format(beta) + "...")
        
        # Solve the long run model and prepare to calculate steady state
        ThisType.DiscFac = beta
        ThisType.solve()
        ThisType.initialize_sym()
        
        # Calculate steady state assets and consumption for this type
        X = ThisType._simulator
        X.make_transition_matrices(hank_sam_grid_specs, norm="PermShk")
        X.find_steady_state()
        A_dstn = np.dot(X.steady_state_dstn, X.outcome_arrays[0]["aNrm"])
        A_ss = np.dot(A_dstn, X.outcome_grids[0]["aNrm"])
        C_dstn = np.dot(X.steady_state_dstn, X.outcome_arrays[0]["cNrm"])
        C_ss = np.dot(C_dstn, X.outcome_grids[0]["cNrm"])
        C_ss_all[e,d] = C_ss
        A_ss_all[e,d] = A_ss
        del X
        
        print("Solved the agents' long run model in {:.2f}".format(time() - type_start) + " seconds.")
        
        # Loop over each shock variable and compute HA-SSJs for them
        for s,param in enumerate(shock_params):
            shock_start = time()
            Cjac, Ajac = ThisType.make_basic_SSJ(param, ["cNrm", "aNrm"], hank_sam_grid_specs,
                                                 eps=dx, T_max=bigT, norm="PermShk",
                                                 solved=True, offset=offset_list[s])
            Cjac_all[e,d,s,:,:] = Cjac
            Ajac_all[e,d,s,:,:] = Ajac
            print("Calculated HA-SSJ for " + param + " in {:.1f}".format(time() - shock_start) + " seconds.")
                    
        print("Work on agents with " + educ_names[e] + " education and beta={:.4f}".format(beta) + " took {:.1f}".format((time() - type_start)/60) + " minutes.")

print("")  # skip a line
print('Calculating all HA-SSJs took {:.2f}'.format((time() - overall_start)/3600.) + " hours.")

###############################################################################

"""
Aggregate the type-specific HA-SSJs into single arrays, one each for consumption and assets.
"""

# Calculate overall steady state consumption and assets
C_ss = 0.0
A_ss = 0.0
for d in range(num_discfacs): # discount factor
    for e in range(num_educ_types): # education type
        type_weight = DiscFacDstns[e].pmv[d] * weights_of_educ_types[e]
        C_ss += type_weight * C_ss_all[e,d]
        A_ss += type_weight * A_ss_all[e,d]
        
# Calculate overall HA-SSJs for each shock variable
CJAC = np.zeros((num_shocks, bigT, bigT))
AJAC = np.zeros((num_shocks, bigT, bigT))
for s in range(num_shocks):
    for d in range(num_discfacs):
        for e in range(num_educ_types):
            type_weight = DiscFacDstns[e].pmv[d] * weights_of_educ_types[e]
            CJAC[s,:,:] += type_weight * Cjac_all[e,d,s]
            AJAC[s,:,:] += type_weight * Ajac_all[e,d,s]

# Calculate HA-SSJs by education level for each shock variable
CJAC_by_educ = np.zeros((num_shocks, num_educ_types, bigT, bigT))
AJAC_by_educ = np.zeros((num_shocks, num_educ_types, bigT, bigT))
for s in range(len(shock_params)):
    for d in range(num_discfacs): # discount factor
        for e in range(num_educ_types): # education type
            type_weight = DiscFacDstns[e].pmv[d]
            CJAC_by_educ[s,e,:,:] += Cjac_all[e,d,s]
            AJAC_by_educ[s,e,:,:] += Ajac_all[e,d,s]

###############################################################################

"""
Package the HA-SSJs and aggregate steady state values into a nested dictionary.
"""

# Initialize empty dictionaries
CJAC_dict = {}
AJAC_dict = {}
CJAC_dict_educ = {}
AJAC_dict_educ = {}

# Fill in HA-SSJs by shock in the dictionaries
for s,shk in enumerate(shock_params):
    CJAC_dict[shk] = deepcopy(CJAC[s,:,:])
    AJAC_dict[shk] = deepcopy(AJAC[s,:,:])

# Fill in HA-SSJs by shock and education
for e,educ in enumerate(educ_names):    
    CJAC_dict_temp_i = {}
    AJAC_dict_temp_i = {}
    for s,shk in enumerate(shock_params):
        CJAC_dict_temp_i[shk] = deepcopy(CJAC_by_educ[s,e,:,:])
        AJAC_dict_temp_i[shk] = deepcopy(AJAC_by_educ[s,e,:,:])
    
    CJAC_dict_educ[educ] = deepcopy(CJAC_dict_temp_i)
    AJAC_dict_educ[educ] = deepcopy(AJAC_dict_temp_i)

# Put everything into a single dictionary
big_dict_to_save = {
       'C' : CJAC_dict,
       'A' : AJAC_dict,
       'C_by_educ' :  CJAC_dict_educ,
       'A_by_educ' :  AJAC_dict_educ,
       'C_SS' : C_ss,
       'A_SS' : A_ss,
       }

###############################################################################

"""
Write the big results dictionary to a pickled file.
"""
            
with open(output_filename, 'wb') as f:
    pickle.dump(big_dict_to_save, f)
    f.close()
print("Wrote HA-SSJs to " + output_filename + "!")
