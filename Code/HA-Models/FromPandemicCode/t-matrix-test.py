import os

os.chdir('./Code/HA-Models/FromPandemicCode')

import numpy as np
from HARK.distribution import DiscreteDistribution
from AggFiscalModel import AggFiscalType, AggregateDemandEconomy
from copy import deepcopy
from Parameters import returnParameters
import matplotlib.pyplot as plt
    
[init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
DiscFacCount, AgentCountTotal, base_dict, num_max_iterations_solvingAD,\
convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
data_EducShares, max_recession_duration, num_experiment_periods,\
recession_changes, UI_changes, recession_UI_changes,\
TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes] = \
    returnParameters(Parametrization='Baseline',OutputFor='_Main.py')
      
agent1 = AggFiscalType(**init_dropout)
agent1.cycles = 0
agent2 = AggFiscalType(**init_highschool)
agent2.cycles = 0
agent3 = AggFiscalType(**init_college)
agent3.cycles = 0
AggDemandEconomy = AggregateDemandEconomy(**init_ADEconomy)
agent1.get_economy_data(AggDemandEconomy)
agent2.get_economy_data(AggDemandEconomy)
agent3.get_economy_data(AggDemandEconomy)
BaseTypeList = [agent1, agent2, agent3]
          
# Fill in the Markov income distribution for each base type
# NOTE: THIS ASSUMES NO LIFECYCLE
IncShkDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([agent1.IncUnemp])])
IncShkDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), \
                                                   [np.array([1.0]), np.array([agent1.IncUnempNoBenefits])])
        
for ThisType in BaseTypeList:
    EmployedIncShkDstn = deepcopy(ThisType.IncShkDstn[0])
    ThisType.IncShkDstn = [[ThisType.IncShkDstn[0]] + \
                           [IncShkDstn_unemp]*UBspell_normal + [IncShkDstn_unemp_nobenefits]]
    ThisType.IncShkDstn_base = ThisType.IncShkDstn
        
    IncShkDstn_recession = [ThisType.IncShkDstn[0]*(2*(num_experiment_periods+1))] 
    ThisType.IncShkDstn_recession = IncShkDstn_recession
    ThisType.IncShkDstn_recessionUI = IncShkDstn_recession
        
    EmployedIncShkDstn.atoms[0][1] = EmployedIncShkDstn.atoms[0][1]*ThisType.TaxCutIncFactor
    TaxCutStatesIncShkDstn = [EmployedIncShkDstn] + \
        [IncShkDstn_unemp]*UBspell_normal + [IncShkDstn_unemp_nobenefits] 
    IncShkDstn_recessionTaxCut = deepcopy(IncShkDstn_recession)
    # Tax states are 2,3 (q1) 4,5 (q2) ... 16,17 (q8)
    for i in range(2*num_base_MrkvStates,18*num_base_MrkvStates,1):
        IncShkDstn_recessionTaxCut[0][i] =  TaxCutStatesIncShkDstn[np.mod(i,4)]
    ThisType.IncShkDstn_recessionTaxCut = IncShkDstn_recessionTaxCut
        
    ThisType.IncShkDstn_recessionCheck = deepcopy(IncShkDstn_recession)
    ThisType.mCount = 5
    ThisType.mFac = 3
    ThisType.mMin = 1e-4
    ThisType.mMax = 10000

# set up sandbox agent
testAgent = agent3
testState = 0

##################################################################################################
# These 3 things that prevented the matrices from matching:

# permanent growth factors to zero
testAgent.PermGroFac = [[1.0, 1.0, 1.0, 1.0]] 
testAgent.PermGroFac_base = 1.0

# initial permanent income to zero
testAgent.pLvlInitMean = 0
testAgent.pLvlInitStd = 0

# testAgent.IncShkDstn[0][0].atoms = testAgent.IncShkDstn[0][0].atoms * 0 + 1
# testAgent.IncShkDstn[0][1].atoms = testAgent.IncShkDstn[0][1].atoms * 0 + 1

# 1-state degenerate Markov arrays
# testAgent.MrkvArray = [np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])]
# testAgent.CondMrkvArrays = [np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])]
# testAgent.MrkvArray = [np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])]
# testAgent.CondMrkvArrays = [np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])]
# testAgent.MrkvArray = [np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]])]
# testAgent.CondMrkvArrays = [np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]])]
# testAgent.MrkvArray = [np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])]
# testAgent.CondMrkvArrays = [np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])]

# 2-state Markov arrays
# testAgent.MrkvArray = [np.array([[0.98150051, 0.01849949, 0, 0], 
#                                  [0.66666667, 0.33333333, 0, 0], 
#                                  [1, 0, 0, 0], [1, 0, 0, 0]])]
# testAgent.CondMrkvArrays = [np.array([[0.98150051, 0.01849949, 0, 0], 
#                                       [0.66666667, 0.33333333, 0, 0], 
#                                  [1, 0, 0, 0], [1, 0, 0, 0]])]

# testAgent.MrkvArray = [np.array([[1.0, 0.0], 
#                                  [1.0, 0.0]])]
# testAgent.CondMrkvArrays = [np.array([[1.0, 0.0], 
#                                  [1.0, 0.0]])]

# testAgent.MrkvArray = [np.array([[0.0, 1.0], 
#                                  [0.0, 1.0]])]
# testAgent.CondMrkvArrays = [np.array([[0.0, 1.0], 
#                                  [0.0, 1.0]])]

testAgent.MrkvArray = [np.array([[0.98150051, 0.01849949], 
                                 [0.66666667, 0.33333333]])]
testAgent.CondMrkvArrays = [np.array([[0.98150051, 0.01849949], 
                                      [0.66666667, 0.33333333]])]

##################################################################################################

# Simulate with Monte Carlo
testAgent.solve()
testAgent.track_vars = ["aLvl"]
testAgent.reset()
testAgent.initialize_sim()
testAgent.AggDemandFac = 1.0
testAgent.RfreeNow = 1.01
testAgent.CaggNow = 1.0
testAgent.Cratio = 1.0
testAgent.simulate()  
print(np.mean(testAgent.state_now["cLvl"]))
print(np.mean(testAgent.state_now["aLvl"]))

# Transition Matrices
testAgent.compute_steady_state()
print(testAgent.C_ss)
print(testAgent.A_ss)

##################################################################################################

# testAgent.calc_transition_matrix_base(state = testState)
mat1 = testAgent.tran_matrix

# testAgent.calc_transition_matrix()
# mat2 = testAgent.tran_matrix

testAgent.MrkvArray = [np.array([[0.0, 1.0], 
                                 [0.0, 1.0]])]
testAgent.CondMrkvArrays = [np.array([[1.0, 0.0], 
                                 [1.0, 0.0]])]

testAgent.compute_steady_state(state = testState)
# print(testAgent.C_ss)
# print(testAgent.A_ss)
mat2 = testAgent.tran_matrix


# print(mat2 - mat1)
