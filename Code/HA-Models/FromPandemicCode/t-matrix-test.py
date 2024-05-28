import os

os.chdir('./Code/HA-Models/FromPandemicCode')

import numpy as np
from HARK.distribution import DiscreteDistribution
from AggFiscalModel import AggFiscalType, AggregateDemandEconomy
from copy import deepcopy
from Parameters import returnParameters
    
[init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
DiscFacCount, AgentCountTotal, base_dict, num_max_iterations_solvingAD,\
convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
data_EducShares, max_recession_duration, num_experiment_periods,\
recession_changes, UI_changes, recession_UI_changes,\
TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes] = returnParameters(Parametrization='Baseline',OutputFor='_Main.py')
      
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
IncShkDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([agent1.IncUnempNoBenefits])])
        
for ThisType in BaseTypeList:
    EmployedIncShkDstn = deepcopy(ThisType.IncShkDstn[0])
    ThisType.IncShkDstn = [[ThisType.IncShkDstn[0]] + [IncShkDstn_unemp]*UBspell_normal + [IncShkDstn_unemp_nobenefits]]
    ThisType.IncShkDstn_base = ThisType.IncShkDstn
        
    IncShkDstn_recession = [ThisType.IncShkDstn[0]*(2*(num_experiment_periods+1))] # for normal, rec, recovery  
    ThisType.IncShkDstn_recession = IncShkDstn_recession
    ThisType.IncShkDstn_recessionUI = IncShkDstn_recession
        
    EmployedIncShkDstn.atoms[0][1] = EmployedIncShkDstn.atoms[0][1]*ThisType.TaxCutIncFactor
    TaxCutStatesIncShkDstn = [EmployedIncShkDstn] + [IncShkDstn_unemp]*UBspell_normal + [IncShkDstn_unemp_nobenefits] 
    IncShkDstn_recessionTaxCut = deepcopy(IncShkDstn_recession)
    # Tax states are 2,3 (q1) 4,5 (q2) ... 16,17 (q8)
    for i in range(2*num_base_MrkvStates,18*num_base_MrkvStates,1):
        IncShkDstn_recessionTaxCut[0][i] =  TaxCutStatesIncShkDstn[np.mod(i,4)]
    ThisType.IncShkDstn_recessionTaxCut = IncShkDstn_recessionTaxCut
        
    ThisType.IncShkDstn_recessionCheck = deepcopy(IncShkDstn_recession)
    ThisType.mCount = 200
    ThisType.mFac = 3
    ThisType.mMin = 1e-4
    ThisType.mMax = 10000

agent3.DiscFac = .988    
agent3.solve()
agent3.define_distribution_grid(num_pointsP=110, timestonest=3)

mGrid = agent3.dist_mGrid

agent3.neutral_measure = True

agent3.update_income_process()

# Problem: cFunc for this model is different
# needs to be indexed and requires a second argument of cRatios
agent3.calc_transition_matrix()

agent3.compute_steady_state()

print(agent3.C_ss)
print(agent3.A_ss)
