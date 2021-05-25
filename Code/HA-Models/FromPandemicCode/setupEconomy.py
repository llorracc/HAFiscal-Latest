from Parameters import T_sim, init_infhorizon, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
     figs_dir, num_max_iterations_solvingAD, convergence_tol_solvingAD, num_recovery_states,\
     UBspell_normal
from AggFiscalModel import AggFiscalType, AggregateDemandEconomy
from HARK.distribution import DiscreteDistribution
import numpy as np
from copy import deepcopy

base_dict_agg = deepcopy(base_dict)
    
# Make baseline types - for now only one type, might add more
num_types = 1 
# This is not the number of discount factors, but the number of household types; in pandemic paper, there were different education groups
InfHorizonTypeAgg = AggFiscalType(**init_infhorizon)
InfHorizonTypeAgg.cycles = 0
AggDemandEconomy = AggregateDemandEconomy(**init_ADEconomy)
InfHorizonTypeAgg.getEconomyData(AggDemandEconomy)
BaseTypeList = [InfHorizonTypeAgg]
  
# Fill in the Markov income distribution for each base type
# NOTE: THIS ASSUMES NO LIFECYCLE
IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg.IncUnemp])])
IncomeDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg.IncUnempNoBenefits])])
# IncomeDstn_big = []
# for ThisType in BaseTypeList:
#     IncomeDstn_taxcut = deepcopy(ThisType.IncomeDstn[0])
#     IncomeDstn_taxcut.X[1] = IncomeDstn_taxcut.X[1]*ThisType.TaxCutIncFactor
    

#     for i in range(4): # for normal, rec, UI normal, UI rec states
#         IncomeDstn_big.append(ThisType.IncomeDstn[0])
#         IncomeDstn_big.append(IncomeDstn_unemp_nobenefits)
#         IncomeDstn_big.append(IncomeDstn_unemp)

#     for i in range(32): # for 16 tax cut states for each business cycle state
#         IncomeDstn_big.append(IncomeDstn_taxcut)
#         IncomeDstn_big.append(IncomeDstn_unemp_nobenefits)
#         IncomeDstn_big.append(IncomeDstn_unemp)

#     for i in range(2): # check state in normal and rec
#         IncomeDstn_big.append(ThisType.IncomeDstn[0])
#         IncomeDstn_big.append(IncomeDstn_unemp_nobenefits)
#         IncomeDstn_big.append(IncomeDstn_unemp)
        
#     IncomeDstn_big = [IncomeDstn_big] #required to bring it in right form
                       
                           
#     ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp]
#     ThisType.IncomeDstn_big = IncomeDstn_big
#     ThisType.AgentCount = AgentCountTotal
#     ThisType.DiscFac = 0.96
#     ThisType.seed = 0

for ThisType in BaseTypeList:
    ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0]] + [IncomeDstn_unemp]*UBspell_normal + [IncomeDstn_unemp_nobenefits] 
    IncomeDstn_recession = [ThisType.IncomeDstn[0]*(2 + num_recovery_states)] # for normal, rec, recovery  
    ThisType.IncomeDstn_base = ThisType.IncomeDstn
    ThisType.IncomeDstn_recession = IncomeDstn_recession
    ThisType.AgentCount = AgentCountTotal
    ThisType.DiscFac = 0.96
    ThisType.seed = 0
    
    
# Make the overall list of types
TypeList = []
n = 0
for b in range(DiscFacDstns[0].X.size):
    for e in range(num_types):
        DiscFac = DiscFacDstns[e].X[b]
        AgentCount = int(np.floor(AgentCountTotal*TypeShares[e]*DiscFacDstns[e].pmf[b]))
        ThisType = deepcopy(BaseTypeList[e])
        ThisType.AgentCount = AgentCount
        ThisType.DiscFac = DiscFac
        ThisType.seed = n
        TypeList.append(ThisType)
        n += 1
AggDemandEconomy.agents = TypeList

AggDemandEconomy.solve()

AggDemandEconomy.reset()
for agent in AggDemandEconomy.agents:
    agent.initializeSim()
    agent.AggDemandFac = 1.0
    agent.RfreeNow = 1.0
    agent.CaggNow = 1.0

AggDemandEconomy.makeHistory()   
AggDemandEconomy.saveState()   
AggDemandEconomy.switchToCounterfactualMode("base")
AggDemandEconomy.makeIdiosyncraticShockHistories()

output_keys = ['NPV_AggIncome', 'NPV_AggCons', 'AggIncome', 'AggCons']

max_policy_duration = 6
PolicyUBspell = AggDemandEconomy.agents[0].PolicyUBspell #NOTE - this should come from the market, not the agent
PolicyUBpersist = 1.-1./PolicyUBspell
policy_prob_array = np.array([PolicyUBpersist**t*(1-PolicyUBpersist) for t in range(max_policy_duration)])
policy_prob_array[-1] = 1.0 - np.sum(policy_prob_array[:-1])

max_recession_duration = 21
Rspell = AggDemandEconomy.agents[0].Rspell #NOTE - this should come from the market, not the agent
R_persist = 1.-1./Rspell
recession_prob_array = np.array([R_persist**t*(1-R_persist) for t in range(max_recession_duration)])
recession_prob_array[-1] = 1.0 - np.sum(recession_prob_array[:-1])

recession_Cond9q_prob_array = deepcopy(recession_prob_array[0:13])
recession_Cond9q_prob_array[-1] = 1.0 - np.sum(recession_Cond9q_prob_array[:-1])