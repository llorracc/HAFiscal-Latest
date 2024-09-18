import os

os.chdir('./Code/HA-Models/FromPandemicCode')

import numpy as np
from HARK.distribution import DiscreteDistribution
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from copy import deepcopy
from Parameters import returnParameters
import scipy.sparse as sp
import matplotlib.pyplot as plt
    
[init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
DiscFacCount, AgentCountTotal, base_dict, num_max_iterations_solvingAD,\
convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
data_EducShares, max_recession_duration, num_experiment_periods,\
recession_changes, UI_changes, recession_UI_changes,\
TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes] = \
    returnParameters(Parametrization='Baseline',OutputFor='_Main.py')
      
init_dropout["mCount"] = 200
init_dropout["mFac"] = 3
init_dropout["mMin"] = 1e-4
init_dropout["mMax"] = 10000
init_highschool["mCount"] = 200
init_highschool["mFac"] = 3
init_highschool["mMin"] = 1e-4
init_highschool["mMax"] = 10000
init_college["mCount"] = 200
init_college["mFac"] = 3
init_college["mMin"] = 1e-4
init_college["mMax"] = 10000

agent_DO = MarkovConsumerType(**init_dropout)
agent_DO.cycles = 0
agent_HS = MarkovConsumerType(**init_highschool)
agent_HS.cycles = 0
agent_CG = MarkovConsumerType(**init_college)
agent_CG.cycles = 0
AggDemandEconomy = MarkovConsumerType(**init_ADEconomy)
# agent_DO.get_economy_data(AggDemandEconomy)
# agent_HS.get_economy_data(AggDemandEconomy)
# agent_CG.get_economy_data(AggDemandEconomy)
BaseTypeList = [agent_DO, agent_HS, agent_CG]
          

bigT = 100
dx = 0.0001

##################################################################################################
# Income distributions
IncShkDstn = []
IncShkDstn_dx = []

# HAF distributions
for ThisType in BaseTypeList:
    IncShkDstn_emp = deepcopy(ThisType.IncShkDstn[0])
    IncShkDstn_emp_dx = deepcopy(IncShkDstn_emp)
    IncShkDstn_emp_dx.atoms[1] = IncShkDstn_emp_dx.atoms[1] * (1 + dx)    

    # quasi HAF unemp
    quasiHAFue = deepcopy(IncShkDstn_emp)
    # quasiHAFue.atoms[0] = quasiHAFue.atoms[0] * 0 + 1.0
    quasiHAFue.atoms[1] = quasiHAFue.atoms[1] * 0 + 0.7
    quasiHAFue_dx = deepcopy(quasiHAFue)
    quasiHAFue_dx.atoms[1] = quasiHAFue_dx.atoms[1] * (1 + dx)    

    # quasi HAF unemp
    quasiHAFue2 = deepcopy(IncShkDstn_emp)
    # quasiHAFue2.atoms[0] = quasiHAFue2.atoms[0] * 0 + 1.0
    quasiHAFue2.atoms[1] = quasiHAFue2.atoms[1] * 0 + 0.7
    quasiHAFue2_dx = deepcopy(quasiHAFue2)
    quasiHAFue2_dx.atoms[1] = quasiHAFue2_dx.atoms[1] * (1 + dx) 

    quasiHAFuenb = deepcopy(IncShkDstn_emp)
    # quasiHAFuenb.atoms[0] = quasiHAFue.atoms[0] * 0 + 1.0
    quasiHAFuenb.atoms[1] = quasiHAFue.atoms[1] * 0 + 0.5
    quasiHAFuenb_dx = deepcopy(quasiHAFuenb)
    quasiHAFuenb_dx.atoms[1] = quasiHAFuenb_dx.atoms[1] * (1 + dx) 

    IncShkDstn.append([deepcopy(IncShkDstn_emp), deepcopy(quasiHAFue), deepcopy(quasiHAFue2), deepcopy(quasiHAFuenb)])
    IncShkDstn_dx.append([deepcopy(IncShkDstn_emp_dx), deepcopy(quasiHAFue_dx), deepcopy(quasiHAFue2_dx), deepcopy(quasiHAFuenb_dx)])


##################################################################################################

def compute_type_jacobian(agent, dict, DiscFac, IncDist, IncDist_dx, param):
    dx = 0.0001

    agent_SS = deepcopy(agent)
    agent_SS.IncShkDstn = deepcopy(IncDist)
    agent_SS.DiscFac = DiscFac
    agent_SS.compute_steady_state()

    c = agent_SS.cPol_Grid
    a = agent_SS.aPol_Grid

    ##################################################################################################
    # Finite Horizon

    params = deepcopy(dict)
    params["T_cycle"] = bigT
    params["LivPrb"] = params["T_cycle"] * [agent_SS.LivPrb[0]]
    params["PermGroFac"] = params["T_cycle"] * [agent_SS.PermGroFac[0]]
    params["PermShkStd"] = params["T_cycle"] * [agent_SS.PermShkStd[0]]
    params["TranShkStd"] = params["T_cycle"] * [agent_SS.TranShkStd[0]]
    params["Rfree"] = params["T_cycle"] * [agent_SS.Rfree]
    params["MrkvArray"] = params["T_cycle"] * agent_SS.MrkvArray
    params["DiscFac"] = DiscFac
    params["cycles"] = 1

    FinHorizonAgent = MarkovConsumerType(**params)
    FinHorizonAgent.dist_pGrid = params["T_cycle"] * [np.array([1])]
    FinHorizonAgent.IncShkDstn = params["T_cycle"] * deepcopy(IncDist)
    FinHorizonAgent.solution_terminal = deepcopy(agent_SS.solution[0])

    if param == "IncShkDstn":
        agent_inc_dx = deepcopy(agent)
        agent_inc_dx.DiscFac = DiscFac
        agent_inc_dx.IncShkDstn = deepcopy(IncDist_dx)
        agent_inc_dx.neutral_measure = True
        agent_inc_dx.harmenberg_income_process()
        FinHorizonAgent.del_from_time_inv(
            "IncShkDstn",
        )
        FinHorizonAgent.IncShkDstn = (params["T_cycle"] - 1) * deepcopy(IncDist) + deepcopy(IncDist_dx)
        FinHorizonAgent.add_to_time_vary("IncShkDstn", "PermShkDstn", "TranShkDstn")
    elif param == "MrkvArray":
        Mrkv_dx = deepcopy(agent.MrkvArray[0])
        Mrkv_dx[0][0] = Mrkv_dx[0][0] + dx
        Mrkv_dx[1][0] = Mrkv_dx[1][0] + dx
        Mrkv_dx[2][0] = Mrkv_dx[2][0] + dx
        Mrkv_dx[3][0] = Mrkv_dx[3][0] + dx

        Mrkv_dx[0][1] = Mrkv_dx[0][1] - dx
        Mrkv_dx[1][2] = Mrkv_dx[1][2] - dx
        Mrkv_dx[2][3] = Mrkv_dx[2][3] - dx
        Mrkv_dx[3][3] = Mrkv_dx[3][3] - dx
        FinHorizonAgent.MrkvArray = (params["T_cycle"] - 1) * agent.MrkvArray + [Mrkv_dx]
    elif param == "Rfree":
        FinHorizonAgent.del_from_time_inv(
            "Rfree",
        )  # delete Rfree from time invariant list since it varies overtime
        FinHorizonAgent.add_to_time_vary("Rfree")
        FinHorizonAgent.Rfree = (params["T_cycle"] - 1) * [agent.Rfree] + [agent.Rfree + dx] + [agent.Rfree]
    elif param == "DiscFac":
        FinHorizonAgent.del_from_time_inv(
            "DiscFac",
        )  # delete Rfree from time invariant list since it varies overtime
        FinHorizonAgent.add_to_time_vary("DiscFac")
        FinHorizonAgent.DiscFac = (params["T_cycle"] - 1) * [DiscFac] + [DiscFac + dx]

    FinHorizonAgent.solve()

    if param == "IncShkDstn":
        FinHorizonAgent.IncShkDstn = (params["T_cycle"] - 1) * deepcopy(agent_SS.IncShkDstn) + deepcopy(agent_inc_dx.IncShkDstn)
    else:
        FinHorizonAgent.IncShkDstn = params["T_cycle"] * deepcopy(agent_SS.IncShkDstn)

    # Calculate Transition Matrices
    FinHorizonAgent.neutral_measure = True
    # FinHorizonAgent.harmenberg_income_process()
    FinHorizonAgent.define_distribution_grid()
    FinHorizonAgent.calc_transition_matrix() 

    ##################################################################################################
    # period zero shock agent

    Zeroth_col_agent = MarkovConsumerType(**params)
    Zeroth_col_agent.solution_terminal = deepcopy(agent_SS.solution[0])
    Zeroth_col_agent.IncShkDstn = params["T_cycle"] * deepcopy(IncDist)
    Zeroth_col_agent.solve()

    if param == "IncShkDstn":
        Zeroth_col_agent.IncShkDstn = deepcopy(agent_inc_dx.IncShkDstn) + (params["T_cycle"]) * deepcopy(agent_SS.IncShkDstn)
    elif param == "MrkvArray":
        Zeroth_col_agent.MrkvArray = [Mrkv_dx] + (params["T_cycle"] - 1) * agent.MrkvArray
    elif param == "Rfree":
        Zeroth_col_agent.Rfree = [agent.Rfree + dx] + (params["T_cycle"] - 1) * [agent.Rfree]
    elif param == "DiscFac":
        Zeroth_col_agent.DiscFac = [DiscFac + dx] + (params["T_cycle"] - 1) * [DiscFac]

    if param != "IncShkDstn":
        Zeroth_col_agent.IncShkDstn = params["T_cycle"] * deepcopy(agent_SS.IncShkDstn)

    Zeroth_col_agent.neutral_measure = True
    Zeroth_col_agent.define_distribution_grid()
    Zeroth_col_agent.calc_transition_matrix()

    #################################################################################################
    # calculate Jacobian

    D_ss = agent_SS.vec_erg_dstn

    c_ss = agent_SS.cPol_Grid.flatten()
    a_ss = agent_SS.aPol_Grid.flatten()

    c_t_unflat = FinHorizonAgent.cPol_Grid
    a_t_unflat = FinHorizonAgent.aPol_Grid

    A_ss = agent_SS.A_ss
    C_ss = agent_SS.C_ss
    
    transition_matrices = FinHorizonAgent.tran_matrix

    c_t_flat = np.zeros((params["T_cycle"], int(params["mCount"] * 4)))
    a_t_flat = np.zeros((params["T_cycle"], int(params["mCount"] * 4)))

    for t in range( params["T_cycle"] ):
        c_t_flat[t] = c_t_unflat[t].flatten()
        a_t_flat[t] = a_t_unflat[t].flatten()

    tranmat_ss = agent_SS.tran_matrix

    tranmat_t = np.insert(transition_matrices, params["T_cycle"], tranmat_ss, axis = 0)

    c_t = np.insert(c_t_flat, params["T_cycle"] , c_ss , axis = 0)
    a_t = np.insert(a_t_flat, params["T_cycle"] , a_ss , axis = 0)

    CJAC_perfect, AJAC_perfect = compile_JAC(a_ss, c_ss, a_t, c_t, tranmat_ss, tranmat_t, D_ss, C_ss, A_ss, Zeroth_col_agent, 100)

    return CJAC_perfect, AJAC_perfect, c, a



##################################################################################################

def compile_JAC(a_ss, c_ss, a_t, c_t, tranmat_ss, tranmat_t, D_ss, C_ss, A_ss, Zeroth_col_agent, bigT):

    T = bigT

    # Expectation vectors
    exp_vecs_a_e = []
    exp_vec_a_e = a_ss
    
    exp_vecs_c_e = []
    exp_vec_c_e = c_ss
    
    for i in range(T):
        
        exp_vecs_a_e.append(exp_vec_a_e)
        exp_vec_a_e = np.dot(tranmat_ss.T, exp_vec_a_e)
        
        exp_vecs_c_e.append(exp_vec_c_e)
        exp_vec_c_e = np.dot(tranmat_ss.T, exp_vec_c_e)
    
    
    exp_vecs_a_e = np.array(exp_vecs_a_e)
    exp_vecs_c_e = np.array(exp_vecs_c_e)

    
    da0_s = []
    dc0_s = []

    for i in range(T):
        da0_s.append(a_t[T - i] - a_ss)
        dc0_s.append(c_t[T - i] - c_ss)
    
        
    da0_s = np.array(da0_s)
    dc0_s = np.array(dc0_s)

    dA0_s = []
    dC0_s = []

    for i in range(T):
        dA0_s.append(np.dot(da0_s[i], D_ss))
        dC0_s.append(np.dot(dc0_s[i], D_ss))
    
    dA0_s = np.array(dA0_s)
    A_curl_s = dA0_s/dx
    
    dC0_s = np.array(dC0_s)
    C_curl_s = dC0_s/dx
    
    dlambda0_s = []
    
    for i in range(T):
        dlambda0_s.append(tranmat_t[T - i] - tranmat_ss)
    
    dlambda0_s = np.array(dlambda0_s)
    
    dD0_s = []
    
    for i in range(T):
        dD0_s.append(np.dot(dlambda0_s[i], D_ss))
    
    dD0_s = np.array(dD0_s)
    D_curl_s = dD0_s/dx
    
    Curl_F_A = np.zeros((T , T))
    Curl_F_C = np.zeros((T , T))
    
    # WARNING: SWAPPED THESE LINES TO MAKE DEMO RUN
    # Curl_F_A[0] = A_curl_s
    # Curl_F_C[0] = C_curl_s
    Curl_F_A[0] = A_curl_s.T[0]
    Curl_F_C[0] = C_curl_s.T[0]

    for i in range(T-1):
        for j in range(T):
            Curl_F_A[i + 1][j] = np.dot(exp_vecs_a_e[i], D_curl_s[j])[0]
            Curl_F_C[i + 1][j] = np.dot(exp_vecs_c_e[i], D_curl_s[j])[0]

    J_A = np.zeros((T, T))
    J_C = np.zeros((T, T))

    for t in range(T):
        for s in range(T):
            if (t ==0) or (s==0):
                J_A[t][s] = Curl_F_A[t][s]
                J_C[t][s] = Curl_F_C[t][s]
                
            else:
                J_A[t][s] = J_A[t - 1][s - 1] + Curl_F_A[t][s]
                J_C[t][s] = J_C[t - 1][s - 1] + Curl_F_C[t][s]
     
    # Zeroth Column of the Jacobian
    Zeroth_col_agent.tran_matrix = np.array(Zeroth_col_agent.tran_matrix)
    
    C_t = np.zeros(T)
    A_t = np.zeros(T)
    
    dstn_dot = D_ss
    
    for t in range(T):
        tran_mat_t = Zeroth_col_agent.tran_matrix[t]

        dstn_all = np.dot(tran_mat_t, dstn_dot)

        C = np.dot(c_ss, dstn_all)
        A = np.dot(a_ss, dstn_all)
        
        C_t[t] = C[0]
        A_t[t] = A[0]

        dstn_dot = dstn_all
        
    J_A.T[0] = (A_t - A_ss) / dx
    J_C.T[0] = (C_t - C_ss) / dx

    return J_C, J_A

dicts = [init_dropout, init_highschool, init_college]
shock_params = ["Rfree", "DiscFac", "IncShkDstn", "MrkvArray"]

CJacs = []
AJacs = []
C_sss = []
A_sss = []

for i in range(3):
    betas = DiscFacDstns[i].atoms[0]
    dict = dicts[i]
    IncDist = [IncShkDstn[i]]
    IncDist_dx = [IncShkDstn_dx[i]]
    for beta in betas:
        for param in shock_params:
            CJac, AJac, C_ss, A_ss = compute_type_jacobian(BaseTypeList[i], dict, beta, IncDist, IncDist_dx, param)
            CJacs.append(CJac)
            AJacs.append(AJac)
            C_sss.append(C_ss)
            A_sss.append(A_ss)

for i in range(3 * 7 * 4):
    plt.plot(CJacs[i].T[0])
    plt.plot(CJacs[i].T[10])
    plt.plot(CJacs[i].T[30])
    plt.plot(CJacs[i].T[50])

plt.plot(np.zeros(100), 'k')
plt.title('Consumption Jacobian')
plt.show()

for i in range(3 * 7 * 4):
    plt.plot(AJacs[i].T[0])
    plt.plot(AJacs[i].T[10])
    plt.plot(AJacs[i].T[30])
    plt.plot(AJacs[i].T[50])

plt.plot(np.zeros(100), 'k')
plt.title('Asset Jacobian')
plt.show()
