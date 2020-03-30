# Import python tools

import sys
import os

import numpy as np
import random
from copy import deepcopy

# Import needed tools from HARK

from HARK.utilities import approxUniform, getPercentiles
from HARK.parallel import multiThreadCommands
from HARK.estimation import minimizeNelderMead
from HARK.ConsumptionSaving.ConsIndShockModel import *
from HARK.cstwMPC.SetupParamsCSTW import init_infinite

# Set key problem-specific parameters

TypeCount =  1   # Number of consumer types with heterogeneous discount factors
AdjFactor = 1.0  # Factor by which to scale all of MPCs in Table 9
T_kill = 400     # Don't let agents live past this age
#T_kill = 100     # Don't let agents live past this age
Splurge = 0.0    # Consumers automatically spend this amount of any lottery prize
do_secant = True # If True, calculate MPC by secant, else point MPC
drop_corner = False # If True, ignore upper left corner when calculating distance


# Set standard HARK parameter values

base_params = deepcopy(init_infinite)
### NOTE: These parameters below are ANNUAL calibrations, we will want a quarterly model - stick with the parameterization from init_infinite for now
base_params['LivPrb'] = [0.975**0.25]
#base_params['LivPrb'] = [0.975]
base_params['Rfree'] = (1.04**0.25)/base_params['LivPrb'][0]
#base_params['Rfree'] = 1.04/base_params['LivPrb'][0]
base_params['PermShkStd'] = [0.1]
base_params['TranShkStd'] = [0.1]
base_params['T_age'] = T_kill # Kill off agents if they manage to achieve T_kill working years
base_params['AgentCount'] = 10000
base_params['pLvlInitMean'] = np.log(23.72) # From Table 1, in thousands of USD

#### T_sim needs to be long enough to reach the ergodic distribution, maybe longer that 1000
base_params['T_sim'] = 100 # No point simulating past when agents would be killed off

# Define the MPC targets from Fagereng et al Table 9; element i,j is lottery quartile i, deposit quartile j

MPC_target_base = np.array([[1.047, 0.745, 0.720, 0.490],
                            [0.762, 0.640, 0.559, 0.437],
                            [0.663, 0.546, 0.390, 0.386],
                            [0.354, 0.325, 0.242, 0.216]])
MPC_target = AdjFactor*MPC_target_base


# Define the four lottery sizes, in thousands of USD; these are eyeballed centers/averages

lottery_size = np.array([1.625, 3.3741, 7.129, 40.0])

#%%

# Make several consumer types to be used during estimation

BaseType = IndShockConsumerType(**base_params)
EstTypeList = []
for j in range(TypeCount):
    EstTypeList.append(deepcopy(BaseType))
    EstTypeList[-1](seed = j)
    
    
# Define the objective function

def FagerengObjFunc(center,spread,verbose=False):
    '''
    Objective function for the quick and dirty structural estimation to fit
    Fagereng, Holm, and Natvik's Table 9 results with a basic infinite horizon
    consumption-saving model (with permanent and transitory income shocks).

    Parameters
    ----------
    center : float
        Center of the uniform distribution of discount factors.
    spread : float
        Width of the uniform distribution of discount factors.
    verbose : bool
        When True, print to screen MPC table for these parameters.  When False,
        print (center, spread, distance).

    Returns
    -------
    distance : float
        Euclidean distance between simulated MPCs and (adjusted) Table 9 MPCs.
    '''
    # Give our consumer types the requested discount factor distribution
    beta_set = approxUniform(N=TypeCount,bot=center-spread,top=center+spread)[1]
    for j in range(TypeCount):
        EstTypeList[j](DiscFac = beta_set[j])

    # Solve and simulate all consumer types, then gather their wealth levels
    multiThreadCommands(EstTypeList,['solve()','initializeSim()','simulate()','unpackcFunc()'])
    WealthNow = np.concatenate([ThisType.aLvlNow for ThisType in EstTypeList])

    # Get wealth quartile cutoffs and distribute them to each consumer type
    quartile_cuts = getPercentiles(WealthNow,percentiles=[0.25,0.50,0.75])
    for ThisType in EstTypeList:
        WealthQ = np.zeros(ThisType.AgentCount,dtype=int)
        for n in range(3):
            WealthQ[ThisType.aLvlNow > quartile_cuts[n]] += 1
        ThisType(WealthQ = WealthQ)

    # Keep track of MPC sets in lists of lists of arrays
    MPC_set_list = [ [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]] ]

    # Calculate the MPC for each of the four lottery sizes for all agents
    ## NOTE - this is calculated as though the model is ANNUAL. Need to redo
    ## for the quarterly model. Best is probably to assume an equal probability
    ## of the lottery arriving in each of the four quarters, and make sure
    ## what we calculate matches what Fagereng et al estimate
    
    for ThisType in EstTypeList:
        
        c_base = np.zeros((ThisType.AgentCount,4)) #c_base (in case of no lottery win) for each quarter
        c_actu = np.zeros((ThisType.AgentCount,4)) #c_actu (actual consumption in case of lottery win in one random quarter) for each quarter
        a_actu = np.zeros((ThisType.AgentCount,4)) #a_actu captures the actual market resources after potential lottery win was added and c_actu deducted
        
        LotteryWin = np.zeros((ThisType.AgentCount,4)) 
        # Array with AgentCount x 4 periods many entries; there is only one 1 in each row indicating the quarter of the Lottery win for the agent in each row
        for i in range(ThisType.AgentCount):
            LotteryWin[i,random.randint(0,3)] = 1
        
        for period in range(4): #Simulate for 4 quarters as opposed to 1 year
            
            # Simulate forward for one quarter
            ThisType.simulate(1)           
            
#            # Determine whether Lottery win occurs in this quarter
#            if LotteryWinHasNotOccuredYet:
#                if period < 3: #we are not in the last quarter
#                    LotteryWinThisQuarter = random.randint(1,4)<=1 #25% chance of receiving lottery this quarter
#                    LotteryWinHasNotOccuredYet = LotteryWinThisQuarter==0
#                else: #we are in the last quarter
#                    LotteryWinThisQuarter = True; #If lottery hasn't been won yet, Win is certain in the last quarter
#                    LotteryWinHasNotOccuredYet = False;
#            else:
#                LotteryWinThisQuarter = False;
#                
#            print('LotteryWinThisQuarter',LotteryWinThisQuarter)
#            print('LotteryWinHasNotOccuredYet',LotteryWinHasNotOccuredYet)
                
                
            c_base[:,period] = ThisType.cNrmNow #Consumption in absence of lottery win
            MPC_this_type = np.zeros((ThisType.AgentCount,4)) #Empty array 
            
            
            for k in range(4): # Get MPC for all agents of this type for different lottery sizes
                
                Llvl = lottery_size[k]*LotteryWin[:,period]  #Lottery win occurs only if LotteryWinThisQuarter = True
                Lnrm = np.divide(Llvl,ThisType.pLvlNow)
                SplurgeNrm = np.multiply(Splurge/ThisType.pLvlNow,LotteryWin[:,period] ) #Splurge occurs only if LotteryWinThisQuarter = True
                
                if period == 0:
                    m_adj = ThisType.mNrmNow + Lnrm - SplurgeNrm
                    c_actu[:,period] = ThisType.cFunc[0](m_adj) + SplurgeNrm
                    a_actu[:,period] = ThisType.mNrmNow + Lnrm - c_actu[:,period]
                else:  
                    m_adj = a_actu[:,period-1]*base_params['Rfree']/ThisType.PermShkNow + ThisType.TranShkNow + Lnrm - SplurgeNrm
                    c_actu[:,period] = ThisType.cFunc[0](m_adj) + SplurgeNrm
                    a_actu[:,period] = a_actu[:,period-1]*base_params['Rfree']/ThisType.PermShkNow + ThisType.TranShkNow + Lnrm - c_actu[:,period]               
                            
                if period == 3: #last period
                    MPC_this_type[:,k] = (np.sum(c_actu,axis=1) - np.sum(c_base,axis=1))/(lottery_size[k]/ThisType.pLvlNow)
                
        # Sort the MPCs into the proper MPC sets
        for q in range(4):
            these = ThisType.WealthQ == q
            for k in range(4):
                MPC_set_list[k][q].append(MPC_this_type[these,k])

    # Calculate average within each MPC set
    simulated_MPC_means = np.zeros((4,4))
    for k in range(4):
        for q in range(4):
            MPC_array = np.concatenate(MPC_set_list[k][q])
            simulated_MPC_means[k,q] = np.mean(MPC_array)
            

    # Calculate Euclidean distance between simulated MPC averages and Table 9 targets
    diff = simulated_MPC_means - MPC_target
    if drop_corner:
        diff[0,0] = 0.0
    distance = np.sqrt(np.sum((diff)**2))
    if verbose:
        print(simulated_MPC_means)
    else:
        print (center, spread, distance)
    return [distance,simulated_MPC_means,Lnrm,c_actu, c_base,LotteryWin]


#%% Test function
guess = [0.96,0.01]
[distance,simulated_MPC_means,Lnrm,c_actu, c_base,LotteryWin]=FagerengObjFunc(guess[0],guess[1])
print(simulated_MPC_means)
print(c_actu[0:5,:])
print(c_base[0:5,:])
print(LotteryWin[0:5,:])
#%% Create matrix


##%% Conduct the estimation
#
#guess = [0.92,0.03]
#f_temp = lambda x : FagerengObjFunc(x[0],x[1])
#opt_params = minimizeNelderMead(f_temp, guess, verbose=True)
#print('Finished estimating for scaling factor of ' + str(AdjFactor) + ' and "splurge amount" of $' + str(1000*Splurge))
#print('Optimal (beta,nabla) is ' + str(opt_params) + ', simulated MPCs are:')
#dist = FagerengObjFunc(opt_params[0],opt_params[1],True)
#print('Distance from Fagereng et al Table 9 is ' + str(dist))


    






