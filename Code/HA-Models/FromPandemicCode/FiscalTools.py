'''
This file has major functions that are used by GiveItAwayMAIN.py
'''
import warnings
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from HARK import multiThreadCommands, multiThreadCommandsFake

mystr = lambda x : '{:.2f}'.format(x)
mystr2 = lambda x : '{:.3f}'.format(x)

def runExperiment(Agents,RecessionShock = False,TaxCutShock = False, \
                  ExtendedUIShock =False, UpdatePrb = 1.0, Splurge = 0.0):
    '''
    Conduct an experiment in which the recession hits and/or fiscal policy is initiated.
    
    Parameters
    ----------
    Agents : [AgentType]
        List of agent types in the economy.
    RecessionShock : bool
        Indicator for whether the recession actually hits.
        
    Returns
    -------
    TBD
    '''
    T = Agents[0].T_sim
    
    # Make dictionaries of parameters to give to the agents
    experiment_dict = {
            'use_prestate' : True,
            'RecessionShock' : RecessionShock,
            'TaxCutShock' : TaxCutShock,
            'ExtendedUIShock' : ExtendedUIShock,
            'UpdatePrb' : UpdatePrb
            }
      
    # Begin the experiment by resetting each type's state to the baseline values
    PopCount = 0
    for ThisType in Agents:
        ThisType.read_shocks = True
        ThisType(**experiment_dict)
        PopCount += ThisType.AgentCount
        
    # Update the perceived and actual Markov arrays, solve and re-draw shocks if
    # warranted, then impose the recession shock, and finally
    # simulate the model for three years.
    experiment_commands = ['updateMrkvArray()', 'solveIfChanged()',
                           'makeShocksIfChanged()', 'initializeSim()',
                           'hitWithRecessionShock()',
                           'simulate()']
    multiThreadCommandsFake(Agents, experiment_commands)
    
    # Extract simulated consumption, labor income, and weight data
    cNrm_all = np.concatenate([ThisType.history['cNrmNow'] for ThisType in Agents], axis=1)
    Mrkv_hist = np.concatenate([ThisType.history['MrkvNow'] for ThisType in Agents], axis=1)
    pLvl_all = np.concatenate([ThisType.history['pLvlNow'] for ThisType in Agents], axis=1)
    TranShk_all = np.concatenate([ThisType.history['TranShkNow'] for ThisType in Agents], axis=1)
    mNrm_all = np.concatenate([ThisType.history['mNrmNow'] for ThisType in Agents], axis=1)
    aNrm_all = np.concatenate([ThisType.history['aNrmNow'] for ThisType in Agents], axis=1)
    cLvl_all = cNrm_all*pLvl_all
    # Calculate Splurge results (agents splurge on some of their income, and follow model for the rest)
    cLvl_all_splurge = (1.0-Splurge)*cLvl_all + Splurge*pLvl_all*TranShk_all
    # Get initial Markov states
    Mrkv_init = np.concatenate([ThisType.history['MrkvNow'][0,:] for ThisType in Agents])
    return_dict = {'cNrm_all' : cNrm_all,
                   'TranShk_all' : TranShk_all,
                   'cLvl_all' : cLvl_all,
                   'pLvl_all' : pLvl_all,
                   'Mrkv_hist' : Mrkv_hist,
                   'Mrkv_init' : Mrkv_init,
                   'mNrm_all' : mNrm_all,
                   'aNrm_all' : aNrm_all,
                   'cLvl_all_splurge' : cLvl_all_splurge}
    return return_dict

