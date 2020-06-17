'''
Runs Fiscal Experiments:
    1) Parker style one off payment, arriving in 2 quarters
    2) Bush style tax cut on wages for a duration of about 2 years
'''

from builtins import str
from builtins import range
import numpy as np
from time import time
from copy import deepcopy
from StickyEmodel import StickyEmarkovConsumerType, StickySmallOpenMarkovEconomy
import matplotlib.pyplot as plt
import StickyEparams as Params
from StickyEtools import runParkerExperiment, makeStickyEdataFile


ignore_periods = Params.ignore_periods # Number of simulated periods to ignore as a "burn-in" phase
interval_size = Params.interval_size   # Number of periods in each non-overlapping subsample
total_periods = Params.periods_to_sim  # Total number of periods in simulation
interval_count = (total_periods-ignore_periods) // interval_size # Number of intervals in the macro regressions
periods_to_sim_micro = Params.periods_to_sim_micro # To save memory, micro regressions are run on a smaller sample
AgentCount_micro = Params.AgentCount_micro # To save memory, micro regressions are run on a smaller sample
my_counts = [interval_size,interval_count]
long_counts = [interval_size*interval_count,1]
mystr = lambda number : "{:.3f}".format(number)
results_dir = Params.results_dir

# Run models and save output if this module is called from main
if __name__ == '__main__':

    ###############################################################################
    ########## SMALL OPEN ECONOMY WITH MACROECONOMIC MARKOV STATE##################
    ###############################################################################

    run_models = True
    run_parker = True
    verbose_main = True
    save_data = True
    if run_models:
        # Choose parameter values depending on whether or not the Parker experiment
        # is being run right now.  The main results use a single discount factor.
        if not run_parker:
            TypeCount = Params.TypeCount
            IncUnemp = Params.IncUnemp
            DiscFacSetSOE = Params.DiscFacSetSOE
        else:
            TypeCount = Params.TypeCount_parker
            IncUnemp = Params.IncUnemp_parker
            DiscFacSetSOE = Params.DiscFacSetSOE_parker
        
        # Make consumer types to inhabit the small open Markov economy
        init_dict = deepcopy(Params.init_SOE_mrkv_consumer)
        init_dict['IncUnemp'] = IncUnemp
        init_dict['AgentCount'] = Params.AgentCount // TypeCount
        StickySOEmarkovBaseType = StickyEmarkovConsumerType(**init_dict)
        StickySOEmarkovBaseType.IncomeDstn[0] = Params.StateCount*[StickySOEmarkovBaseType.IncomeDstn[0]]
        StickySOEmarkovBaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
        StickySOEmarkovConsumers = []
        for n in range(TypeCount):
            StickySOEmarkovConsumers.append(deepcopy(StickySOEmarkovBaseType))
            StickySOEmarkovConsumers[-1].seed = n
            StickySOEmarkovConsumers[-1].DiscFac = DiscFacSetSOE[n]

        # Make a small open economy for the agents
        StickySOmarkovEconomy = StickySmallOpenMarkovEconomy(agents=StickySOEmarkovConsumers, **Params.init_SOE_mrkv_market)
        StickySOmarkovEconomy.track_vars += ['TranShkAggNow','wRteNow']
        StickySOmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
        for n in range(TypeCount):
            StickySOEmarkovConsumers[n].getEconomyData(StickySOmarkovEconomy) # Have the consumers inherit relevant objects from the economy

        # Solve the small open Markov model
        t_start = time()
        print('Now solving the SOE model; this will take a few minutes.')
        StickySOmarkovEconomy.solveAgents()
        t_end = time()
        print('Solving the small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

        # Plot the consumption function in each Markov state
        my_new_title = 'Consumption function for one type in the small open Markov economy:'
        m = np.linspace(0.,20.,500)
        M = np.ones_like(m)
        c = np.zeros((Params.StateCount,m.size))
        for i in range(Params.StateCount):
            c[i,:] = StickySOEmarkovConsumers[0].solution[0].cFunc[i](m,M)
            plt.plot(m,c[i,:])
        plt.title(my_new_title)
        plt.xlim([0.,20.])
        plt.ylim([0.,None])
        if verbose_main:
            print(my_new_title)
            plt.show()
        plt.close()

        # Simulate the sticky small open Markov economy
        t_start = time()
        for agent in StickySOmarkovEconomy.agents:
            agent(UpdatePrb = Params.UpdatePrb)
        StickySOmarkovEconomy.makeHistory()
        t_end = time()
        print('Simulating the sticky small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

        # Make results for the sticky small open Markov economy
        desc = 'Results for the sticky small open Markov economy with update probability ' + mystr(Params.UpdatePrb)
        name = 'SOEmarkovSticky'
        makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data)

        # Simulate the frictionless small open Markov economy
        t_start = time()
        for agent in StickySOmarkovEconomy.agents:
            agent(UpdatePrb = 1.0)
        StickySOmarkovEconomy.makeHistory()
        t_end = time()
        print('Simulating the frictionless small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

        # Make results for the frictionless small open Markov economy
        desc = 'Results for the frictionless small open Markov economy (update probability 1.0)'
        name = 'SOEmarkovFrictionless'
        makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data)

    # Run the "Parker experiment"
    if run_parker and run_models:
        t_start = time()
        
        # First, clear the simulation histories for all of the types to free up memory space;
        # this allows the economy to be copied without blowing up the computer.
        attr_list = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
        for agent in StickySOmarkovEconomy.agents:
            for attr in attr_list:
                delattr(agent,attr+'_hist')
            agent.track_vars = [] # Don't need to track any simulated variables
            
        # The market is at the end of its pre-generated simulated shock history, so it needs to be
        # reset back to an earlier shock index that has the same Markov state as the current one.
        MrkvNow = StickySOmarkovEconomy.MrkvNow
        Shk_idx_reset = np.where(StickySOmarkovEconomy.MrkvNow_hist == MrkvNow)[0][0]
        StickySOmarkovEconomy.Shk_idx = Shk_idx_reset
        
        # Run Parker experiments for different lead times for the policy
        runParkerExperiment(StickySOmarkovEconomy,0.05,1,4,True) # One quarter ahead
        runParkerExperiment(StickySOmarkovEconomy,0.05,2,4,True) # Two quarters ahead
        runParkerExperiment(StickySOmarkovEconomy,0.05,3,4,True) # Three quarters ahead
        
        t_end = time()
        print('Running the "Parker experiment" took ' + str(t_end-t_start) + ' seconds.')
