'''
This file has an extension of MarkovConsumerType that is used for the Fiscal project.
'''
import warnings
import numpy as np
from HARK.distribution import DiscreteDistribution, Bernoulli, Uniform
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import MargValueFunc, ConsumerSolution
from HARK.ConsumptionSaving.ConsAggShockModel import MargValueFunc2D
from HARK.interpolation import LinearInterp, LowerEnvelope, BilinearInterp, VariableLowerBoundFunc2D, \
                                LinearInterpOnInterp1D, LowerEnvelope2D, UpperEnvelope, ConstantFunction
from HARK import Market
from HARK.core import distanceMetric, HARKobject
from Parameters import makeMrkvArray, T_sim
from copy import deepcopy
import matplotlib.pyplot as plt

# Define a modified MarkovConsumerType
class AggFiscalType(MarkovConsumerType):
    time_inv_ = MarkovConsumerType.time_inv_ 
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        MarkovConsumerType.__init__(self,cycles=1,time_flow=True,**kwds)
        self.shock_vars += ['update_draw']
        self.solveOnePeriod = solveAggConsMarkovALT
        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(MarkovConsumerType.time_vary_)
        self.time_inv = deepcopy(MarkovConsumerType.time_inv_)
        self.delFromTimeInv('Rfree', 'vFuncBool', 'CubicBool')
        
    def preSolve(self):
        self.MrkvArray = self.MrkvArray_pcvd
        MarkovConsumerType.preSolve(self)
        self.updateSolutionTerminal()
        
    def initializeSim(self):
        MarkovConsumerType.initializeSim(self)
        if hasattr(self,'use_prestate'):
            self.restoreState()
            self.MrkvArray = self.MrkvArray_sim
        else:   # set to ergodic unemployment rate during normal times
            init_unemp_dist = DiscreteDistribution(1.0-self.Urate_normal, np.array([0,1]), seed=self.RNG.randint(0,2**31-1))
            self.MrkvNow[:] = init_unemp_dist.drawDiscrete(self.AgentCount)
            if not hasattr(self,'mortality_off'):
                self.calcAgeDistribution()
                self.initializeAges()
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow[:] = self.Mrkv_univ

    def getEconomyData(self, Economy):
        '''
        Imports economy-determined objects into self from a Market.

        Parameters
        ----------
        Economy : Market
            The "macroeconomy" in which this instance "lives".  
        Returns
        -------
        None
        '''
        self.T_sim = Economy.act_T                   # Need to be able to track as many periods as economy runs
        self.Cgrid = self.CgridBase                  # Ratio of consumption to steady state consumption
        self.CFunc = Economy.CFunc                   # Next period's consumption ratio function
        self.ADFunc = Economy.ADFunc                 # Function that takes aggregate consumption to agg. demand function
        self.PermGroFacAgg = Economy.PermGroFacAgg   # Aggregate permanent productivity growth
        self.addToTimeInv('Cgrid', 'CFunc', 'PermGroFacAgg','ADFunc')

        
    def getMortality(self):
        '''
        A modified version of getMortality that reads mortality history if the
        attribute read_mortality exists.  This is a workaround to make sure the
        history of death events is identical across simulations.
        '''
        if (self.read_shocks or hasattr(self,'read_mortality')):
            who_dies = self.who_dies_backup[self.t_sim,:]
        else:
            who_dies = self.simDeath()
        self.simBirth(who_dies)
        self.who_dies = who_dies
        return None
    
    
    def simDeath(self):
        if hasattr(self,'mortality_off'):
            return np.zeros(self.AgentCount, dtype=bool)
        else:
            return MarkovConsumerType.simDeath(self)
        # Note - resources of agents who die just disappear
        
    def getRfree(self):
        '''
        Returns an array of size self.AgentCount with self.RfreeNow in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        '''
        RfreeNow = self.RfreeNow*np.ones(self.AgentCount)
        return RfreeNow
    
    def marketAction(self):
        '''
        In the aggregate shocks model, the "market action" is to simulate one
        period of receiving income and choosing how much to consume.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.simulate(1)
        
    def getCaggNow(self):  # This function exists to be overwritten in StickyE model
        return self.CaggNow*np.ones(self.AgentCount)
    
    def getAggDemandFacNow(self):  # This function exists to be overwritten in StickyE model
        return self.AggDemandFac*np.ones(self.AgentCount)

    def getShocks(self):
        MarkovConsumerType.getShocks(self)
        self.TranShkNow = self.TranShkNow*self.getAggDemandFacNow() # For simulation, just multiply transitive shock by the aggregate demand factor
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow = self.MrkvNow_temp # Make sure real sequence is recorded
        self.update_draw = self.RNG.permutation(np.array(range(self.AgentCount))) # A list permuted integers, low draws will update their aggregate Markov state
            
    def getStates(self):
        MarkovConsumerType.getStates(self)
        
        # Initialize the random draw of Pi*N agents who update
        how_many_update = int(round(self.UpdatePrb*self.AgentCount))
        self.update = self.update_draw < how_many_update
        # Only updaters change their perception of the Markov state
        if hasattr(self,'MrkvNowPcvd'):
            self.MrkvNowPcvd[self.update] = self.MrkvNow[self.update]
        else: # This only triggers in the first simulated period
            self.MrkvNowPcvd = np.ones(self.AgentCount,dtype=int)*self.MrkvNow
        #$$$$$$$$$$ 
        # update the idiosyncratic state (employed, unemployed with benefits, unemployed without benefits)
        # but leave the macro state as it is (idiosyncratic state is 'modulo 3')
        self.MrkvNowPcvd = np.remainder(self.MrkvNow,3) + 3*np.floor_divide(self.MrkvNowPcvd,3)
        

    def getMarkovStates(self):
        '''
        A modified method that forces all agents to be in a particular Markov
        state when the attribute Mrkv_univ is not None.  This allows us to draw
        income shocks for every Markov state for each agent in each simulated
        period when pre-specifying the shocks.  When the model is *actually*
        simulated, this ensures that agent i in period t and Markov state k will
        get the same income shocks *no matter which specification we use*.
        '''
        MarkovConsumerType.getMarkovStates(self) # Basic Markov state draw
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow_temp = self.MrkvNow
            self.MrkvNow = self.Mrkv_univ*np.ones(self.AgentCount, dtype=int)
            # ^^ Store the real states but force income shocks to be based on one particular state
            
    #$$$$$$$$$$    
    def updateMrkvArray(self):
        '''
        Constructs an updated MrkvArray_pcvd attribute to be used in solution (perceived),
        as well as MrkvArray_sim attribute to be used in simulation (actual).
        '''
        self.MrkvArray_pcvd = makeMrkvArray(self.Urate_normal, self.Uspell_normal, self.UBspell_normal, self.Urate_recession_real, self.Uspell_recession_real, self.Rspell_real, self.UBspell_extended_real, self.PolicyUBspell_real, self.PolicyTaxCutspell_real)
        self.MrkvArray_sim  = makeMrkvArray(self.Urate_normal, self.Uspell_normal, self.UBspell_normal, self.Urate_recession_pcvd, self.Uspell_recession_pcvd, self.Rspell_pcvd, self.UBspell_extended_pcvd, self.PolicyUBspell_pcvd, self.PolicyTaxCutspell_pcvd)
    
    def calcAgeDistribution(self):
        '''
        Calculates the long run distribution of t_cycle in the population.
        '''
        if self.T_cycle==1:
            T_cycle_actual = 400
            LivPrb_array = [[self.LivPrb[0][0]]]*T_cycle_actual
        else:
            T_cycle_actual = self.T_cycle
            LivPrb_array = self.LivPrb
        AgeMarkov = np.zeros((T_cycle_actual+1,T_cycle_actual+1))
        for t in range(T_cycle_actual):
            p = LivPrb_array[t][0]
            AgeMarkov[t,t+1] = p
            AgeMarkov[t,0] = 1. - p
        AgeMarkov[-1,0] = 1.
        
        AgeMarkovT = np.transpose(AgeMarkov)
        vals, vecs = np.linalg.eig(AgeMarkovT)
        dist = np.abs(np.abs(vals) - 1.)
        idx = np.argmin(dist)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore warning about casting complex eigenvector to float
            LRagePrbs = vecs[:,idx].astype(float)
        LRagePrbs /= np.sum(LRagePrbs)
        age_vec = np.arange(T_cycle_actual+1).astype(int)
        self.LRageDstn = DiscreteDistribution(LRagePrbs, age_vec,
                                seed=self.RNG.randint(0,2**31-1))
        
        
    def initializeAges(self):
        '''
        Assign initial values of t_cycle to simulated agents, using the attribute
        LRageDstn as the distribution of discrete ages.
        '''
        age = self.LRageDstn.drawDiscrete(self.AgentCount)
        age = age.astype(int)
        if self.T_cycle!=1:
            self.t_cycle = age
        self.t_age = age
    
    def switchToCounterfactualMode(self):
        '''
        Very small method that swaps in the "big" Markov-state versions of some
        solution attributes, replacing the "small" two-state versions that are used
        only to generate the pre-recession initial distbution of state variables.
        It then prepares this type to create alternate shock histories so it can
        run counterfactual experiments.
        '''
        del self.solution
        self.delFromTimeVary('solution')
        
        # Swap in "big" versions of the Markov-state-varying attributes
        self.LivPrb = self.LivPrb_big
        self.PermGroFac = self.PermGroFac_big
        self.MrkvArray = self.MrkvArray_big
        self.Rfree = self.Rfree_big
        self.IncomeDstn = self.IncomeDstn_big
        
        # Adjust simulation parameters for the counterfactual experiments
        self.T_sim = T_sim
        self.track_vars = ['cNrmNow','pLvlNow','aNrmNow','mNrmNow','MrkvNowPcvd']
        self.use_prestate = None
        self.MrkvArray_pcvd = self.MrkvArray
        #print('Finished type ' + str(self.seed) + '!')
        
        
    def makeAlternateShockHistories(self):
        '''
        Make a history of Markov states and income shocks starting from each Markov state.
        '''
        self.MrkvArray = self.MrkvArray_sim
        J = self.MrkvArray[0].shape[0]
        DeathHistAll = np.zeros((J,self.T_sim,self.AgentCount), dtype=bool)
        UpdateDrawHistAll = np.zeros((J,self.T_sim,self.AgentCount), dtype=int)
        MrkvHistAll = np.zeros((J,self.T_sim,self.AgentCount), dtype=int)
        TranShkHistCond = np.zeros((J,self.T_sim,self.AgentCount))
        PermShkHistCond = np.zeros((J,self.T_sim,self.AgentCount))
        for j in range(J):
            self.Mrkv_univ = j
            self.read_shocks = False
            self.makeShockHistory()
            DeathHistAll[j,:,:] = self.history['who_dies']
            UpdateDrawHistAll[j,:,:] = self.history['update_draw']
            MrkvHistAll[j,:,:] = self.history['MrkvNow']
            PermShkHistCond[j,:,:] = self.history['PermShkNow']
            TranShkHistCond[j,:,:] = self.history['TranShkNow']
            self.read_mortality = True # Make sure that every death history is the same
            self.who_dies_backup = self.history['who_dies'].copy()
        
        # Transfer income shocks conditional on each Markov state into the histories
        # that start in each Markov state
        TranShkHistAll = np.zeros((J,self.T_sim,self.AgentCount))
        PermShkHistAll = np.zeros((J,self.T_sim,self.AgentCount))
        for j in range(J):
            for k in range(J):
                these = MrkvHistAll[k,:,:] == j
                PermShkHistAll[k,][these] = PermShkHistCond[j,][these]
                TranShkHistAll[k,][these] = TranShkHistCond[j,][these]
        
        # Store as attributes of self
        self.DeathHistAll = DeathHistAll
        self.UpdateDrawHistAll = UpdateDrawHistAll
        self.MrkvHistAll = MrkvHistAll
        self.PermShkHistAll = PermShkHistAll
        self.TranShkHistAll = TranShkHistAll
        self.PermShkHistCond = PermShkHistCond
        self.TranShkHistCond = TranShkHistCond
        self.Mrkv_univ = None
        self.MrkvArray_sim_prev = self.MrkvArray_sim
        self.R_shared_prev = self.R_shared
        del(self.read_mortality)
        
        
    def solveIfChanged(self):
        '''
        Re-solve the lifecycle model only if the attributes MrkvArray_pcvd 
        do not match those in MrkvArray_pcvd_prev .
        '''
        # Check whether MrkvArray_pcvd has changed (and whether they exist at all!)
        try:
            same_MrkvArray = distanceMetric(self.MrkvArray_pcvd, self.MrkvArray_pcvd_prev) == 0.
            if (same_MrkvArray):
                return
        except:
            pass
        
        # Re-solve the model, then note the values in MrkvArray_pcvd
        self.solve()
        self.MrkvArray_pcvd_prev = self.MrkvArray_pcvd
        
        
    def makeShocksIfChanged(self):
        '''
        Re-draw the histories of Markov states and income shocks only if the attributes
        MrkvArray_sim and R_shared do not match those in MrkvArray_sim_prev and R_shared_prev.
        '''
        # Check whether MrkvArray_sim and R_shared have changed (and whether they exist at all!)
        try:
            same_MrkvArray = distanceMetric(self.MrkvArray_sim, self.MrkvArray_sim_prev) == 0.
            same_shared = self.R_shared == self.R_shared_prev
            if (same_MrkvArray and same_shared):
                return
        except:
            pass
        
        # Re-draw the shock histories, then note the values in MrkvArray_sim and R_shared
        self.makeAlternateShockHistories()
   
    
    def saveState(self):
        '''
        Record the current state of simulation variables for later use.
        '''
        self.aNrm_base = self.aNrmNow.copy()
        self.pLvl_base = self.pLvlNow.copy()
        self.Mrkv_base = self.MrkvNow.copy()
        self.MrkvPcvd_base = self.MrkvNowPcvd.copy()
        self.cycle_base  = self.t_cycle.copy()
        self.age_base  = self.t_age.copy()
        self.t_sim_base = self.t_sim
        self.PlvlAgg_base = self.PlvlAggNow


    def restoreState(self):
        '''
        Restore the state of the simulation to some baseline values.
        '''
        self.aNrmNow = self.aNrm_base.copy()
        self.pLvlNow = self.pLvl_base.copy()
        self.MrkvNow = self.Mrkv_base.copy()
        self.MrkvNowPcvd = self.MrkvPcvd_base.copy()
        self.t_cycle = self.cycle_base.copy()
        self.t_age   = self.age_base.copy()
        self.PlvlAggNow = self.PlvlAgg_base
        
    def hitWithRecessionShock(self):
        '''
        Alter the Markov state of each simulated agent, jumping some people into
        recession states
        '''
        # Shock unemployment up to ergodic unemployment level in normal or recession state
        if self.RecessionShock:
            this_Urate = self.Urate_recession_real
        else:
            this_Urate = self.Urate_normal
        
        # Draw new Markov states for each agents who are employed
        draws = Uniform(seed=self.RNG.randint(0,2**31-1)).draw(self.AgentCount)
        draws = self.RNG.permutation(draws)
        MrkvNew = self.MrkvNow
        old_Urate = self.Urate_normal
        draws_empy2umemp = draws > 1.0-(this_Urate-old_Urate)/(1.0-old_Urate)
        MrkvNew[np.logical_and(np.equal(self.MrkvNow,0), draws_empy2umemp) ] = 2 # Move people from employment to unemployment such that total unemployment rate is as required. Don't touch already unemployed people.
        #$$$$$$$$$$
        if (self.RecessionShock and not self.R_shared): # If the recssion actually occurs,
            MrkvNew += 3 # then put everyone into the recession 
            # This is (momentarily) skipped over if the recession state is shared
            # rather than idiosyncratic.  See a few lines below.
        if self.ExtendedUIShock:
            MrkvNew += 6 # put everyone in the extended UI states
        if self.TaxCutShock:
            MrkvNew +=12 # put everyone in the tax cut states
        if (self.ExtendedUIShock and self.TaxCutShock):
            print("Cannot handle UI and TaxCut experiments at the same time (yet)")
            return
        
        # Move agents to those Markov states 
        self.MrkvNow = MrkvNew
        
        # Take the appropriate shock history for each agent, depending on their state
        J = self.MrkvArray[0].shape[0]
        for j in range(J):
            these = self.MrkvNow == j
            self.history['who_dies'][:,these] = self.DeathHistAll[j,:,:][:,these]
            self.history['update_draw'][:,these] = self.UpdateDrawHistAll[j,:,:][:,these]
            self.history['MrkvNow'][:,these] = self.MrkvHistAll[j,:,:][:,these]
            self.history['PermShkNow'][:,these] = self.PermShkHistAll[j,:,:][:,these]
            self.history['TranShkNow'][:,these] = self.TranShkHistAll[j,:,:][:,these]
      
#        NEED TO FIX BELOW IF WE WANT SHARED RECESSION - NECESSARY TO CHANGE IF WE WANT SHOCKS TO BE CONTINGENT ON RECESSION STATE
#        POSSIBLE FIX - TAKE HISTORY FROM PermShkHistCond up to the point where the recession ends, then take history starting 
#        IN NON_RECESSION STATE THAT FOLLOWS AFTER (NEED TO CALC PROBABILITIES OF BEING IN EACH STATE AFTER RECESSION ENDS BASED ON STATE IN RECESSION)    
        if self.R_shared:
            print( "R_shared not implemented yet" )
#        # If the recession is a common/shared event, rather than idiosyncratic, bump
#        # everyone into the lockdown state for *exactly* T_lockdown periods
#        if (self.RecessionShock and self.R_shared):
#            T = self.T_recession
#            self.history['MrkvNow'][0:T,:] += 2
                   
    def getControls(self):
        '''
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        CaggNow = self.getCaggNow()
        J = self.MrkvArray[0].shape[0]
        
        MrkvBoolArray = np.zeros((J,self.AgentCount), dtype=bool)
        for j in range(J):
            MrkvBoolArray[j,:] = j == self.MrkvNowPcvd # Agents choose control based on *perceived* Markov state
        
        for t in range(self.T_cycle):
            right_t = t == self.t_cycle
            for j in range(J):
                these = np.logical_and(right_t, MrkvBoolArray[j,:])
                cNrmNow[these] = self.solution[t].cFunc[j](self.mNrmNow[these], CaggNow[these])
                # Marginal propensity to consume
                MPCnow[these]  = self.solution[t].cFunc[j].derivativeX(self.mNrmNow[these], CaggNow[these])
        self.cNrmNow = cNrmNow
        self.MPCnow  = MPCnow
                    
                
def solveAggConsMarkovALT(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                 MrkvArray,BoroCnstArt,aXtraGrid, Cgrid, CFunc, ADFunc):
    '''
    Solves a single period consumption-saving problem with risky income and
    stochastic transitions between discrete states, in a Markov fashion.  Has
    identical inputs as solveConsIndShock, except for a discrete
    Markov transitionrule MrkvArray.  Markov states can differ in their interest
    factor, permanent growth factor, and income distribution, so the inputs Rfree,
    PermGroFac, and IncomeDstn are arrays or lists specifying those values in each
    (succeeding) Markov state.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : DiscreteDistribution
        A representation of permanent and transitory income shocks that might
        arrive at the beginning of next period.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : np.array
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroFac : np.array
        Expected permanent income growth factor at the end of this period
        for each Markov state in the succeeding period.
    MrkvArray : np.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.

    Returns
    -------
    solution : ConsumerSolution
        The solution to the single period consumption-saving problem. Includes
        a consumption function cFunc (using cubic or linear splines), a marg-
        inal value function vPfunc, a minimum acceptable level of normalized
        market resources mNrmMin.  All of these attributes are lists or arrays, 
        with elements corresponding to the current Markov state.  E.g.
        solution.cFunc[0] is the consumption function when in the i=0 Markov
        state this period.
    '''
    # Get sizes of grids
    aCount = aXtraGrid.size
    Ccount = Cgrid.size
    StateCount = MrkvArray.shape[0]

    # Loop through next period's states, assuming we reach each one at a time.
    # Construct EndOfPrdvP_cond functions for each state.
    EndOfPrdvPfunc_cond = []
    BoroCnstNat_cond = []
    for j in range(StateCount):
        # Unpack next period's solution
        vPfuncNext = solution_next.vPfunc[j]
        mNrmMinNext = solution_next.mNrmMin[j]

        # Unpack the income shocks
        ShkPrbsNext = IncomeDstn[j].pmf
        PermShkValsNext = IncomeDstn[j].X[0]
        TranShkValsNext = IncomeDstn[j].X[1]
        ShkCount = ShkPrbsNext.size
        aXtra_tiled = np.tile(np.reshape(aXtraGrid, (1, aCount, 1)), (Ccount, 1, ShkCount))

        # Make tiled versions of the income shocks
        # Dimension order: aNow, Shk
        ShkPrbsNext_tiled = np.tile(np.reshape(ShkPrbsNext, (1, 1, ShkCount)), (Ccount, aCount, 1))
        PermShkValsNext_tiled = np.tile(np.reshape(PermShkValsNext, (1, 1, ShkCount)), (Ccount, aCount, 1))
        TranShkValsNext_tiled_noAD = np.tile(np.reshape(TranShkValsNext, (1, 1, ShkCount)), (Ccount, aCount, 1))
        
        # Calculate aggregate consumption next period
        CaggGrid = CFunc[j](Cgrid)
        Cnext_array = np.tile(np.reshape(CaggGrid, (Ccount, 1, 1)), (1, aCount, ShkCount)) ##$$$$$$$$ NOTE THIS WILL DEPEND ON THE STATE YOU MOVE TO! NEED CFunc to vary by state from AND state to

        # Calculate AggDemandFac
        AggDemandFacnext_array = ADFunc(Cnext_array)  
        TranShkValsNext_tiled = AggDemandFacnext_array*TranShkValsNext_tiled_noAD
        
        # Find the natural borrowing constraint for each value of C in the Cgrid.
        aNrmMin_candidates = PermGroFac[j]*PermShkValsNext_tiled/Rfree[j]* \
            (mNrmMinNext(Cnext_array[:, 0, :]) - TranShkValsNext_tiled[:, 0, :])
        aNrmMin_vec = np.max(aNrmMin_candidates, axis=1)
        BoroCnstNat_vec = aNrmMin_vec
        aNrmMin_tiled = np.tile(np.reshape(aNrmMin_vec, (Ccount, 1, 1)), (1, aCount, ShkCount))
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled


        # Calculate market resources next period (and a constant array of capital-to-labor ratio)
        mNrmNext_array = Rfree[j]*aNrmNow_tiled/PermShkValsNext_tiled + TranShkValsNext_tiled

        # Find marginal value next period at every income shock realization and every aggregate market resource gridpoint
        vPnext_array = Rfree[j]*PermShkValsNext_tiled**(-CRRA)*vPfuncNext(mNrmNext_array, Cnext_array)

        # Calculate expectated marginal value at the end of the period at every asset gridpoint
        EndOfPrdvP = DiscFac*np.sum(vPnext_array*ShkPrbsNext_tiled, axis=2)
        
        # Make the conditional end-of-period marginal value function
        BoroCnstNat = LinearInterp(np.insert(CaggGrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0))
        EndOfPrdvPnvrs = np.concatenate((np.zeros((Ccount, 1)), EndOfPrdvP**(-1./CRRA)), axis=1)
        EndOfPrdvPnvrsFunc_base = BilinearInterp(np.transpose(EndOfPrdvPnvrs), np.insert(aXtraGrid, 0, 0.0), CaggGrid)
        EndOfPrdvPnvrsFunc = VariableLowerBoundFunc2D(EndOfPrdvPnvrsFunc_base, BoroCnstNat)
        EndOfPrdvPfunc_cond.append(MargValueFunc2D(EndOfPrdvPnvrsFunc, CRRA))
        BoroCnstNat_cond.append(BoroCnstNat)
        
    # Prepare some objects that are the same across all current states
    aXtra_tiled = np.tile(np.reshape(aXtraGrid, (1, aCount)), (Ccount, 1))
    cFuncCnst = BilinearInterp(np.array([[0.0, 0.0], [1.0, 1.0]]),
                               np.array([BoroCnstArt, BoroCnstArt+1.0]), np.array([0.0, 1.0]))

    # Now loop through *this* period's discrete states, calculating end-of-period
    # marginal value (weighting across state transitions), then construct consumption
    # and marginal value function for each state.
    cFuncNow = []
    vPfuncNow = []
    mNrmMinNow = []
    for i in range(StateCount):
        # Find natural borrowing constraint for this state by Cagg
        Cnext = CFunc[i](Cgrid)
        aNrmMin_candidates = np.zeros((StateCount, Ccount)) + np.nan
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:  # Irrelevant if transition is impossible
                aNrmMin_candidates[j, :] = BoroCnstNat_cond[j](Cnext)
        aNrmMin_vec = np.nanmax(aNrmMin_candidates, axis=0)
        BoroCnstNat_vec = aNrmMin_vec

        # Make tiled grids of aNrm and Cagg
        aNrmMin_tiled = np.tile(np.reshape(aNrmMin_vec, (Ccount, 1)), (1, aCount))
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled
        Cnext_tiled = np.tile(np.reshape(Cnext, (Ccount, 1)), (1, aCount))

        
        # # Find the minimum allowable market resources
        # if BoroCnstArt is not None:
        #     mNrmMin = np.maximum(BoroCnstArt, aNrmMin)
        # else:
        #     mNrmMin = aNrmMin
        # mNrmMinNow.append(mNrmMin)
        
        # Loop through feasible transitions and calculate end-of-period marginal value
        EndOfPrdvP = np.zeros((Ccount, aCount))
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:
                temp = EndOfPrdvPfunc_cond[j](aNrmNow_tiled, Cnext_tiled)
                EndOfPrdvP += MrkvArray[i, j]*temp
        EndOfPrdvP *= LivPrb[i] # Account for survival out of the current state
        
        # Calculate consumption and the endogenous mNrm gridpoints for this state
        cNrmNow = EndOfPrdvP**(-1./CRRA)
        mNrmNow = aNrmNow_tiled + cNrmNow

        # Loop through the values in Cgrid and make a piecewise linear consumption function for each
        cFuncBaseByC_list = []
        for n in range(Ccount):
            c_temp = np.insert(cNrmNow[n, :], 0, 0.0)  # Add point at bottom
            m_temp = np.insert(mNrmNow[n, :] - BoroCnstNat_vec[n], 0, 0.0)
            cFuncBaseByC_list.append(LinearInterp(m_temp, c_temp))
            # Add the C-specific consumption function to the list
            
        # Construct the unconstrained consumption function by combining the M-specific functions
        BoroCnstNat = LinearInterp(np.insert(Cgrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0))
        cFuncBase = LinearInterpOnInterp1D(cFuncBaseByC_list, Cgrid)
        cFuncUnc = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat)

        # Combine the constrained consumption function with unconstrained component
        cFuncNow.append(LowerEnvelope2D(cFuncUnc, cFuncCnst))

        # Make the minimum m function as the greater of the natural and artificial constraints
        mNrmMinNow.append(UpperEnvelope(BoroCnstNat, ConstantFunction(BoroCnstArt)))

        # Construct the marginal value function using the envelope condition
        vPfuncNow.append(MargValueFunc2D(cFuncNow[-1], CRRA))

    # Pack up and return the solution
    solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow)
    return solution_now


class AggregateDemandEconomy(Market):
    '''
    A class to represent an economy in which productivity responds to aggregate
    consumption
    '''
    def __init__(self,
                 agents=None,
                 **kwds):
        '''
        Make a new instance of AggregateDemandEconomy by filling in attributes
        specific to this kind of market.
        '''
        agents = agents if agents is not None else list()

        Market.__init__(self, agents=agents,
                        sow_vars=['CaggNow', 'AggDemandFac','MrkvNow'],
                        reap_vars=['cLvlNow', 'pLvlNow'],
                        track_vars=['CaggNow', 'AggDemandFac','MrkvNow'],
                        dyn_vars=['CFunc'],
                        **kwds)
        self.update()


    def millRule(self, cLvlNow, pLvlNow):
        self.CaggNow = np.mean(np.array(cLvlNow))/np.mean(pLvlNow)  
        self.AggDemandFac = self.ADFunc(self.CaggNow)
        mill_return = HARKobject()
        mill_return.CaggNow = self.CaggNow
        mill_return.AggDemandFac = self.AggDemandFac
        MrkvNow = self.MrkvNow_hist[self.Shk_idx]
        mill_return.MrkvNow = MrkvNow
        return mill_return

    def calcDynamics(self):
        return self.calcCFunc()

    def update(self):
        '''
        '''
        self.CaggNow_init = 1.0
        self.AggDemandFac_init = 1.0
        self.ADFunc = lambda C : C**self.ADelasticity
        StateCount = self.MrkvArray.shape[0]
        CFunc_all = []
        for i in range(StateCount):
            CFunc_all.append(CRule(self.intercept_prev[i], self.slope_prev[i]))
        self.CFunc = CFunc_all

    def reset(self):
        '''
        Reset the economy to prepare for a new simulation.  Sets the time index
        of aggregate shocks to zero and runs Market.reset().

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.Shk_idx = 0
        Market.reset(self)

    def calcCFunc(self):
        StateCount = self.MrkvArray.shape[0]
        CFunc_all = []
        for i in range(StateCount):
            CFunc_all.append(CRule(self.intercept_prev[i], self.slope_prev[i]))
        self.CFunc = CFunc_all
    
class CRule(HARKobject):
    '''
    A class to represent agent beliefs about aggregate consumption dynamics.
    '''
    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope
        self.distance_criteria = ['slope', 'intercept']

    def __call__(self, Cnow):
        Cnext = np.exp(self.intercept + self.slope*np.log(Cnow))
        return Cnext