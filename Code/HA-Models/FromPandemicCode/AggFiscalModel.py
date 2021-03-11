'''
This file has an extension of MarkovConsumerType that is used for the Fiscal project.
'''
import warnings
import numpy as np
from HARK.distribution import DiscreteDistribution, Bernoulli, Uniform
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import MargValueFunc, ConsumerSolution
from HARK.ConsumptionSaving.ConsAggShockModel import MargValueFunc2D, AggShockMarkovConsumerType, AggShockConsumerType
from HARK.interpolation import LinearInterp, LowerEnvelope, BilinearInterp, VariableLowerBoundFunc2D, \
                                LinearInterpOnInterp1D, LowerEnvelope2D, UpperEnvelope, ConstantFunction
from HARK import Market, multiThreadCommands, multiThreadCommandsFake
from HARK.core import distanceMetric, HARKobject
from FiscalModel import FiscalType
from Parameters import makeMacroMrkvArray, T_sim
from copy import copy, deepcopy
import matplotlib.pyplot as plt

# Define a modified MarkovConsumerType
class AggFiscalType(FiscalType):
    time_inv_ = MarkovConsumerType.time_inv_ 
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        MarkovConsumerType.__init__(self,cycles=1,time_flow=True,**kwds)
        self.shock_vars += ['update_draw','unemployment_draw']
        self.solveOnePeriod = solveAggConsMarkovALT
        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(MarkovConsumerType.time_vary_)
        self.time_inv = deepcopy(MarkovConsumerType.time_inv_)
        self.delFromTimeInv('vFuncBool', 'CubicBool')
        self.addToTimeVary('IncomeDstn','PermShkDstn','TranShkDstn')
        self.addToTimeInv('aXtraGrid')
        
    def updateSolutionTerminal(self):
        AggShockConsumerType.updateSolutionTerminal(self)
        # Make replicated terminal period solution
        StateCount = self.MrkvArray[-1].shape[0]
        self.solution_terminal.cFunc = StateCount*[self.solution_terminal.cFunc]
        self.solution_terminal.vPfunc = StateCount*[self.solution_terminal.vPfunc]
        self.solution_terminal.mNrmMin = StateCount*[self.solution_terminal.mNrmMin]
        
    def preSolve(self):
        self.MrkvArray = self.MrkvArray
        MarkovConsumerType.preSolve(self)
        self.updateSolutionTerminal()
        
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
        self.Cgrid = Economy.CgridBase               # Ratio of consumption to steady state consumption
        self.CFunc = Economy.CFunc                   # Next period's consumption ratio function
        self.ADFunc = Economy.ADFunc                 # Function that takes aggregate consumption to agg. demand function
        self.addToTimeInv('Cgrid', 'CFunc','ADFunc')
        # self.PermGroFacAgg = Economy.PermGroFacAgg   # Aggregate permanent productivity growth
        #self.addToTimeInv('Cgrid', 'CFunc', 'PermGroFacAgg','ADFunc')
        
    def makeAlternateShockHistories(self):
        return "Not applicable for Aggregate model"
    
    def makeIdiosyncraticShockHistories(self):     
        print('makeIdiosyncraticShockHistories called')
        self.Mrkv_univ = 0
        self.read_shocks = False
        self.makeShockHistory()
        self.who_dies_fixed_hist = self.history['who_dies'].copy()
        self.update_draw_fixed_hist = self.history['update_draw'].copy()
        self.perm_shock_fixed_hist = self.history['PermShkNow'].copy()
        self.tran_shock_fixed_hist = self.history['TranShkNow'].copy()
        self.unemployment_draw_fixed_hist = self.history['unemployment_draw'].copy()
        self.Mrkv_univ = None
        
    def hitWithRecessionShock(self):
        '''
        Alter the Markov state of each simulated agent, jumping some people into
        recession states
        '''
        # Shock unemployment up to ergodic unemployment level in normal or recession state
        if self.RecessionShock:
            this_Urate = self.Urate_recession
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
        if (self.RecessionShock): # If the recssion actually occurs,
            MrkvNew += 3 # then put everyone into the recession 
        if self.ExtendedUIShock:
            MrkvNew += 6 # put everyone in the extended UI states
        if self.TaxCutShock:
            MrkvNew +=12 # put everyone in the tax cut states
            #might not change if we keep the order, if +12 relates to 1q of the reform
        if (self.ExtendedUIShock and self.TaxCutShock):
            print("Cannot handle UI and TaxCut experiments at the same time (yet)")
            return
        # Move agents to those Markov states 
        self.MrkvNow = MrkvNew
       
        self.history['MrkvNow'] = np.ones_like(self.history['PermShkNow'])
        t_age_start = copy(self.t_age)
        self.MicroMrkvNow = self.MrkvNow % 3
        self.MacroMrkvNow = np.floor(self.MrkvNow/3).astype(int)
        MicroMrkvNow_start = copy(self.MicroMrkvNow)
        MacroMrkvNow_start = copy(self.MacroMrkvNow)
        for t in range(self.T_sim):
            self.t_age = 1 - self.who_dies_fixed_hist[t] # hack to get newborns have t_age=0
            self.MacroMrkvNow = self.EconomyMrkvNow_hist[t] 
            unemployment_draw = self.unemployment_draw_fixed_hist[t]
            self.getMicroMarkvStates_guts(unemployment_draw)
            MrkvNow = 3*self.MacroMrkvNow + self.MicroMrkvNow
            self.history['MrkvNow'][t] = MrkvNow.astype(int)
        self.t_age = t_age_start
        self.MicroMrkvNow = MicroMrkvNow_start
        self.MacroMrkvNow = MacroMrkvNow_start
        self.MrkvNow = 3*self.MacroMrkvNow + self.MicroMrkvNow
            
        tax_cut_multiplier = np.ones_like(self.history['MrkvNow'])
        tax_cut_multiplier[np.greater(self.history['MrkvNow'], 11)] *= self.TaxCutIncFactor #$$$$$$$$$$ assumes all markov states above 11 are tax cut states
        employed = np.equal(self.history['MrkvNow']%3, 0)
        self.history['PermShkNow'][employed] = self.perm_shock_fixed_hist[employed]
        self.history['TranShkNow'][employed] = self.tran_shock_fixed_hist[employed]*tax_cut_multiplier[employed]
        unemp_without_benefits = np.equal(self.history['MrkvNow']%3, 1)
        self.history['PermShkNow'][unemp_without_benefits] = 1.0
        self.history['TranShkNow'][unemp_without_benefits] = self.IncUnempNoBenefits
        unemp_with_benefits = np.equal(self.history['MrkvNow']%3, 2)
        self.history['PermShkNow'][unemp_with_benefits] = 1.0
        self.history['TranShkNow'][unemp_with_benefits] = self.IncUnemp
        self.history['who_dies'] = self.who_dies_fixed_hist
        self.history['update_draw'] = self.update_draw_fixed_hist
        self.history['unemployment_draw'] = self.unemployment_draw_fixed_hist
        
    def switchToCounterfactualMode(self):
        FiscalType.switchToCounterfactualMode(self)
        self.track_vars += ['unemployment_draw']
        
    def getRfree(self):
        RfreeNow = self.Rfree[self.MrkvNow]*np.ones(self.AgentCount)
        return RfreeNow
    
    def marketAction(self):
        self.simulate(1)
        
    def getCratioNow(self):  # This function exists to be overwritten in StickyE model
        return self.CratioNow*np.ones(self.AgentCount)
    
    def getAggDemandFacNow(self):  
        return self.AggDemandFac*np.ones(self.AgentCount)

    def getShocks(self):
        MarkovConsumerType.getShocks(self)
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow = self.MrkvNow_temp # Make sure real sequence is recorded
        self.update_draw = self.RNG.permutation(np.array(range(self.AgentCount))) # A list permuted integers, low draws will update their aggregate Markov state
                   
    def getStates(self):
        FiscalType.getStates(self)
        self.mNrmNow = self.bNrmNow + self.TranShkNow*self.AggDemandFac # Market resources after income accounting for Agg Demand factor (this is for simulation)
        
        
    def getMacroMarkovStates(self):
        self.MacroMrkvNow = self.EconomyMrkvNow*np.ones(self.AgentCount, dtype=int)
                   
    def getControls(self):
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        CratioNow = self.getCratioNow()
        J = self.MrkvArray[0].shape[0]
        
        MrkvBoolArray = np.zeros((J,self.AgentCount), dtype=bool)
        for j in range(J):
            MrkvBoolArray[j,:] = j == self.MrkvNowPcvd # agents choose control based on *perceived* Markov state
        
        for t in range(self.T_cycle):
            right_t = t == self.t_cycle
            for j in range(J):
                these = np.logical_and(right_t, MrkvBoolArray[j,:])
                cNrmNow[these] = self.solution[t].cFunc[j](self.mNrmNow[these], CratioNow[these])
                # Marginal propensity to consume
                MPCnow[these]  = self.solution[t].cFunc[j].derivativeX(self.mNrmNow[these], CratioNow[these])
        self.cNrmNow = cNrmNow
        self.MPCnow  = MPCnow
        self.cLvlNow = cNrmNow*self.pLvlNow
        #self.cLvl_splurgeNow = (1.0-self.Splurge)*self.cLvlNow + self.Splurge*self.pLvlNow*self.TranShkNow
        self.cLvl_splurgeNow = (1.0-self.Splurge)*self.cLvlNow + self.Splurge*self.pLvlNow*self.TranShkNow*self.AggDemandFac   #added last term relaive to Edmund's Version
        
    def reset(self):
        return # do nothing
                    
                
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
        Cnext_array = np.tile(np.reshape(Cgrid, (Ccount, 1, 1)), (1, aCount, ShkCount)) 

        # Calculate AggDemandFac
        AggState = np.floor(j/3)
        RecState = AggState % 2 == 1
        AggDemandFacnext_array = ADFunc(Cnext_array,RecState)
        TranShkValsNext_tiled = AggDemandFacnext_array*TranShkValsNext_tiled_noAD
        
        # Find the natural borrowing constraint for each value of C in the Cgrid.
        aNrmMin_candidates = PermGroFac[j]*PermShkValsNext_tiled[:, 0, :]/Rfree[j]* \
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
        BoroCnstNat = LinearInterp(Cgrid, BoroCnstNat_vec)
        EndOfPrdvPnvrs = np.concatenate((np.zeros((Ccount, 1)), EndOfPrdvP**(-1./CRRA)), axis=1)
        EndOfPrdvPnvrsFunc_base = BilinearInterp(np.transpose(EndOfPrdvPnvrs), np.insert(aXtraGrid, 0, 0.0), Cgrid)
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
        # Find natural borrowing constraint for this state by Cratio NOTE THIS CODE IS NOT 100% CHECKED AND SHOULD BE LOOKED OVER
        aNrmMin_candidates = np.zeros((StateCount, Ccount)) + np.nan
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:  # Irrelevant if transition is impossible
                Cnext = CFunc[i][j](Cgrid)
                aNrmMin_candidates[j, :] = BoroCnstNat_cond[j](Cnext)
        aNrmMin_vec = np.nanmax(aNrmMin_candidates, axis=0)
        BoroCnstNat_vec = aNrmMin_vec

        # Make tiled grids of aNrm and Cratio
        aNrmMin_tiled = np.tile(np.reshape(aNrmMin_vec, (Ccount, 1)), (1, aCount))
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled

        
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
                Cnext = CFunc[i][j](Cgrid)
                Cnext_tiled = np.tile(np.reshape(Cnext, (Ccount, 1)), (1, aCount))
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
            
        # Construct the unconstrained consumption function by combining the C-specific functions
        BoroCnstNat = LinearInterp(Cgrid, BoroCnstNat_vec)
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
                        sow_vars=['CratioNow', 'AggDemandFac', 'AggDemandFacPrev','EconomyMrkvNow'],
                        reap_vars=['cLvl_splurgeNow'],
                        track_vars=['CratioNow','CratioPrev', 'AggDemandFac', 'AggDemandFacPrev','EconomyMrkvNow'],
                        dyn_vars=['CFunc'],
                        **kwds)
        self.update()


    def millRule(self, cLvl_splurgeNow):
        if self.Shk_idx==0:
            EconomyMrkvNow = 0
        else:
            EconomyMrkvNow = self.EconomyMrkvNow_hist[self.Shk_idx-1]   
        EconomyMrkvNext = self.EconomyMrkvNow_hist[self.Shk_idx]
        if hasattr(self,'base_AggCons'):
            cLvl_all_splurge = np.concatenate([this_cLvl for this_cLvl in cLvl_splurgeNow])      
            AggCons   = np.sum(cLvl_all_splurge)
            self.CratioNow = AggCons/self.base_AggCons[self.Shk_idx] 
            CratioNext = self.CFunc[EconomyMrkvNow*3][EconomyMrkvNext*3](self.CratioNow)
        else:
            self.CratioNow = 1.0
            CratioNext = 1.0
        self.AggDemandFacPrev = self.AggDemandFac
        self.CratioPrev = self.CratioNow
        RecState = EconomyMrkvNext % 2 == 1
        AggDemandFacNext = self.ADFunc(CratioNext,RecState)
        mill_return = HARKobject()
        mill_return.CratioNow = CratioNext
        mill_return.AggDemandFac = AggDemandFacNext
        mill_return.AggDemandFacPrev = self.AggDemandFacPrev
        mill_return.EconomyMrkvNow = EconomyMrkvNext
        self.Shk_idx += 1
        return mill_return

    def calcDynamics(self):
        return self.calcCFunc()

    def update(self):
        '''
        '''
        self.CratioNow_init = 1.0
        self.AggDemandFac_init = 1.0
        self.AggDemandFacPrev_init = 1.0
        self.ADFunc = lambda C, RecState : C**(RecState*self.ADelasticity)
        self.EconomyMrkvNow_hist = [0] * self.act_T
        StateCount = self.MrkvArray[0].shape[0]
        CFunc_all = []
        for i in range(StateCount):
            CFunc_i = []
            for j in range(StateCount):
                CFunc_i.append(CRule(self.intercept_prev[i,j], self.slope_prev[i,j]))
            CFunc_all.append(copy(CFunc_i))
        self.CFunc = CFunc_all

    def reset(self):
        self.Shk_idx = 0
        Market.reset(self)
        #self.EconomyMrkvNow_hist = [0] * self.act_T
        for agent in self.agents:
            agent.initializeSim()
        
    def runExperiment(self, RecessionShock = False,TaxCutShock = False, \
                      ExtendedUIShock =False, UpdatePrb = 1.0, Splurge = 0.0, EconomyMrkv_init = [0], Full_Output = True):
        # Make the macro markov history
        self.EconomyMrkvNow_hist = [0] * self.act_T
        self.EconomyMrkvNow_hist[0:len(EconomyMrkv_init)] = EconomyMrkv_init
    
        
        self.CratioNow_init = self.CFunc[0][EconomyMrkv_init[0]*3].intercept
        RecState = EconomyMrkv_init[0] % 2 == 1
        self.AggDemandFac_init = self.ADFunc(self.CratioNow_init,RecState)
        
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
        for ThisType in self.agents:
            ThisType.read_shocks = True
            ThisType(**experiment_dict)
            ThisType.updateMrkvArray()
            ThisType.solveIfChanged()
            ThisType.initializeSim()
            ThisType.EconomyMrkvNow_hist = self.EconomyMrkvNow_hist
            ThisType.hitWithRecessionShock()
            PopCount += ThisType.AgentCount
        self.makeHistory()
        
        
           
        # Extract simulated consumption, labor income, and weight data
        cNrm_all    = np.concatenate([ThisType.history['cNrmNow'] for ThisType in self.agents], axis=1)
        Mrkv_hist   = np.concatenate([ThisType.history['MrkvNow'] for ThisType in self.agents], axis=1)
        pLvl_all    = np.concatenate([ThisType.history['pLvlNow'] for ThisType in self.agents], axis=1)
        TranShk_all = np.concatenate([ThisType.history['TranShkNow'] for ThisType in self.agents], axis=1)
        mNrm_all    = np.concatenate([ThisType.history['mNrmNow'] for ThisType in self.agents], axis=1)
        aNrm_all    = np.concatenate([ThisType.history['aNrmNow'] for ThisType in self.agents], axis=1)
        cLvl_all    = np.concatenate([ThisType.history['cLvlNow'] for ThisType in self.agents], axis=1)
        cLvl_all_splurge = np.concatenate([ThisType.history['cLvl_splurgeNow'] for ThisType in self.agents], axis=1)
        
        IndIncome = pLvl_all*TranShk_all*np.array(self.history['AggDemandFac'])[:,None] #changed this to AggDemandFac
        AggIncome = np.sum(IndIncome,1)
        AggCons   = np.sum(cLvl_all_splurge,1)
        
        # Function calculates the net present value of X, which can be income or consumption
        # Periods defintes the horizon of the NPV measure, R the interest rate at which future income is discounted
        def calculate_NPV(X,Periods,R):
            NPV_discount = np.zeros(Periods)
            for t in range(Periods):
                NPV_discount[t] = 1/(R**t)
            NPV = np.zeros(Periods)
            for t in range(Periods):
                NPV[t] = np.sum(X[0:t+1]*NPV_discount[0:t+1])    
            return NPV
        
    
        
        # calculate NPV
        NPV_AggIncome = calculate_NPV(AggIncome,self.act_T,ThisType.Rfree[0])
        NPV_AggCons   = calculate_NPV(AggCons,self.act_T,ThisType.Rfree[0])
        
        # calculate Cratio_hist
        if hasattr(self,'base_AggCons'):
            Cratio_hist = np.divide(AggCons,self.base_AggCons)
        else:
            Cratio_hist = np.divide(AggCons,AggCons)
        
                
        # Get initial Markov states
        Mrkv_init = np.concatenate([ThisType.history['MrkvNow'][0,:] for ThisType in self.agents])
        
        if Full_Output:
            return_dict = {'cNrm_all' : cNrm_all,
                           'TranShk_all' : TranShk_all,
                           'cLvl_all' : cLvl_all,
                           'pLvl_all' : pLvl_all,
                           'Mrkv_hist' : Mrkv_hist,
                           'Mrkv_init' : Mrkv_init,
                           'mNrm_all' : mNrm_all,
                           'aNrm_all' : aNrm_all,
                           'cLvl_all_splurge' : cLvl_all_splurge,
                           'NPV_AggIncome': NPV_AggIncome,
                           'NPV_AggCons': NPV_AggCons,
                           'AggIncome': AggIncome,
                           'AggCons': AggCons,
                           'Cratio_hist' : Cratio_hist}
        else:
            return_dict = {'NPV_AggIncome': NPV_AggIncome,
                           'NPV_AggCons':   NPV_AggCons,
                           'AggIncome':     AggIncome,
                           'AggCons':       AggCons,
                           'Cratio_hist':   Cratio_hist}    
                
        return return_dict

    def calcCFunc(self):
        StateCount = self.MrkvArray[0].shape[0]
        CFunc_all = []
        for i in range(StateCount):
            CFunc_i = []
            for j in range(StateCount):
                CFunc_i.append(CRule(self.intercept_prev[i,j], self.slope_prev[i,j]))
            CFunc_all.append(copy(CFunc_i))
        self.CFunc = CFunc_all
        
    def switchToCounterfactualMode(self):
        '''
        Very small method that swaps in the "big" Markov-state versions of some
        solution attributes, replacing the "small" two-state versions that are used
        only to generate the pre-recession initial distbution of state variables.
        It then prepares this type to create alternate shock histories so it can
        run counterfactual experiments.
        '''
        self.MrkvArray = self.MrkvArray_big
        self.intercept_prev = self.intercept_prev_big
        self.slope_prev = self.slope_prev_big
        self.calcCFunc()
        
        # Adjust simulation parameters for the counterfactual experiments
        self.act_T = T_sim
        for agent in self.agents:
            agent.getEconomyData(self)
            agent.switchToCounterfactualMode()
            
    def saveState(self):
        for agent in self.agents:
            agent.saveState()
            
    def storeBaseline(self, AggCons):
        self.base_AggCons = copy(AggCons)
        self.stored_solutions = dict()
        self.storeADsolution('baseline')
            
    def storeADsolution(self, name):
        self.stored_solutions[name] = HARKobject()
        self.stored_solutions[name].CFunc = copy(self.CFunc)
        self.stored_solutions[name].ADelasticity = self.ADelasticity
        self.stored_solutions[name].agent_solutions = []
        for i in range(len(self.agents)):
            self.stored_solutions[name].agent_solutions.append(copy(self.agents[i].solution))
                       
    def restoreADsolution(self,name):
        self.CFunc = self.stored_solutions[name].CFunc
        self.ADelasticity = self.stored_solutions[name].ADelasticity
        for i in range(len(self.agents)):
            self.agents[i].solution = self.stored_solutions[name].agent_solutions[i]
            self.agents[i].getEconomyData(self)
        
    def makeIdiosyncraticShockHistories(self):
        for agent in self.agents:
            agent.makeIdiosyncraticShockHistories()
            
    def solve(self):
        for agent in self.agents:
            agent.solve()
            
    def Macro2MicroCFunc(self, MacroCFunc):
        '''
        Converts the aggregate CFunc for Macro transitions to one for micro transitions
        '''
        dim = len(MacroCFunc)
        MicroCFunc = [[CRule(1.0,0.0) for i in range(dim*3)] for j in range(dim*3)]
        for i in range(dim*3):
            for j in range(dim*3):
                MicroCFunc[i][j] = MacroCFunc[int(np.floor(i/3))][int(np.floor(j/3))]
        return MicroCFunc
    
    def CompareCFuncConvergence(self,Old_Cfunc,New_Cfunc):
        dim=len(Old_Cfunc)
        DiffSlopes      = np.zeros((dim,dim))
        DiffIntercepts  = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                DiffSlopes[i,j]     = abs(New_Cfunc[i][j].slope - Old_Cfunc[i][j].slope)
                DiffIntercepts[i,j] = abs(New_Cfunc[i][j].intercept - Old_Cfunc[i][j].intercept)
        Slopes_Diff                         = np.linalg.norm(DiffSlopes)
        [i,j]                               = np.unravel_index(DiffSlopes.argmax(),DiffSlopes.shape)
        FromMrkState_Slopes_Largest_Diff    = int(np.floor(i/3))
        ToMrkState_Slopes_Largest_Diff      = int(np.floor(j/3))
        
        Intercept_Diff                      = np.linalg.norm(DiffIntercepts)
        [i,j]                               = np.unravel_index(DiffIntercepts.argmax(),DiffIntercepts.shape)
        FromMrkState_Intercept_Largest_Diff = int(np.floor(i/3))
        ToMrkState_Intercept_Largest_Diff   = int(np.floor(j/3))
        
        Total_Diff          = (Slopes_Diff**2 + Intercept_Diff**2)**0.5
        print('Diff in Slopes in CFunc: ', Slopes_Diff)
        print('Largest diff', np.max(DiffSlopes))
        print('Slope: Largest Diff from Mrk State: ', FromMrkState_Slopes_Largest_Diff)
        print('Slope: Largest Diff to Mrk State: ', ToMrkState_Slopes_Largest_Diff)
        
        print('Diff in Intercepts in CFunc: ', Intercept_Diff) 
        print('Largest diff', np.max(DiffIntercepts))
        print('Intercept: Largest Diff from Mrk State: ', FromMrkState_Intercept_Largest_Diff)
        print('Intercept: Largest Diff to Mrk State: ', ToMrkState_Intercept_Largest_Diff)
        #print('Total Diff in CFunc: ', Total_Diff)
        return Total_Diff
            
            
    def solveAD_TaxCut(self, num_max_iterations, convergence_cutoff=1E-3, name = None):
        self.ADelasticity = self.demand_ADelasticity
        self.update()    
        TaxCut_dict = {
             'RecessionShock' : False,
             'ExtendedUIShock' : False,
             'TaxCutShock' : True,
             'UpdatePrb': 1.0,
             'Splurge': 0.32,
             }
        dim = int(len(self.CFunc)/3)
        MacroCFunc = [[CRule(1.0,0.0) for i in range(dim)] for j in range(dim)]  
        # this sets the belief for agg consumption in next period when running experiment
        # The consumption rule has intercept 1 and slope 0 implying the prediction is simply 1 and thus baseline consumption ratio
        
        for i in range(num_max_iterations):
            print("Iteration ", i+1,":")
            taxcut_all_results = []
            for j in [0,1]:
                TaxCut_dict['EconomyMrkv_init'] = np.array([ 4,  6,  8, 10, 12, 14, 16, 18, 20*j,  22*j,  24*j, 26*j, 28*j, 30*j, 32*j, 34*j]) # Either 1 or 2 cycles of tax cuts
                this_taxcut_results = self.runExperiment(**TaxCut_dict)  #run under the belief as imposed above
                taxcut_all_results += [this_taxcut_results]  
           
            
            # In the following MacroCFunc is updated according to what actually happened
            # MacroCFunc[i][j] is the prediction function for Agg Cons today to tomorrow when one jumps from macro state i to j
            # However, we need to add here first a cascade of checks, comparing MacroCFunc as current with CRule
            MacroCFunc[0][4] = CRule(taxcut_all_results[0]['Cratio_hist'][0],0.0)    # Rule is set to be whatever consumption jumps to (when going from state 0 to state 4) with no slope because earlier consumption cannot predict what happens next at beginning
            for t in range(15):
                MacroCFunc[4+2*t][6+2*t] = CRule(taxcut_all_results[1]['Cratio_hist'][t+1],0.0)  #When tax shock is ongoing one assumes consumption to stay constant
            
            
            # When tax shocks end or repeats there is a discrete jump
            MacroCFunc[18][0] = CRule(taxcut_all_results[0]['Cratio_hist'][8],0.0)
            MacroCFunc[34][0] = CRule(taxcut_all_results[1]['Cratio_hist'][16],0.0)  
            
            # The next tries to get at the slope by checking how Consumption falls
            MacroCFunc[0][0] = CRule(1.0, np.mean((np.array(taxcut_all_results[0]['Cratio_hist'][9:13])-1)/(np.array(taxcut_all_results[0]['Cratio_hist'][8:12])-1)))  # when you return to normal state, aggregate consumption will not be equal to baseline
            
                             
            # The new MacroCFunc is imposed and the economy resolved
            # then the experiment above is repeated and the MacroCFunc updated again until believes are consistent with what happens           
            self.MacroCFunc = MacroCFunc
            Old_Cfunc = self.CFunc
            self.CFunc = self.Macro2MicroCFunc(MacroCFunc) 
            for agent in self.agents:
                agent.CFunc = self.CFunc
            print("solving again...")
            self.solve()
            
            Total_Diff = self.CompareCFuncConvergence(Old_Cfunc,self.CFunc)

            if Total_Diff < convergence_cutoff:
                print("Convergence criterion reached.")
                break
            else:                    
                print("Convergence criterion not reached.")
            
            
        if name != None:
            self.storeADsolution(name)
            
    def solveAD_Recession(self, num_max_iterations, convergence_cutoff=1E-3, name = None):
        self.ADelasticity = self.demand_ADelasticity
        self.update()   
        recession_dict = {
             'RecessionShock' : True,
             'ExtendedUIShock' : False,
             'TaxCutShock' : False,
             'UpdatePrb': 1.0,
             'Splurge': 0.32,
             }
        dim = int(len(self.CFunc)/3)
        MacroCFunc = [[CRule(1.0,0.0) for i in range(dim)] for j in range(dim)]
        for i in range(num_max_iterations):
            print("Iteration ", i+1,":")
            recession_all_results = []
            max_recession = 19
            for t in [0,max_recession-1]:
                recession_dict['EconomyMrkv_init'] = [1]*(t+1)
                this_recession_results = self.runExperiment(**recession_dict)
                recession_all_results += [this_recession_results]
            
            MacroCFunc[0][1] = CRule(recession_all_results[1]['Cratio_hist'][0],0.0) # consumption when you jump into recession from steady state
            # If stays in recession for a long time, then Cratio will hit an asymtote. Take advantage of that here:
            
            old_code = True
            
            if old_code:
                startt = 2
                endd = max_recession
                slope_if_recession     = (recession_all_results[1]['Cratio_hist'][startt+1] - recession_all_results[1]['Cratio_hist'][endd-1])/(recession_all_results[1]['Cratio_hist'][startt] - recession_all_results[1]['Cratio_hist'][endd-2])
                intercept_if_recession =  recession_all_results[1]['Cratio_hist'][startt+1] - slope_if_recession*(recession_all_results[1]['Cratio_hist'][startt]-1)
                MacroCFunc[1][1]       = CRule(intercept_if_recession,slope_if_recession) 
            else:
                # best fit curve:
                slope_if_recession, intercept_if_recession = np.polyfit((recession_all_results[1]['Cratio_hist'][0:19] - 1), recession_all_results[1]['Cratio_hist'][1:20], 1)
                MacroCFunc[1][1]       = CRule(intercept_if_recession,slope_if_recession) 
            
            
            # Behavior when recession is left: similar idea
            # slope_on_exit          = (recession_all_results[0]['Cratio_hist'][1] - recession_all_results[1]['Cratio_hist'][max_recession  ])/(recession_all_results[0]['Cratio_hist'][0] - recession_all_results[1]['Cratio_hist'][max_recession-1])
            # intercept_on_exit      =  recession_all_results[0]['Cratio_hist'][1] - slope_on_exit*(recession_all_results[0]['Cratio_hist'][0]-1)
            # MacroCFunc[1][0]       = CRule(intercept_on_exit,slope_on_exit)
            # Behavior when recession is left: this converges slightly faster
            slope = (recession_all_results[0]['Cratio_hist'][1]-1)/(recession_all_results[0]['Cratio_hist'][0]-1)
            MacroCFunc[1][0] = CRule(1.0,slope)
            

            # In normal times, Cratio=1 must map to Cratio=1, so just calculate slope
            slope_normal           = (recession_all_results[0]['Cratio_hist'][2]-1)/(recession_all_results[0]['Cratio_hist'][1]-1)
            MacroCFunc[0][0]       = CRule(1.0,slope_normal) 
            
            
            self.MacroCFunc = MacroCFunc
            Old_Cfunc  = self.CFunc
            New_Cfunc  = self.Macro2MicroCFunc(MacroCFunc)
            
            step = self.Cfunc_iter_stepsize 
            dim = int(len(self.CFunc))
            Step_Cfunc = [[CRule(1.0,0.0) for i in range(dim)] for j in range(dim)]
            for ii in range(dim):
                for jj in range(dim):
                    Step_Cfunc[ii][jj].slope      = Old_Cfunc[ii][jj].slope     + step*(New_Cfunc[ii][jj].slope-Old_Cfunc[ii][jj].slope)
                    Step_Cfunc[ii][jj].intercept  = Old_Cfunc[ii][jj].intercept + step*(New_Cfunc[ii][jj].intercept-Old_Cfunc[ii][jj].intercept)
                    
            self.CFunc = Step_Cfunc
            for agent in self.agents:
                agent.CFunc = self.CFunc
            print("solving again...")
            self.solve()
            
            
            Total_Diff = self.CompareCFuncConvergence(Old_Cfunc,self.CFunc)

            if Total_Diff < convergence_cutoff:
                print("Convergence criterion reached.")
                break
            else:                    
                print("Convergence criterion not reached.")
                
        if name != None:
            self.storeADsolution(name)
            
            
    # need a function that combines solveAD_recession and _taxcut
    # it needs to capture all possible markov transitions, i.e. from each period of the tax cut
    # one can either transition in the next tax cut period in or outside the recession!
            
    def solveAD_Recession_TaxCut(self, num_max_iterations, convergence_cutoff=1E-3, name = None):
        self.ADelasticity = self.demand_ADelasticity
        self.update()   
        recession_taxcut_dict = {
             'RecessionShock' : True,
             'ExtendedUIShock' : False,
             'TaxCutShock' : True,
             'UpdatePrb': 1.0,
             'Splurge': 0.32,
             }
        dim = int(len(self.CFunc)/3)
        MacroCFunc = [[CRule(1.0,0.0) for i in range(dim)] for j in range(dim)]
        for i in range(num_max_iterations):
            print("Iteration ", i+1,":")
            recession_all_results = []
            max_recession = 19
            
            if self.TaxCutContinuationProb_Rec > 0:
                cases = [0,1,2,3]
            else:
                cases = [0,1]
                
            for c in cases:
                if c == 0:
                    recession_taxcut_dict['EconomyMrkv_init'] = np.array([ 5,  6,  8, 10, 12, 14, 16, 18, 0])
                elif c == 1:
                    recession_taxcut_dict['EconomyMrkv_init'] = np.concatenate((np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1, np.array([1]*(11))))
                elif c == 2:
                    recession_taxcut_dict['EconomyMrkv_init'] = np.concatenate((np.array([ 4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34])+1, np.array([1]*(3))))
                elif c == 3:
                    recession_taxcut_dict['EconomyMrkv_init'] = np.concatenate((np.array([ 4,  6,  8, 10, 12, 14, 16, 18])+1, np.array([21, 22, 24, 26, 28, 30, 32, 34]), np.array([0]*3)))
                this_recession_results = self.runExperiment(**recession_taxcut_dict)
                recession_all_results += [this_recession_results]
                
            # consumption when you jump into recession/taxcut from steady state    
            MacroCFunc[0][5] = CRule(recession_all_results[0]['Cratio_hist'][0],0.0) 
            
            # When recession remains for the entire first 8q of tax cut
            for t in range(7):
                MacroCFunc[5+2*t][7+2*t] = CRule(recession_all_results[1]['Cratio_hist'][t+1],0.0)  #When tax shock is ongoing one assumes consumption to stay constant    
            
                
            
            # When recession ends during the first 8q of tax cut
            slope = (recession_all_results[0]['Cratio_hist'][1]-1)/(recession_all_results[0]['Cratio_hist'][0]-1)
            for t in range(7):    
                MacroCFunc[5+2*t][6+2*t] = CRule(1.0,slope) 

            # Once recession state is left
            for t in range(6):
                MacroCFunc[6+2*t][8+2*t] = CRule(recession_all_results[0]['Cratio_hist'][t+1+1],0.0)
                # 6 to 8 occurs from q1 to q2, thus t+2
                
            # Leaving tax cut
            MacroCFunc[18][0] = CRule(recession_all_results[0]['Cratio_hist'][8],0.0)    #When tax shocks end, there is a discrete jump
            MacroCFunc[19][0] = CRule(recession_all_results[0]['Cratio_hist'][8],0.0)    #When tax shocks end, there is a discrete jump                  
            MacroCFunc[19][1] = CRule(recession_all_results[1]['Cratio_hist'][8],0.0) 
            
            # When there is a second cycle of tax cuts
            if self.TaxCutContinuationProb_Rec > 0:
            
                MacroCFunc[19][21] = CRule(recession_all_results[2]['Cratio_hist'][8],0.0) # discrete jump
                #MacroCFunc[19][20] = CRule(recession_all_results[3]['Cratio_hist'][8],0.0) # discrete jump
                
                # continuing recession
                for t in range(7):    
                    MacroCFunc[21+2*t][23+2*t] = CRule(recession_all_results[2]['Cratio_hist'][t+8+1],0.0)
                
                # end of recession
                slope = (recession_all_results[3]['Cratio_hist'][9]-1)/(recession_all_results[3]['Cratio_hist'][8]-1)
                for t in range(7):    
                    MacroCFunc[21+2*t][22+2*t] = CRule(1.0,slope) 
                    
                # Once recession state is left
                for t in range(7):
                    MacroCFunc[20+2*t][22+2*t] = CRule(recession_all_results[3]['Cratio_hist'][t+8+1],0.0)
                    
                # print("recession_all_results[3]['Cratio_hist'] 0 to 9",recession_all_results[3]['Cratio_hist'][0:9])
                # print("recession_all_results[3]['Cratio_hist'] 10 to 16",recession_all_results[3]['Cratio_hist'][9:16])
                # print("recession_all_results[1]['Cratio_hist']",recession_all_results[1]['Cratio_hist'][0:8])
                # print("recession_all_results[1]['Cratio_hist']",recession_all_results[1]['Cratio_hist'][8:19])    
                
                # Leaving tax cut
                MacroCFunc[34][0] = CRule(recession_all_results[3]['Cratio_hist'][16],0.0)    
                MacroCFunc[35][0] = CRule(recession_all_results[3]['Cratio_hist'][16],0.0)                      
                MacroCFunc[35][1] = CRule(recession_all_results[2]['Cratio_hist'][16],0.0)
                
                
                
            
            # If stays in recession for a long time, then Cratio will hit an asymtote. Take advantage of that here:
            startt = 8
            slope_if_recession     = (recession_all_results[1]['Cratio_hist'][startt+1] - recession_all_results[1]['Cratio_hist'][max_recession-1])/(recession_all_results[1]['Cratio_hist'][startt] - recession_all_results[1]['Cratio_hist'][max_recession-2])
            intercept_if_recession =  recession_all_results[1]['Cratio_hist'][startt+1] - slope_if_recession*(recession_all_results[1]['Cratio_hist'][startt]-1)

            # # Slow move to new expectations
            # old_i = self.CFunc[3*1][3*1].intercept
            # old_s = self.CFunc[3*1][3*1].slope
            # step = self.Cfunc_iter_stepsize   
            # new_i = old_i + step * (intercept_if_recession-old_i)
            # new_s = old_s + step * (slope_if_recession-old_s)
            MacroCFunc[1][1]       = CRule(intercept_if_recession,slope_if_recession)           
            
            # In normal times, Cratio=1 must map to Cratio=1, so just calculate slope
            slope_normal           = np.mean((np.array(recession_all_results[0]['Cratio_hist'][9:19])-1)/(np.array(recession_all_results[0]['Cratio_hist'][8:18])-1))
            MacroCFunc[0][0]       = CRule(1.0,slope_normal) 

            
            self.MacroCFunc = MacroCFunc
            Old_Cfunc  = self.CFunc
            New_Cfunc  = self.Macro2MicroCFunc(MacroCFunc)
            
            step = self.Cfunc_iter_stepsize 
            dim = int(len(self.CFunc))
            Step_Cfunc = [[CRule(1.0,0.0) for i in range(dim)] for j in range(dim)]
            for ii in range(dim):
                for jj in range(dim):
                    Step_Cfunc[ii][jj].slope      = Old_Cfunc[ii][jj].slope     + step*(New_Cfunc[ii][jj].slope-Old_Cfunc[ii][jj].slope)
                    Step_Cfunc[ii][jj].intercept  = Old_Cfunc[ii][jj].intercept + step*(New_Cfunc[ii][jj].intercept-Old_Cfunc[ii][jj].intercept)
                    
            self.CFunc = Step_Cfunc
            for agent in self.agents:
                agent.CFunc = self.CFunc
            print("solving again...")
            self.solve()
            
            Total_Diff = self.CompareCFuncConvergence(Old_Cfunc,self.CFunc)

            if Total_Diff < convergence_cutoff:
                print("Convergence criterion reached.")
                break
            else:                    
                print("Convergence criterion not reached.")
                
        if name != None:
            self.storeADsolution(name)
            
            

class CRule(HARKobject):
    '''
    A class to represent agent beliefs about aggregate consumption dynamics.
    '''
    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope
        self.distance_criteria = ['slope', 'intercept']

    def __call__(self, Cnow):
        #Cnext = np.exp(self.intercept + self.slope*np.log(Cnow))
        Cnext = self.intercept + self.slope*(Cnow-1.0)        # Not logs!
        return Cnext