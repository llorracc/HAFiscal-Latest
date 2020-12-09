'''Configurable parameters. This file is edited by run_config.py by putting the new
values at the bottom of the file, superseding the default value.
'''
import numpy as np

# Parameters concerning the distribution of discount factors
DiscFacMean = 0.9637   # Mean intertemporal discount factor 
DiscFacSpread = 0.0253  # Half-width of uniform distribution of discount factors

# Parameters concerning Markov transition matrix

    # Normal
Urate_normal = 0.05          # Unemployment rate in normal times
Uspell_normal = 1.5          # Average duration of unemployment spell in normal times, in quarters
UBspell_normal = 2           # Average duration of unemployment benefits in normal times, in quarters
    # Recession
Urate_recession = 0.1        # Unemployment rate in recession
Uspell_recession = 4         # Average duration of unemployment spell in recession, in quarters
Rspell = 6                   # Expected length of recession, in quarters. If R_shared = True, must be an integer
R_shared = False             # Indicator for whether the recession shared (True) or idiosyncratic (False)
    # UI extension
UBspell_extended = 4         # Average duration of unemployment benefits when extended and assuming policy remains in place, in quarters
PolicyUBspell = 2            # Average duration that policy of extended unemployment benefits is in place
    # Tax Cut parameter
PolicyTaxCutspell = 2        # Average duration that policy of payroll tax cuts
TaxCutIncFactor = 1.02       # Amount by which the payroll tax cut increases after-tax income
TaxCutPeriods = 8            # Deterministic duration of tax cut 


UpdatePrb = 0.25    # probability of updating macro state (when sticky expectations is on)
Splurge = 0.32      # amount of income that is splurged


# Basic model parameters: CRRA, growth factors, unemployment parameters (for normal times)
CRRA = 1.0              # Coefficient of relative risk aversion
PopGroFac = 1.0 #1.01**0.25  # Population growth factor
PermGroFacAgg = 1.0 #1.01**0.25 # Technological growth rate or aggregate productivity growth factor

IncUnemp = 0.3          # Unemployment benefits replacement rate (proportion of permanent income)
IncUnempNoBenefits = 0.05          # Unemployment income when benefits run out (proportion of permanent income)

# Parameters concerning the initial distribution of permanent income 
pLvlInitMean = 0 
pLvlInitStd = 0.4           # Standard deviation of initial log permanent income 

# Parameters concerning grid sizes: assets, permanent income shocks, transitory income shocks
aXtraMin = 0.001        # Lowest non-zero end-of-period assets above minimum gridpoint
aXtraMax = 40           # Highest non-zero end-of-period assets above minimum gridpoint
aXtraCount = 48         # Base number of end-of-period assets above minimum gridpoints
aXtraExtra = [0.002,0.003] # Additional gridpoints to "force" into the grid
aXtraNestFac = 3        # Exponential nesting factor for aXtraGrid (how dense is grid near zero)
PermShkCount = 7        # Number of points in equiprobable discrete approximation to permanent shock distribution
TranShkCount = 7        # Number of points in equiprobable discrete approximation to transitory shock distribution

