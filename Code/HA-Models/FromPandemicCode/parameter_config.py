'''Configurable parameters. This file is edited by run_config.py by putting the new
values at the bottom of the file, superseding the default value.
'''
import numpy as np

# Parameters concerning the distribution of discount factors
DiscFacMean = 0.9637   # Mean intertemporal discount factor 
DiscFacSpread = 0.0253  # Half-width of uniform distribution of discount factors

# Parameters concerning Markov transition matrix
Urate_normal = 0.05          # Unemployment rate in normal times
Uspell_normal = 1.5          # Average duration of unemployment spell in normal times, in quarters
UBspell_normal = 2           # Average duration of unemployment benefits in normal times, in quarters
Urate_recession_real = 0.1   # Actual unemployment rate in recession
Uspell_recession_real = 4    # Actual average duration of unemployment spell in recession, in quarters
Rspell_real = 6              # Actual expected length of recession, in quarters. If R_shared = True, must be an integer
Urate_recession_pcvd = 0.1   # Perceived unemployment rate in recession
Uspell_recession_pcvd = 4    # Perceived average duration of unemployment spell in recession, in quarters
Rspell_pcvd = 6              # Perceived expected length of recession, in quarters
R_shared = False        # Indicator for whether the recession shared (True) or idiosyncratic (False)

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

