# filename: do_all.py

# Import the exec function
from builtins import exec
import sys 
import os


#%% This script is a reduced version of do_all.py
# It only executes Step 4 of that file (so skips the estimation of the splurge, discount factors and robustness runs)
# For step 4, the simulation of the policy impacts, it only consider one policy, the extension of UI benefits
# and simulates N= agents, rather than N= as in the full simulation


print('Step 4: Comparing policies\n')
script_path = "AggFiscalMAIN.py"
os.chdir('FromPandemicCode')
exec(open(script_path).read())
os.chdir('../')
print('Concluded Step 4. \n')