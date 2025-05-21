# filename: do_all.py

# Import the exec function
from builtins import exec
import sys 
import os

#%%
# Step 1: Estimation of the splurge factor: 
# This file replicates the results from section 3.1 in the paper, creates Figure 1 (in Target_AggMPCX_LiquWealth/Figures),
# and saves results in Target_AggMPCX_LiquWealth as .txt files to be used in the later steps.
print('Step 1: Estimating the splurge factor\n')
script_path = "Target_AggMPCX_LiquWealth/Estimation_BetaNablaSplurge.py"
exec(open(script_path).read())
print('Concluded Step 1.\n\n')


#%%
# Step 2: Baseline results. Estimate the discount factor distributions and plot figure 2. This replicates results from section 3.3.3 in the paper. 
print('Step 2: Estimating discount factor distributions (this takes a while!)\n')
os.chdir('FromPandemicCode')
exec(open("EstimAggFiscalMAIN.py").read())
exec(open("CreateLPfig.py").read()) # No argument -> create baseline figure
os.chdir('../')
print('Concluded Step 2.\n\n')


#%%
# Step 3: Robustness results. Estimate discount factor distributions with Splurge = 0. The results for Splurge = 0 are in the Online Appendix.
print('Step 3: Robustness results (note: this repeats step 2)\n')
run_robustness_results = False  
if run_robustness_results:
    os.chdir('FromPandemicCode')
    # Order of input arguments: interest rate, risk aversion, replacement rate w/benefits, replacement rate w/o benefits, Splurge   
    sys.argv = ['EstimAggFiscalMAIN.py', '1.01', '2.0', '0.7', '0.5', '0']    
    exec(open("EstimAggFiscalMAIN.py").read())
    os.chdir('../')
else:
    print('Skipping robustness for Splurge = 0 this time (see do_all.py line 32)')
print('Concluded Step 3.\n\n')


#%%
# Step 4: Solves the HANK and SAM model in Section 5 and creates Figure 5.
print('Step 4: HANK Robustness Check\n')
os.chdir('FromPandemicCode')

# compute household Jacobians
script_path = 'HA-Fiscal-HANK-SAM.py'
os.system("python " + script_path) 

# run HANK-SAM experiments
script_path = 'HA-Fiscal-HANK-SAM-to-python.py'
os.system("python " + script_path)  
os.chdir('../')
print('Concluded Step 4. \n')


#%%
# Step 5: Comparing fiscal stimulus policies: This file replicates the results from section 4 in the paper, 
# creates Figure 4 (located in FromPandemicCode/Figures), creates tables (located in FromPandemicCode/Tables)
# and creates robustness results for the case where the Splurge = 0.
# This also creates Figure 6 which uses results from Step 4 (hence, the order is different than in the presentation in the paper). 
print('Step 4: Comparing policies\n')
script_path = "AggFiscalMAIN.py"
os.chdir('FromPandemicCode')
exec(open(script_path).read())
os.chdir('../')
print('Concluded Step 5. \n')
