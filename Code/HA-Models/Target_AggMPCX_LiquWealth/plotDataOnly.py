import sys
import os
import numpy as np
import random
from copy import deepcopy
import pandas as pd

# Import needed tools from HARK
from HARK.distribution import Uniform
from HARK.utilities import getPercentiles, getLorenzShares, make_figs
from HARK.parallel import multiThreadCommands
from HARK.estimation import minimizeNelderMead
from HARK.ConsumptionSaving.ConsIndShockModel import *
#from HARK.cstwMPC.SetupParamsCSTW import init_infinite


# for plotting
import matplotlib.pyplot as plt

# for output
cwd             = os.getcwd()
folders         = cwd.split(os.path.sep)
top_most_folder = folders[-1]
if top_most_folder == 'Target_AggMPCX_LiquWealth':
    Abs_Path = cwd
else:
    Abs_Path = cwd + '\\Target_AggMPCX_LiquWealth'

# Define the agg MPCx targets from Fagereng et al. Figure 2; first element is same-year response, 2nd element, t+1 response etcc
Agg_MPCX_target = np.array([0.5056845, 0.1759051, 0.1035106, 0.0444222, 0.0336616])


plt.figure()
xAxis = np.arange(0,5)
plt.scatter(xAxis,Agg_MPCX_target,c='black', marker='o', s=80)
plt.legend(['Fagereng, Holm and Natvik (2021)'])
plt.xticks(np.arange(min(xAxis), max(xAxis)+1, 1.0))
plt.xlabel('year', fontsize=20)
plt.ylabel('% of lottery win spent', fontsize=20)
make_figs('AggMPC_LotteryWin_DataOnly', True , False, target_dir=Abs_Path+'/Figures/')

plt.show()  
