

    
MrkvSim = np.concatenate(([0],[3]*4,[2]*0,[1]*3,[0]*20)).astype(int)
UI_dict['EconomyMrkv_init'] = MrkvSim[1:]

Errors = []
MaxError = []
UI_results = []
for j in range(4):
    if j == 0:
        UI_dict['EconomyMrkv_init'] = np.array([3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
    elif j == 1:
        UI_dict['EconomyMrkv_init'] = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    elif j == 2:
        UI_dict['EconomyMrkv_init'] = np.array([3, 2, 2, 0]) 
    elif j == 3:
        UI_dict['EconomyMrkv_init'] = np.array([3, 0]) 
    this_UI_results = AggDemandEconomy.runExperiment(**UI_dict)
    MrkvSim = np.concatenate(([0],UI_dict['EconomyMrkv_init'],[0]*20))
    cratio_hist = np.concatenate(([1.0],this_UI_results['Cratio_hist'][0:18]))
    this_Errors = [AggDemandEconomy.CFunc[3*MrkvSim[i]][3*MrkvSim[i+1]](cratio_hist[i]) for i in range(19)]/this_UI_results['Cratio_hist'][0:19]
    this_MaxError = np.max(np.abs(this_Errors-1.0))
    Errors.append(this_Errors)
    MaxError.append(this_MaxError)
    UI_results.append(this_UI_results)
    
    
AggDemandEconomy.restoreADsolution(name = 'Recession')
recession_dict = base_dict_agg.copy()
recession_dict.update(**recession_changes)
Errors = []
MaxError = []
UI_results = []
for j in range(4):
    if j == 0:
        recession_dict['EconomyMrkv_init'] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
    elif j == 1:
        recession_dict['EconomyMrkv_init'] = np.array([ 1])
    elif j == 2:
        recession_dict['EconomyMrkv_init'] = np.array([1,1,1]) 
    elif j == 3:
        recession_dict['EconomyMrkv_init'] = np.array([1, 0]) 
    this_UI_results = AggDemandEconomy.runExperiment(**recession_dict)
    MrkvSim = np.concatenate(([0],recession_dict['EconomyMrkv_init'],[0]*50))
    cratio_hist = np.concatenate(([1.0],this_UI_results['Cratio_hist'][0:38]))
    this_Errors = [AggDemandEconomy.CFunc[3*MrkvSim[i]][3*MrkvSim[i+1]](cratio_hist[i]) for i in range(39)]/this_UI_results['Cratio_hist'][0:39]
    this_MaxError = np.max(np.abs(this_Errors-1.0))
    Errors.append(this_Errors)
    MaxError.append(this_MaxError)
    UI_results.append(this_UI_results)