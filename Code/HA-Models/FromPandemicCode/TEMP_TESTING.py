

    
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
    this_Errors = [AggDemandEconomy.CFunc[3*MrkvSim[i]][3*MrkvSim[i+1]](this_UI_results['Cratio_hist'][i]) for i in range(19)]/this_UI_results['Cratio_hist'][0:19]
    this_MaxError = np.max(np.abs(this_Errors-1.0))
    Errors.append(this_Errors)
    MaxError.append(this_MaxError)
    UI_results.append(this_UI_results)