# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:56:41 2021

@author: edmun
"""
'''
This is the main script for the paper
'''
def calculate_NPV(X,Periods,R):
    NPV_discount = np.zeros(Periods)
    for t in range(Periods):
        NPV_discount[t] = 1/(R**t)
    NPV = np.zeros(Periods)
    for t in range(Periods):
        NPV[t] = np.sum(X[0:t+1]*NPV_discount[0:t+1])    
    return NPV

cNrm_all    = recessionAD_history['cNrmNow']
Mrkv_hist   = recessionAD_history['MrkvNow']
pLvl_all    = recessionAD_history['pLvlNow']
TranShk_all = recessionAD_history['TranShkNow']
mNrm_all    = recessionAD_history['mNrmNow']
aNrm_all    = recessionAD_history['aNrmNow']
cLvl_all    = recessionAD_history['cLvlNow']
cLvl_all_splurge = recessionAD_history['cLvl_splurgeNow']

IndIncome = pLvl_all*TranShk_all*np.array(recessionAD_history_economy['AggDemandFacPrev'])[:,None] #changed this to AggDemandFac
AggIncome = np.sum(IndIncome,1)
AggCons   = np.sum(cLvl_all_splurge,1)
AggCons_nosplurge   = np.sum(cLvl_all,1)

NPV_AggIncome = calculate_NPV(AggIncome,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])
NPV_AggCons   = calculate_NPV(AggCons,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])
NPV_AggCons_nosplurge   = calculate_NPV(AggCons_nosplurge,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])

cNrm_all_UI    = recessionAD_historyUI['cNrmNow']
Mrkv_hist_UI   = recessionAD_historyUI['MrkvNow']
pLvl_all_UI    = recessionAD_historyUI['pLvlNow']
TranShk_all_UI = recessionAD_historyUI['TranShkNow']
mNrm_all_UI    = recessionAD_historyUI['mNrmNow']
aNrm_all_UI    = recessionAD_historyUI['aNrmNow']
cLvl_all_UI    = recessionAD_historyUI['cLvlNow']
cLvl_all_splurge_UI = recessionAD_historyUI['cLvl_splurgeNow']

IndIncome_UI = pLvl_all_UI*TranShk_all_UI*np.array(recessionAD_historyUI_economy['AggDemandFacPrev'])[:,None] #changed this to AggDemandFac
AggIncome_UI = np.sum(IndIncome_UI,1)
AggCons_UI   = np.sum(cLvl_all_splurge_UI,1)
AggCons_nosplurge_UI   = np.sum(cLvl_all_UI,1)

NPV_AggIncome_UI = calculate_NPV(AggIncome_UI,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])
NPV_AggCons_UI   = calculate_NPV(AggCons_UI,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])
NPV_AggCons_nosplurge_UI   = calculate_NPV(AggCons_nosplurge_UI,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])

mNrm_sum      = np.sum(mNrm_all,1)
mNrm_sum_UI   = np.sum(mNrm_all_UI,1)
aNrm_sum      = np.sum(aNrm_all,1)
aNrm_sum_UI   = np.sum(aNrm_all_UI,1)
TranShk_sum      = np.sum(TranShk_all,1)
TranShk_sum_UI   = np.sum(TranShk_all_UI,1)
pLvl_sum      = np.sum(pLvl_all,1)
pLvl_sum_UI   = np.sum(pLvl_all_UI,1)
Mrkv_hist_sum      = np.sum(Mrkv_hist,1)
Mrkv_hist_UI_sum   = np.sum(Mrkv_hist_UI,1)
cNrm_sum      = np.sum(cNrm_all,1)
cNrm_sum_UI   = np.sum(cNrm_all_UI,1)


i=2
icNrm_all_UI    = recessionAD_historyUI['cNrmNow'][:,i]
iMrkv_hist_UI   = recessionAD_historyUI['MrkvNow'][:,i]
ipLvl_all_UI    = recessionAD_historyUI['pLvlNow'][:,i]
iTranShk_all_UI = recessionAD_historyUI['TranShkNow'][:,i]
imNrm_all_UI    = recessionAD_historyUI['mNrmNow'][:,i]
iaNrm_all_UI    = recessionAD_historyUI['aNrmNow'][:,i]
icLvl_all_UI    = recessionAD_historyUI['cLvlNow'][:,i]
icLvl_all_splurge_UI = recessionAD_historyUI['cLvl_splurgeNow'][:,i]

icNrm_all    = recessionAD_history['cNrmNow'][:,i]
iMrkv_hist   = recessionAD_history['MrkvNow'][:,i]
ipLvl_all    = recessionAD_history['pLvlNow'][:,i]
iTranShk_all = recessionAD_history['TranShkNow'][:,i]
imNrm_all    = recessionAD_history['mNrmNow'][:,i]
iaNrm_all    = recessionAD_history['aNrmNow'][:,i]
icLvl_all    = recessionAD_history['cLvlNow'][:,i]
icLvl_all_splurge = recessionAD_history['cLvl_splurgeNow'][:,i]



iIncome = np.sum(IndIncome[:,i:i+1],1)
iCons   = np.sum(cLvl_all_splurge[:,i:i+1],1)
iCons_nosplurge   = np.sum(cLvl_all[:,i:i+1],1)
NPV_iIncome = calculate_NPV(iIncome,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])
NPV_iCons   = calculate_NPV(iCons,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])
NPV_iCons_nosplurge   = calculate_NPV(iCons_nosplurge,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])

iIncome_UI = np.sum(IndIncome_UI[:,i:i+1],1)
iCons_UI   = np.sum(cLvl_all_splurge_UI[:,i:i+1],1)
iCons_nosplurge_UI   = np.sum(cLvl_all_UI[:,i:i+1],1)

NPV_iIncome_UI = calculate_NPV(iIncome_UI,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])
NPV_iCons_UI   = calculate_NPV(iCons_UI,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])
NPV_iCons_nosplurge_UI   = calculate_NPV(iCons_nosplurge_UI,AggDemandEconomy.act_T,AggDemandEconomy.agents[0].Rfree[0])

plt.plot(NPV_iIncome_UI/NPV_iIncome)
plt.plot(NPV_iCons_UI/NPV_iCons)

plt.plot((imNrm_all[1:]/(iaNrm_all[:-1]*ipLvl_all[:-1]/ipLvl_all[1:]*AggDemandEconomy.agents[0].Rfree[0] + iTranShk_all[1:]*np.array(recessionAD_history_economy['AggDemandFacPrev'][1:])))[0:14])
plt.plot((imNrm_all_UI[1:]/(iaNrm_all_UI[:-1]*ipLvl_all_UI[:-1]/ipLvl_all_UI[1:]*AggDemandEconomy.agents[0].Rfree[0] + iTranShk_all_UI[1:]*np.array(recessionAD_historyUI_economy['AggDemandFacPrev'][1:])))[0:14])


for i in range(100):
    ipLvl_all_UI    = recessionAD_historyUI['pLvlNow'][:,i]
    plt.plot(ipLvl_all_UI[:-1]/ipLvl_all_UI[1:])



############################################
t=55
AggDemandEconomy.switch_shock_type("recession")
AggDemandEconomy.solve()
recession_dict = base_dict_agg.copy()
recession_dict.update(**recession_changes)
recession_dict['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
recession_dict['EconomyMrkv_init'][0:t+1] = np.array(recession_dict['EconomyMrkv_init'][0:t+1]) +1
this_recession_results = AggDemandEconomy.runExperiment(**recession_dict, Full_Output = False)

AggDemandEconomy.switch_shock_type("recession")
AggDemandEconomy.restoreADsolution(name = 'Recession')
recession_dict = base_dict_agg.copy()
recession_dict.update(**recession_changes)
recession_dict['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
recession_dict['EconomyMrkv_init'][0:t+1] = np.array(recession_dict['EconomyMrkv_init'][0:t+1]) +1
this_recession_results_AD = AggDemandEconomy.runExperiment(**recession_dict, Full_Output = False)

AggDemandEconomy.switch_shock_type("recessionUI")
AggDemandEconomy.solve()
recession_UI_dict = base_dict_agg.copy()
recession_UI_dict.update(**recession_UI_changes)
recession_UI_dict['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
recession_UI_dict['EconomyMrkv_init'][0:t+1] = np.array(recession_UI_dict['EconomyMrkv_init'][0:t+1]) +1
this_recession_UI_results = AggDemandEconomy.runExperiment(**recession_UI_dict, Full_Output = False)

AggDemandEconomy.switch_shock_type("recessionUI")
AggDemandEconomy.restoreADsolution(name = 'UI_Rec')
recession_UI_dict = base_dict_agg.copy()
recession_UI_dict.update(**recession_UI_changes)
recession_UI_dict['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
recession_UI_dict['EconomyMrkv_init'][0:t+1] = np.array(recession_UI_dict['EconomyMrkv_init'][0:t+1]) +1
this_recession_UI_results_AD = AggDemandEconomy.runExperiment(**recession_UI_dict, Full_Output = False)

NPV_AddInc_UI_Rec                       = getSimulationDiff(this_recession_results,this_recession_UI_results,'NPV_AggIncome') # Policy expenditure
NPV_Multiplier_UI_Rec_AD                = getNPVMultiplier(this_recession_results_AD,            this_recession_UI_results_AD,            NPV_AddInc_UI_Rec)   
plt.plot(NPV_AddInc_UI_Rec)
plt.plot(NPV_Multiplier_UI_Rec_AD)
    
    