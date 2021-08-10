



BaseCons = 100
Stimulus = 1
MPC = 0.8
AD = 0.25




Total_Add_Cons_P1 = MPC*Stimulus
for round in range(0,500000):
    Cratio = 1 + Total_Add_Cons_P1/BaseCons
    AD_Fac = Cratio**AD
    Add_Income_Fac = AD_Fac-1
    Total_Add_Cons_P1 = MPC*Stimulus + BaseCons*Add_Income_Fac

print(Total_Add_Cons_P1)

Total_Add_Cons_P2 = (1-MPC)*Stimulus
for round in range(0,500000):
    Cratio = 1 + Total_Add_Cons_P2/BaseCons
    AD_Fac = Cratio**AD
    Add_Income_Fac = AD_Fac-1
    Total_Add_Cons_P2 = (1-MPC)*Stimulus + BaseCons*Add_Income_Fac

print(Total_Add_Cons_P2)