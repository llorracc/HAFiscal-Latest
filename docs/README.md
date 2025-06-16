---
title: "Welfare and Spending Effects of Consumption Stimulus Policies"
author: "Christopher Carroll, Edmund Crawley, Ivan Fancovic, Hakon Tretvoll"
date: 2025/05/16

---

# Welfare and Spending Effects of Consumption Stimulus Policies

## Abstract

Using a heterogeneous agent model calibrated to match measured spending dynamics over four years following an income shock (Fagereng, Holm, and Natvik (2021)), we assess the effectiveness of three fiscal stimulus policies employed during recent recessions. Unemployment insurance (UI) extensions are the clear “bang for the buck” winner, especially when effectiveness is measured in utility terms. Stimulus checks are second best and have the advantage (over UI) of being scalable to any desired size. A temporary (two-year) cut in the rate of wage taxation is considerably less effective than the other policies and has negligible effects in the version of our model without a multiplier.

## Replication from a unix (macOS/linux) command line

### Interactive Reproduction Script

The easiest way to reproduce results is using the interactive script:

```bash
./reproduce.sh
```

This will present a menu with options to reproduce:
- LaTeX documents (a few minutes)
- Minimal computational results (~1 hour) 
- All computational results (1-2 days)

### Non-Interactive Reproduction

For automated/scripted usage, set the `REPRODUCE_TARGETS` environment variable:

```bash
REPRODUCE_TARGETS=docs ./reproduce.sh      # Documents only
REPRODUCE_TARGETS=min ./reproduce.sh       # Minimal results (~1 hour)
REPRODUCE_TARGETS=all ./reproduce.sh       # All results (1-2 days)
REPRODUCE_TARGETS=docs,min ./reproduce.sh  # Multiple targets
```

### Individual Scripts (Legacy)

You can also run individual reproduction scripts directly:

```bash
/bin/bash reproduce/reproduce_computed.sh      # All computational results
/bin/bash reproduce/reproduce_document_pdfs.sh # PDF documents  
/bin/bash reproduce_min.sh                     # Minimal results
```

Some of the statistics hard coded into the reproduce_computed.sh script are calculated using Stata. To reproduce these statistics, run the following do file in Stata:

```
Code/Empirical/make_liquid_wealth.do
```

## Step-by-step replication of computational results

Given the time it takes to replicate all the computational results in the paper, it could be useful to do them step-by-step. To do so, navigate to the directory `Code/Ha-Models` and set the appropriate flags in the file `do_all.py` for the steps to be executed. Then execute the command:  

```
python do_all.py
```

See **Runtime Requirements** below for estimates of the execution time of each step. 

**NOTE**: Our code using optimization routines should be deterministic. However, small differences in environments may lead to numerical differences in the results obtained. Such differences in steps 1 and 2 will lead to numerical differences in the results obtained in later steps as well. 

## Data Availability and Provenance Statements

The data used is saved in the replication packages and is below, along with how to find it publically:

- **Code/Empirical/rscfp2004.dta**: The summary extract data for SCF 2004 in Stata format. 

- **Code/Empirical/rscfp2004.csv**: The summary extract data for SCF 2004 in csv format. 

- **Code/Empirical/ccbal_answer.dta**: Small file created from the full public data set (main survey data) in Stata format. 

- **Code/Empirical/ccbal_answer.csv**: Small file created from the full public data set (main survey data) in csv format. 

The data is also available from the website of the Board of Governors of the Federal Reserve System at this link:

[Federal Reserve Board - 2004 Survey of Consumer Finances](https://www.federalreserve.gov/econres/scf_2004.htm)

Download and unzip the following files to reproduce our results:

- Main survey data: Stata version - **scf2004s.zip** $\Rightarrow$ **p04i6.dta**

- Summary Extract Data set: Stata format - **scfp2004s.zip** $\Rightarrow$ **rscfp2004.dta**

Place these .dta files in the same directory as **make_liquid_wealth.do** before running the file.

**Note**: When releasing new waves of the SCF, the summary extract data for older versions are inflation-adjusted. At the time of writing, downloading the data gives a file where all dollar variables are inflation-adjusted to 2022 dollars. With an adjusted version of **rscfp2004.dta** the numbers that are in dollar amounts will not replicate the numbers used in the paper (this applies to line 2 of Table 2, Panel B). 

### Summary of Availability

- All data **are** publicly available.

## Computational requirements

A current (2025) laptop or equivalent is sufficient to reproduce the results. 

### Software Requirements

- Stata (code was last run with version MP/18.0, but older versions should be fine too)

- Python. The results of the paper come from running the Python scripts above in this environment:
  channels:
  
  - conda-forge
    dependencies:
  
  - python=3.11.7
  
  - econ-ark=0.14.1
  
  - numpy=1.26.4
  
  - matplotlib=3.8.0
  
  - scipy=1.11.4
  
  - pandas=2.1.4
  
  - numba=0.59.0
  
  - pip
  
  - pip:
    
    - sequence-jacobian
  
  - the file "`binder\environment.yml`" lists these dependencies

### Runtime Requirements

The full results take several days to reproduce on a modern laptop. We recommend running the code in the 5 steps which can be found in:

`Code/HA-Models/do_all.py'

On a Windows 11 laptop with 32 gb RAM and an AMD Ryzen 9 5900HS 3.30 GHz processor the timings were approximately as follows

- Step 1: Runtime about 20 minutes

- Step 2: Runtime about 21 hours (~7 hours per education group)

- Step 3: Similar to Step 2

- Step 4: Runtime about 1 hour

- Step 5: Runtime about 65 hours

## List of tables

Table 1 create in
file: Code/HA-Models/Target_AggMPCX_LiquWealth/Estimation_BetaNablaSplurge.py
line: 680
output saved: Code/HA-Models/Target_AggMPCX_LiquWealth/Figures/MPC_WealthQuartiles_Table.tex

Table 2, Panel A: not generated in code, summarizes parameters mentioned in the text
Parameters are used in the file: Code/HA-Models/FromPandemicCode/EstimParameters.py
(Except $\kappa$ = ADelasticity which is used in the file: Code/HA-Models/FromPandemicCode/Parameters.py)

Table 2, Panel B: 
Lines 1-3: Generated by Code/Empirical/make_liquid_wealth.do
Lines 3-6: Not generated in code, summarizes parameters mentioned in the text
Parameters are used in the file Code/HA-Models/FromPandemicCode/EstimParameters.py

Table 3, all panels: not generated in code, summarizes parameters mentioned in the text
Parameters used in the file: Code/HA-Models/FromPandemicCode/Parameters.py

**The file Code/HA-Models/Results/AllResults_CRRA_2.0_R_1.01.txt is written by: Code\HA-Models\FromPandemicCode/EstimAggFiscalMAIN.py
lines: 1105 and 1111**

Table 4, Panel A: numbers from the file Code/HA-Models/Results/AllResults_CRRA_2.0_R_1.01.txt, lines 4 and 10; 14 and 20; 24 and 30

Table 4, Panel B: 
Line 1: Generated by Code/Empirical/make_liquid_wealth.do
Line 2: from the file Code/HA-Models/Results/AllResults_CRRA_2.0_R_1.01.txt, lines 5, 15 and 25

Table 5, Panel A:
Line 1: Generated by Code/Empirical/make_liquid_wealth.do
Line 2: from the file Code/HA-Models/Results/AllResults_CRRA_2.0_R_1.01.txt, line 37
Line 3: from the file Code/HA-Models/Results/AllResults_CRRA_2.0_R_1.01.txt, line 45

Table 5, Panel B: 
Line 1: Generated by Code/Empirical/make_liquid_wealth.do
Line 2: from the file Code/HA-Models/Results/AllResults_CRRA_2.0_R_1.01.txt, line 38
Line 3: from the file Code/HA-Models/Results/AllResults_CRRA_2.0_R_1.01.txt, line 44

Table 6 create in
file: Code/HA-Models/FromPandemicCode/AggFiscalMAIN.py
line: 56 (calling function Code/HA-Models/FromPandemicCode/Output_Results.py, where line 479 creates the table)
output: Code/HA-Models/FromPandemicCode/Tables/CRRA2/Multiplier.tex

Table 7 create in
file: Code/HA-Models/FromPandemicCode/AggFiscalMAIN.py
line: 56 (calling function Code/HA-Models/FromPandemicCode/Output_Results.py, 
which calls Code/HA-Models/FromPandemicCode/Welfare.py, where line 293 creates the table)
output: Code/HA-Models/FromPandemicCode/Tables/CRRA2/welfare6.tex

Table 8 create in
file: Code/HA-Models/FromPandemicCode/AggFiscalMAIN.py
line: 166 (calling function Code/HA-Models/FromPandemicCode/Output_Results.py, 
which calls Code/HA-Models/FromPandemicCode/Welfare.py, where line 326 creates the table)
output: Code/HA-Models/FromPandemicCode/Tables/Splurge0/welfare6_SplurgeComp.tex

Table 9: parametrization table, no results

## List of figures

Figure 1 create in
file: Code/HA-Models/Target_AggMPCX_LiquWealth/Estimation_BetaNablaSplurge.py
line: 596 and 633

Figure 2 create in 
file: Code/HA-Models/FromPandemicCode/CreateLPfig.py
line: 124 

Figure 3 (a) create in
file: Code/HA-Models/FromPandemicCode/CreateMPCfig.py
line: 77
Figure 3 (b) create in 
file: Code/HA-Models/FromPandemicCode/EvalConsDropUponUILeave.py
line: 112 

Figure 4 create in
file: Code/HA-Models/FromPandemicCode/AggFiscalMAIN.py
line: 56 (calling function Code/HA-Models/FromPandemicCode/Output_Results.py, where lines 108, 148, 186, 288, 295 and 301 create the six subfigures)

Figure 5 create in
file: Code/HA-Models/FromPandemicCode/HA-Fiscal-HANK-SAM-to-python.py
line: 1162, 1163, 1164, 1203, 1204, 1205 for panels a, b, c, d, e, f respectively.

Figure 6 create in
file: Code/HA-Models/FromPandemicCode/AggFiscalMAIN.py
line: 56 (calling function Code/HA-Models/FromPandemicCode/Output_Results.py, where line 335 will create the figure)

## References

Board of Governors of the Federal Reserve System. 2007. Survey of Consumer Finances (SCF), 2004 Summary Extract Public Data Dataset. Washington, DC. https://www.federalreserve.gov/econres/scf_2004.htm

## Acknowledgements

Some content on this page was copied from [Hindawi](https://www.hindawi.com/research.data/#statement.templates). Other content was adapted  from [Fort (2016)](https://doi.org/10.1093/restud/rdw057), Supplementary data, with the author's permission.
