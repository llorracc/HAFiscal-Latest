---
title: "Welfare and Spending Effects of Consumption Stimulus Policies"
author: "Christopher Carroll, Edmund Crawley, Ivan Fancovic, Hakon Tretvoll"
date: 2025/05/16

---
# Welfare and Spending Effects of Consumption Stimulus Policies

## Abstract

Using a heterogeneous agent model calibrated to match measured spending dynamics over four years following an income shock (Fagereng, Holm, and Natvik (2021)), we assess the effectiveness of three fiscal stimulus policies employed during recent recessions. Unemployment insurance (UI) extensions are the clear “bang for the buck” winner, especially when effectiveness is measured in utility terms. Stimulus checks are second best and have the advantage (over UI) of being scalable to any desired size. A temporary (two-year) cut in the rate of wage taxation is considerably less effective than the other policies and has negligible effects in the version of our model without a multiplier.

## Replication from a unix (macOS/linux) command line

To reproduce all the computational results of the paper (several days):

```
	/bin/bash reproduce/reproduce_computed.sh
```

To produce pdf version of the paper from a unix (macOS/linux) command line:

```
	/bin/bash reproduce/reproduce_document.sh
```

To reproduce both computational results and the paper:

```
	/bin/bash reproduce.sh
```

 To run a cut-down version of the results (\<1 hour):

 ```
	/bin/bash reproduce_min.sh
```

Some of the statistics hard coded into the reproduce_computed.sh script are calculated using Stata. To reproduce these statistics, run the following do file in Stata:

 ```
Code/Empirical/make_liquid_wealth.do
 ```

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

**Note**: When releasing new waves of the SCF, the summary extract data for older versions are inflation-adjusted. At the time of writing, downloading the data gives a file where all dollar variables are inflation-adjusted to 2022 dollars. With an adjusted version of **rscfp2004.dta** the numbers marked **USD** below will not replicate the numbers used in the paper. 

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

The full results take several days to reproduce on a modern laptop. We recommend running the code in the 4 steps which can be found in:

`Code/HA-Models/do_all.py'


## List of tables and figures

Table 1 create in
file: Code\HA-Models\FromPandemicCode\XXXX.py
line: xxx

.
.
.

Figure 1 create in
file: Code\HA-Models\FromPandemicCode\XXXX.py
line: xxx


## References

Board of Governors of the Federal Reserve System. 2007. Survey of Consumer Finances (SCF), 2004 Summary Extract Public Data Dataset. Washington, DC. https://www.federalreserve.gov/econres/scf_2004.htm

## Acknowledgements

Some content on this page was copied from [Hindawi](https://www.hindawi.com/research.data/#statement.templates). Other content was adapted  from [Fort (2016)](https://doi.org/10.1093/restud/rdw057), Supplementary data, with the author's permission.
