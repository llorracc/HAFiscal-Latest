# HAFiscal Repository - AI Assistant Guide

## Quick Context for AI Assistants

**Project**: Heterogeneous Agent Fiscal Policy Analysis  
**Purpose**: Computational reproduction of "Welfare and Spending Effects of Consumption Stimulus Policies"  
**Authors**: Christopher Carroll, Edmund Crawley, Ivan Fancovic, Hakon Tretvoll  
**Status**: ✅ **Fully reproducible** with HARK 0.15.1  

## Key Technical Concepts

### Economic Model
- **Heterogeneous Agent Model**: Households with different education levels (dropout, high school, college)
- **Markov States**: Employment/unemployment with varying benefit durations
- **Fiscal Policies**: UI extensions, stimulus checks, tax cuts
- **Aggregate Demand Effects**: Endogenous productivity responses to consumption

### Computational Framework
- **HARK (econ-ark)**: Heterogeneous Agent Resource Kit for economic modeling
- **Markov Consumer Type**: State-dependent parameters (Rfree, PermGroFac, LivPrb)
- **Parameter Dictionaries**: Education-specific parameter sets with Markov state arrays
- **Simulation**: Multi-period agent-based simulation with policy shocks

### Key Parameters
- **Rfree**: Interest rate (now `np.array` format for HARK 0.15.1 compatibility)
- **PermGroFac**: Permanent income growth rates
- **LivPrb**: Survival probabilities
- **MrkvArray**: Markov transition matrices for employment states

## Repository Structure

### Core Computational Code
```
Code/HA-Models/FromPandemicCode/
├── AggFiscalModel.py          # Main model class (MarkovConsumerType)
├── EstimAggFiscalModel.py     # Estimation version of model
├── Parameters.py              # Parameter dictionaries (returnParameters function)
├── EstimParameters.py         # Estimation parameters
├── Simulate.py                # Main simulation driver
├── AggFiscalMAIN_reduced.py   # Reduced run entry point
└── HA-Fiscal-HANK-SAM-to-python.py  # HANK robustness checks
```

### Reproduction Scripts
```
reproduce/
├── reproduce_computed_min.sh  # ⭐ Quick test (<1 hour)
├── reproduce_computed.sh      # Full reproduction (several days)
└── reproduce.sh               # Complete paper reproduction
```

### Key Files for AI Understanding
- **`Code/HA-Models/reproduce_min.py`**: Entry point for computational reproduction
- **`Code/HA-Models/FromPandemicCode/Simulate.py`**: Main simulation orchestration
- **`Code/HA-Models/FromPandemicCode/Parameters.py`**: Parameter loading and configuration

## Recent Major Changes (HARK 0.15.1 Upgrade)

### Breaking Changes Fixed
1. **Rfree Format**: Changed from `[np.array(...)]` to `np.array(...)` 
2. **Parameter Dictionaries**: Updated all education types to use correct format
3. **get_Rfree Methods**: Fixed indexing for scalar vs array cases
4. **switch_shock_type**: Updated to set Rfree as numpy array

### Files Modified
- `Code/HA-Models/FromPandemicCode/AggFiscalModel.py`
- `Code/HA-Models/FromPandemicCode/EstimAggFiscalModel.py`
- `Code/HA-Models/FromPandemicCode/Parameters.py`
- `Code/HA-Models/FromPandemicCode/EstimParameters.py`

## Common AI Tasks & Solutions

### Running Computational Reproduction
```bash
# Quick test (recommended for AI testing)
./reproduce/reproduce_computed_min.sh

# Full reproduction (takes hours)
./reproduce/reproduce_computed.sh
```

### Understanding Parameter Structure
- **Education Groups**: 0=dropout, 1=high school, 2=college
- **Markov States**: Employment + unemployment benefit periods
- **Parameter Format**: `np.array(num_base_MrkvStates * base_value)`

### Debugging Common Issues
1. **Rfree Shape Errors**: Check parameter dictionaries use `np.array` not `[np.array]`
2. **Missing Figures**: Restore from git history if needed
3. **Indexing Errors**: Ensure `get_Rfree` methods handle scalar/array cases

## Key Functions for AI Analysis

### Parameter Loading
```python
# In Parameters.py
init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns, ... = returnParameters('Reduced_Run')
```

### Model Initialization
```python
# In AggFiscalModel.py
class AggFiscalType(MarkovConsumerType):
    def get_Rfree(self):
        # Returns Rfree values for current Markov states
```

### Simulation Execution
```python
# In Simulate.py
AggDemandEconomy.make_history()  # Runs the simulation
```

## Data Sources & Empirical Analysis

### Survey of Consumer Finances (SCF) 2004
- **Location**: `Code/Empirical/`
- **Files**: `rscfp2004.dta`, `ccbal_answer.dta`
- **Analysis**: `Code/Empirical/make_liquid_wealth.do` (Stata)

### Key Empirical Targets
- Liquid wealth to permanent income ratios
- Lorenz curve percentiles
- Unemployment rates by education level

## Output & Results

### Tables Generated
- **Table 1**: MPC by wealth quartiles
- **Table 4**: Policy multipliers
- **Table 5**: Welfare effects
- **Table 6**: Multiplier comparisons
- **Table 7**: Welfare analysis
- **Table 8**: Splurge comparison

### Figures Generated
- **Figure 1**: Estimation results
- **Figure 2**: Lifecycle profiles
- **Figure 3**: MPC analysis
- **Figure 4**: Policy effects
- **Figure 5**: HANK robustness
- **Figure 6**: Welfare analysis

## Environment & Dependencies

### Current Setup
- **Python**: 3.11.7
- **HARK**: 0.15.1 (✅ upgraded and tested)
- **Key Packages**: numpy, scipy, pandas, matplotlib, numba
- **LaTeX**: Full TeX Live distribution

### Docker Support
```bash
docker build -t hafiscal .
docker run -it --rm --memory=32g -v $(pwd):/home/hafiscal/hafiscal hafiscal
```

## AI Assistant Best Practices

### When Helping Users
1. **Test Changes**: Always run `./reproduce/reproduce_computed_min.sh` to verify fixes
2. **Check Parameters**: Verify Rfree format in parameter dictionaries
3. **Monitor Output**: Look for convergence messages and error traces
4. **Preserve History**: Document changes in appropriate files

### Common User Requests
- **"Fix Rfree shape error"**: Update parameter dictionaries to use `np.array` format
- **"Missing figure file"**: Restore from git history with `git show HEAD:path/to/file`
- **"Upgrade HARK version"**: Follow the documented upgrade process
- **"Run reproduction"**: Start with `reproduce_computed_min.sh` for testing

### Key Documentation References
- **Main README**: `README.md` - Complete project overview
- **Dependencies**: `DEPENDENCY_MANAGEMENT.md` - Environment setup
- **Upgrade History**: `HARK_0.15.1_Upgrade_Summary.md` - Recent changes
- **CI Setup**: `CI_SETUP.md` - Automated testing

## Quick Reference

### File Purposes
- **`Parameters.py`**: Parameter configuration and loading
- **`AggFiscalModel.py`**: Core economic model implementation
- **`Simulate.py`**: Simulation orchestration
- **`reproduce_computed_min.sh`**: Quick test script

### Key Variables
- **`num_base_MrkvStates`**: Number of Markov states (employment + unemployment periods)
- **`Rfree_base`**: Base interest rate (typically [1.01])
- **`PermGroFac_base`**: Base permanent income growth rates
- **`LivPrb_base`**: Base survival probabilities

### Error Patterns
- **"Rfree not the right shape"**: Parameter format issue
- **"invalid index to scalar variable"**: Indexing in get_Rfree methods
- **"FileNotFoundError: figures/"**: Missing output files

---

**Note**: This repository is actively maintained and the computational reproduction has been successfully tested with HARK 0.15.1. All major breaking changes have been resolved and documented. 