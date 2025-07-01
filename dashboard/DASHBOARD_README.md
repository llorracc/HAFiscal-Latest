# HANK-SAM Model Interactive Dashboard

This interactive dashboard allows you to explore the HANK-SAM model's fiscal multipliers under different monetary and fiscal policy parameters.

## Features

- **Real-time parameter adjustment**: Modify model parameters and see results update automatically
- **Multiple policy regimes**: Compare standard Taylor rule, fixed nominal rate, and fixed real rate monetary policies
- **Three fiscal policies**: Analyze UI extensions, stimulus checks (transfers), and tax cuts
- **Visual outputs**: 
  - Fiscal multipliers over 20 quarters
  - Consumption impulse responses
  - Parameter summary display

## Requirements

- Python 3.7+
- All dependencies from the main project
- Additional: `voila` and `ipywidgets`

## Installation

If you haven't already installed the required packages:

```bash
pip install voila ipywidgets
```

## Running the Dashboard

### Method 1: Voila (Recommended)

From the dashboard directory, run:

```bash
voila app.py --no-browser
```

Then open your browser to the URL shown (typically http://localhost:8866).

### Method 2: Jupyter Notebook

You can also run `app.py` as a Jupyter notebook:

```bash
jupyter notebook app.py
```

Then run all cells to see the interactive dashboard.

## Tunable Parameters

### Monetary Policy
- **Taylor Rule π coefficient (φπ)**: Central bank's response to inflation (1.0-3.0)
- **Taylor Rule Y coefficient (φy)**: Central bank's response to output gap (0.0-1.0)
- **Taylor Rule inertia (ρr)**: Interest rate smoothing parameter (0.0-0.95)
- **Phillips curve slope (κp)**: Price flexibility parameter (0.01-0.2)

### Fiscal & Structural
- **Fiscal adjustment speed (φb)**: How quickly taxes adjust to debt (0.0-0.1)
- **Real wage rigidity**: Degree of wage stickiness (0.0-1.0)

### Policy Durations
- **UI extension length**: Duration of unemployment insurance extension (1-12 quarters)
- **Stimulus check length**: Duration of transfer payments (1-4 quarters)
- **Tax cut length**: Duration of tax reduction (1-16 quarters)

## Understanding the Output

### Fiscal Multipliers Plot
Shows the cumulative fiscal multiplier (output per dollar of fiscal spending) over time for each policy under different monetary regimes:
- Solid lines: Standard Taylor rule
- Dashed lines: Fixed nominal interest rate
- Dotted lines: Fixed real interest rate

### Consumption Response Plot
Shows percentage deviation of consumption from steady state in response to each fiscal policy.

### Key Insights
- UI extensions typically have the highest multipliers due to targeting liquidity-constrained households
- Fixed nominal/real rates amplify fiscal policy effects
- Tax cuts generally have lower multipliers due to savings leakage

## Architecture

```mermaid
graph LR
    A[app.py Dashboard] -->|imports| B[hank_sam.py]
    A -->|parameter overrides| C[compute_fiscal_multipliers]
    C -->|runs| D[Policy Experiments]
    D -->|returns| E[Multipliers & IRFs]
    E -->|displayed in| F[Interactive Plots]
    
    style A fill:#4a90e2
    style B fill:#7ed321
    style F fill:#f5a623
```

## Troubleshooting

1. **Import errors**: Make sure you're running from the dashboard directory
2. **Slow updates**: The model takes 10-30 seconds to run; this is normal
3. **Memory issues**: Close other applications if running on limited RAM

## Citation

If you use this dashboard in your research, please cite the original HAFiscal paper and acknowledge the dashboard implementation. 