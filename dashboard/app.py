# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""HANK-SAM Model Interactive Dashboard.

This Voila dashboard allows interactive exploration of the HANK-SAM model's
fiscal multipliers under different monetary and fiscal policy parameters.
"""

# %%
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from ipywidgets import HTML, HBox, Layout, VBox

# %%
# Import our refactored model module
import hank_sam as hs

# %%
# Create style for sliders
style = {"description_width": "180px"}
slider_layout = Layout(width="320px")

# %%
# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: CREATE PARAMETER WIDGETS
# ═════════════════════════════════════════════════════════════════════════════

# Monetary Policy Parameters
phi_pi_widget = widgets.FloatSlider(
    value=1.5,
    min=1.0,
    max=3.0,
    step=0.1,
    description="Taylor Rule π coeff (φπ):",
    style=style,
    layout=slider_layout,
    continuous_update=False,
)

# %%
phi_y_widget = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=1.0,
    step=0.05,
    description="Taylor Rule Y coeff (φy):",
    style=style,
    layout=slider_layout,
    continuous_update=False,
)

# %%
rho_r_widget = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=0.95,
    step=0.05,
    description="Taylor Rule inertia (ρr):",
    style=style,
    layout=slider_layout,
    continuous_update=False,
)

# %%
kappa_p_widget = widgets.FloatSlider(
    value=0.06191950464396284,
    min=0.01,
    max=0.2,
    step=0.005,
    description="Phillips curve slope (κp):",
    style=style,
    layout=slider_layout,
    readout_format=".3f",
    continuous_update=False,
)

# %%
# Fiscal and Structural Parameters
phi_b_widget = widgets.FloatSlider(
    value=0.015,
    min=0.0,
    max=0.1,
    step=0.005,
    description="Fiscal adjustment (φb):",
    style=style,
    layout=slider_layout,
    readout_format=".3f",
    continuous_update=False,
)

# %%
real_wage_rigidity_widget = widgets.FloatSlider(
    value=0.837,
    min=0.0,
    max=1.0,
    step=0.05,
    description="Real wage rigidity:",
    style=style,
    layout=slider_layout,
    readout_format=".3f",
    continuous_update=False,
)

# %%
# Policy Duration Parameters
ui_extension_widget = widgets.IntSlider(
    value=4,
    min=1,
    max=12,
    step=1,
    description="UI extension (quarters):",
    style=style,
    layout=slider_layout,
    continuous_update=False,
)

# %%
tax_cut_widget = widgets.IntSlider(
    value=8,
    min=1,
    max=16,
    step=1,
    description="Tax cut (quarters):",
    style=style,
    layout=slider_layout,
    continuous_update=False,
)

# %%
# Run button and progress
run_button = widgets.Button(
    description="▶ Run Simulation",
    button_style="success",
    layout=Layout(width="180px", height="40px"),
)

progress_label = widgets.Label(value="Ready to run simulation")

# %%
# Output widgets for the 4 figures from hank_sam.py main
fig1_output = widgets.Output()  # plot_multipliers_three_experiments
fig2_output = widgets.Output()  # plot_consumption_irfs_three_experiments
fig3_output = widgets.Output()  # plot_consumption_irfs_three
fig4_output = widgets.Output()  # plot_multipliers_across_horizon

# %% [markdown]
#


# %%
# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN UPDATE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════


def update_plots(*args) -> None:
    """Run the exact same 4 figures as hank_sam.py main execution."""
    progress_label.value = "Running simulation... (15-30 seconds)"

    # Clear all outputs
    for output in [fig1_output, fig2_output, fig3_output, fig4_output]:
        with output:
            clear_output(wait=True)

    # Get parameter values
    params = {
        "phi_pi": phi_pi_widget.value,
        "phi_y": phi_y_widget.value,
        "rho_r": rho_r_widget.value,
        "kappa_p": kappa_p_widget.value,
        "phi_b": phi_b_widget.value,
        "real_wage_rigidity": real_wage_rigidity_widget.value,
        "UI_extension_length": ui_extension_widget.value,
        "tax_cut_length": tax_cut_widget.value,
    }

    try:
        # This matches the exact sequence in hank_sam.py main
        results = hs.compute_fiscal_multipliers(**params)
        multipliers = results["multipliers"]
        irfs = results["irfs"]

        # Figure 1: Compare multipliers across policies and monetary regimes
        with fig1_output:
            hs.plot_multipliers_three_experiments(
                multipliers["transfers"],
                multipliers["transfers_fixed_nominal"],
                multipliers["transfers_fixed_real"],
                multipliers["UI_extend"],
                multipliers["UI_extend_fixed_nominal"],
                multipliers["UI_extend_fixed_real"],
                multipliers["tax_cut"],
                multipliers["tax_cut_fixed_nominal"],
                multipliers["tax_cut_fixed_real"],
            )

        # Figure 2: Consumption IRFs for all combinations
        with fig2_output:
            hs.plot_consumption_irfs_three_experiments(
                irfs["UI_extend"],
                irfs["UI_extend_fixed_nominal"],
                irfs["UI_extend_fixed_real"],
                irfs["transfer"],
                irfs["transfer_fixed_nominal"],
                irfs["transfer_fixed_real"],
                irfs["tau"],
                irfs["tau_fixed_nominal"],
                irfs["tau_fixed_real"],
            )

        # Figure 3: Baseline consumption responses under standard Taylor rule
        with fig3_output:
            hs.plot_consumption_irfs_three(
                irfs["transfer"],
                irfs["UI_extend"],
                irfs["tau"],
            )

        # Figure 4: Evolution of multipliers over time (standard Taylor rule)
        with fig4_output:
            # Create the multiplier evolution plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                np.arange(20) + 1,
                multipliers["transfers"],
                label="Stimulus Check",
                color="green",
                linewidth=2.5,
            )
            plt.plot(
                np.arange(20) + 1,
                multipliers["UI_extend"],
                label="UI extensions",
                color="blue",
                linewidth=2.5,
            )
            plt.plot(
                np.arange(20) + 1,
                multipliers["tax_cut"],
                label="Tax cut",
                color="red",
                linewidth=2.5,
            )
            plt.legend(loc="lower right")
            plt.ylabel("C multipliers")
            plt.xlabel("quarters")
            plt.xlim(0.5, 12.5)
            plt.title("Fiscal Multipliers Across Time Horizon", fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        progress_label.value = "✓ Simulation complete!"

    except Exception as e:
        progress_label.value = f"❌ Error: {e!s}"
        for output in [fig1_output, fig2_output, fig3_output, fig4_output]:
            with output:
                pass


# %%
# Connect button to update function
run_button.on_click(update_plots)

# %% [markdown]
#

# %%
# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: CREATE DASHBOARD LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# OPTIONS panel - organized in 3 columns with actual sliders
monetary_policy_col = VBox(
    [
        HTML("<h4 style='margin: 5px 0; color: #333;'>Monetary Policy</h4>"),
        phi_pi_widget,
        phi_y_widget,
        rho_r_widget,
        kappa_p_widget,
    ],
    layout=Layout(width="30%", padding="10px"),
)

fiscal_structural_col = VBox(
    [
        HTML("<h4 style='margin: 5px 0; color: #333;'>Fiscal & Structural</h4>"),
        phi_b_widget,
        real_wage_rigidity_widget,
        ui_extension_widget,
        tax_cut_widget,
    ],
    layout=Layout(width="30%", padding="10px"),
)

controls_col = VBox(
    [
        HTML("<h4 style='margin: 5px 0; color: #333;'>Controls</h4>"),
        run_button,
        HTML("<br>"),
        progress_label,
    ],
    layout=Layout(width="25%", padding="10px", align_items="center"),
)

options_panel = VBox(
    [
        HTML(
            "<h2 style='margin: 10px 0; text-align: center; color: #333;'>HANK-SAM Model Dashboard - Options</h2>",
        ),
        HBox(
            [monetary_policy_col, fiscal_structural_col, controls_col],
            layout=Layout(justify_content="space-around"),
        ),
    ],
    layout=Layout(
        border="2px solid #333",
        padding="15px",
        margin="5px",
        background_color="#f8f9fa",
    ),
)

# %%
# MAIN CONTENT - 4 figures matching the wireframe
# FIG 4 gets the large left panel (single plot)
fig4_panel = VBox(
    [
        HTML(
            "<h3 style='text-align: center; margin: 5px 0; background: #fce4ec; padding: 5px;'>FIG 4 - Multiplier Evolution</h3>",
        ),
        fig4_output,
    ],
    layout=Layout(
        border="1px solid #ddd",
        padding="10px",
        margin="5px",
        width="48%",
        min_height="600px",
    ),
)

# FIG 1, 2, 3 are the 3-panel plots that stack on the right
fig1_panel = VBox(
    [
        HTML(
            "<h4 style='text-align: center; margin: 5px 0; background: #e3f2fd; padding: 5px;'>FIG 1 - Multipliers Comparison</h4>",
        ),
        fig1_output,
    ],
    layout=Layout(
        border="1px solid #ddd",
        padding="5px",
        margin="2px",
        width="98%",
        height="190px",
    ),
)

fig2_panel = VBox(
    [
        HTML(
            "<h4 style='text-align: center; margin: 5px 0; background: #e8f5e8; padding: 5px;'>FIG 2 - All IRF Combinations</h4>",
        ),
        fig2_output,
    ],
    layout=Layout(
        border="1px solid #ddd",
        padding="5px",
        margin="2px",
        width="98%",
        height="190px",
    ),
)

fig3_panel = VBox(
    [
        HTML(
            "<h4 style='text-align: center; margin: 5px 0; background: #fff3e0; padding: 5px;'>FIG 3 - Baseline IRFs</h4>",
        ),
        fig3_output,
    ],
    layout=Layout(
        border="1px solid #ddd",
        padding="5px",
        margin="2px",
        width="98%",
        height="190px",
    ),
)

# Arrange as: FIG4 (large left) | FIG1, FIG2, FIG3 (stacked right)
right_panel = VBox([fig1_panel, fig2_panel, fig3_panel], layout=Layout(width="50%"))

main_content = HBox(
    [fig4_panel, right_panel],
    layout=Layout(width="100%", min_height="650px"),
)

# %%
# Complete dashboard
dashboard = VBox([options_panel, main_content])

# %%
# Initialize with welcome message
with fig1_output:
    pass

# %%
# Display dashboard
dashboard
