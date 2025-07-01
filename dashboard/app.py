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
# """HANK-SAM Model Interactive Dashboard.
# Author: Alan Lujan <alujan@jhu.edu>

# This Voila dashboard allows interactive exploration of the HANK-SAM model's
# fiscal multipliers under different monetary and fiscal policy parameters.
# """

# %%
import ipywidgets as widgets
from IPython.display import clear_output
from ipywidgets import HTML, HBox, Layout, VBox

# %%
# Import our refactored model module
import hank_sam as hs

# %%
# Create style for sliders - optimized for compact layout
style = {"description_width": "40%"}  # Relative description width
slider_layout = Layout(width="85%")  # Shorter relative width to prevent overflow

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
    readout=True,
    readout_format=".2f",
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
    readout=True,
    readout_format=".2f",
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
    readout=True,
    readout_format=".2f",
)

kappa_p_widget = widgets.FloatSlider(
    value=0.06191950464396284,
    min=0.01,
    max=0.2,
    step=0.005,
    description="Phillips curve slope (κp):",
    style=style,
    layout=slider_layout,
    continuous_update=False,
    readout=True,
    readout_format=".3f",
)

# Fiscal and Structural Parameters
phi_b_widget = widgets.FloatSlider(
    value=0.015,
    min=0.0,
    max=0.1,
    step=0.005,
    description="Fiscal adjustment (φb):",
    style=style,
    layout=slider_layout,
    continuous_update=False,
    readout=True,
    readout_format=".3f",
)

real_wage_rigidity_widget = widgets.FloatSlider(
    value=0.837,
    min=0.0,
    max=1.0,
    step=0.05,
    description="Real wage rigidity:",
    style=style,
    layout=slider_layout,
    continuous_update=False,
    readout=True,
    readout_format=".3f",
)

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
    readout=True,
)

tax_cut_widget = widgets.IntSlider(
    value=8,
    min=1,
    max=16,
    step=1,
    description="Tax cut (quarters):",
    style=style,
    layout=slider_layout,
    continuous_update=False,
    readout=True,
)

# %%
# Run button and progress
run_button = widgets.Button(
    description="▶ Run Simulation",
    button_style="",
    layout=Layout(width="85%", height="2.5em"),  # Relative sizing
)
# Set custom button styling
run_button.style.button_color = "#27ae60"  # Muted green

progress_label = widgets.Label(value="Ready to run simulation")

# %%
# Output widgets for truly responsive figures - adaptive to container size
fig1_output = widgets.Output(
    layout=Layout(
        overflow="hidden",
        width="100%",
        height="100%",  # Use full available height
        flex="1 1 auto",  # Grow and shrink with container
    )
)
fig2_output = widgets.Output(
    layout=Layout(
        overflow="hidden",
        width="100%",
        height="100%",  # Use full available height
        flex="1 1 auto",  # Grow and shrink with container
    )
)

# %% [markdown]
#


# %%
# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN UPDATE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════


def update_plots(*args) -> None:
    """Run the unified academic figure for the dashboard with enhanced feedback."""
    # Disable button and update progress
    run_button.disabled = True
    progress_label.value = "⏳ Running simulation... (15-30 seconds)"

    # Clear all outputs
    for output in [fig1_output, fig2_output]:
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
        # Run all experiments and get results
        results = hs.compute_fiscal_multipliers(**params)
        multipliers = results["multipliers"]
        irfs = results["irfs"]

        # Create figures with dashboard control over canvas
        import matplotlib.pyplot as plt

        # Figure 1: Fiscal Multipliers - guaranteed fit sizing
        with fig1_output:
            # Use smaller base figure that will definitely fit in containers
            # CSS flexbox will scale up as needed
            # Wide aspect ratio (3.43:1) maintained but at smaller base size
            fig1, axes1 = plt.subplots(
                1, 3, figsize=(9.6, 2.4), sharey=True, constrained_layout=True
            )

            fig1 = hs.plot_multipliers_three_experiments(
                multipliers["transfers"],
                multipliers["transfers_fixed_nominal"],
                multipliers["transfers_fixed_real"],
                multipliers["UI_extend"],
                multipliers["UI_extend_fixed_nominal"],
                multipliers["UI_extend_fixed_real"],
                multipliers["tax_cut"],
                multipliers["tax_cut_fixed_nominal"],
                multipliers["tax_cut_fixed_real"],
                fig_and_axes=(fig1, axes1),
            )
            if fig1 is not None:
                plt.show()

        # Figure 2: Consumption IRFs - guaranteed fit sizing
        with fig2_output:
            # Use same smaller base size as Figure 1 for consistency
            # CSS flexbox will scale both figures consistently
            fig2, axes2 = plt.subplots(
                1, 3, figsize=(9.6, 2.4), sharey=True, constrained_layout=True
            )

            fig2 = hs.plot_consumption_irfs_three_experiments(
                irfs["UI_extend"],
                irfs["UI_extend_fixed_nominal"],
                irfs["UI_extend_fixed_real"],
                irfs["transfer"],
                irfs["transfer_fixed_nominal"],
                irfs["transfer_fixed_real"],
                irfs["tau"],
                irfs["tau_fixed_nominal"],
                irfs["tau_fixed_real"],
                fig_and_axes=(fig2, axes2),
            )
            if fig2 is not None:
                plt.show()

        # Update summary statistics
        stimulus_mult_1yr = multipliers["transfers"][3]  # 1-year (4 quarters)
        ui_mult_1yr = multipliers["UI_extend"][3]
        tax_mult_1yr = multipliers["tax_cut"][3]

        summary_html = f"""
        <div style='display: flex; justify-content: space-between; margin: 0; padding: 0.8em; 
                    background-color: #f8f9fa; border-radius: 4px; color: #495057; font-size: 0.85em;'>
            <div><strong>Stimulus Check:</strong> {stimulus_mult_1yr:.2f}</div>
            <div><strong>UI Extension:</strong> {ui_mult_1yr:.2f}</div>
            <div><strong>Tax Cut:</strong> {tax_mult_1yr:.2f}</div>
        </div>
        """

        # Update the summary section (find and update the HTML widget)
        summary_section.children[1].value = summary_html

        progress_label.value = "✅ Simulation complete!"
        run_button.disabled = False

    except Exception as e:
        progress_label.value = f"❌ Error: {e!s}"
        run_button.disabled = False
        for output in [fig1_output, fig2_output]:
            with output:
                clear_output(wait=True)


# %%
# Connect button to update function
run_button.on_click(update_plots)

# %% [markdown]
#

# %%
# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: CREATE DASHBOARD LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# SIMPLE SIDEBAR - focused on core functionality
options_panel = VBox(
    [
        HTML(
            "<h2 style='margin: 0 0 0.5em 0; color: #2c3e50; font-size: 1em; font-weight: 600;'>Model Parameters</h2>",
        ),
        HTML(
            "<h3 style='margin: 0.5em 0 0.2em 0; color: #2c3e50; font-size: 0.8em; font-weight: 600;'>Monetary Policy</h3>"
        ),
        phi_pi_widget,
        phi_y_widget,
        rho_r_widget,
        kappa_p_widget,
        HTML(
            "<h3 style='margin: 0.8em 0 0.2em 0; color: #2c3e50; font-size: 0.8em; font-weight: 600;'>Fiscal & Structural</h3>"
        ),
        phi_b_widget,
        real_wage_rigidity_widget,
        ui_extension_widget,
        tax_cut_widget,
        HTML(
            "<h3 style='margin: 0.8em 0 0.2em 0; color: #2c3e50; font-size: 0.8em; font-weight: 600;'>Simulation</h3>"
        ),
        run_button,
        progress_label,
    ],
    layout=Layout(
        border="none",
        padding="0",
        margin="0",
        width="100%",
        height="100%",
        overflow_y="auto",  # Allow scrolling only within left panel if needed
        overflow_x="hidden",  # No horizontal scroll
    ),
)

# %%
# MAIN CONTENT - Two figure panels with responsive layout
fig1_panel = VBox(
    [
        HTML(
            "<h3 style='margin: 0 0 0.4em 0; color: #2c3e50; font-size: 0.9em; font-weight: 600;'>Fiscal Multipliers by Policy Type</h3>",
        ),
        fig1_output,
    ],
    layout=Layout(
        border="none",
        padding="0",
        margin="0 0 0.4em 0",  # Reduced margin for more space
        width="100%",
        height="50%",  # Equal split of available height
        flex="1 1 50%",  # Equal flex basis for both panels
        overflow="hidden",  # No scrollbars on figure panels
        display="flex",
        flex_direction="column",
    ),
)

fig2_panel = VBox(
    [
        HTML(
            "<h3 style='margin: 0 0 0.4em 0; color: #2c3e50; font-size: 0.9em; font-weight: 600;'>Consumption Response Functions</h3>",
        ),
        fig2_output,
    ],
    layout=Layout(
        border="none",
        padding="0",
        margin="0",
        width="100%",
        height="50%",  # Equal split of available height
        flex="1 1 50%",  # Equal flex basis for both panels
        overflow="hidden",  # No scrollbars on figure panels
        display="flex",
        flex_direction="column",
    ),
)

# Create introduction section with H1 title and larger body text
intro_section = VBox(
    [
        HTML(
            "<h1 style='margin: 0 0 0.4em 0; color: #2c3e50; font-size: 1.1em; font-weight: 600;'>"
            "HANK-SAM Fiscal Policy Analysis</h1>"
        ),
        HTML(
            "<p style='margin: 0 0 0.4em 0; color: #34495e; font-size: 0.85em; line-height: 1.4;'>"
            "This dashboard explores fiscal multipliers in a Heterogeneous Agent New Keynesian model with Search and Matching frictions. "
            "The model features heterogeneous households, unemployment dynamics, and endogenous job creation, making it ideal for analyzing fiscal policy effectiveness.</p>"
        ),
        HTML(
            "<p style='margin: 0 0 0.4em 0; color: #34495e; font-size: 0.85em; line-height: 1.4;'>"
            "Adjust the monetary and fiscal parameters below to explore how different policy regimes affect consumption multipliers. "
            "Compare results across three fiscal policies: stimulus checks, UI extensions, and tax cuts under standard Taylor rule, fixed nominal rate, and fixed real rate scenarios.</p>"
        ),
        HTML(
            "<p style='margin: 0 0 0.5em 0; color: #7f8c8d; font-size: 0.8em; line-height: 1.3; font-style: italic;'>"
            "Key insight: UI extensions typically generate the highest multipliers due to targeting unemployed households with high marginal propensities to consume.</p>"
        ),
    ],
    layout=Layout(
        width="100%",
        padding="0",
        margin="0 0 0.5em 0",
        flex="0 0 auto",
        overflow="hidden",
    ),
)

# Create summary statistics section (will be populated by simulation results)
summary_section = VBox(
    [
        HTML(
            "<h3 style='margin: 0 0 0.2em 0; color: #2c3e50; font-size: 0.85em; font-weight: 600;'>Key Multipliers (1-Year Horizon)</h3>"
        ),
        HTML(
            "<div id='summary-stats' style='margin: 0 0 0.5em 0; padding: 0.5em; background-color: #f8f9fa; "
            "border-radius: 4px; color: #495057; font-size: 0.75em;'>Run simulation to view key results...</div>"
        ),
    ],
    layout=Layout(
        width="100%",
        padding="0",
        margin="0 0 0.4em 0",
        flex="0 0 auto",  # Don't grow or shrink - use natural content size
        overflow="hidden",
    ),
)

# Create left panel with intro section above model parameters
left_panel = VBox(
    [intro_section, options_panel],
    layout=Layout(
        width="30%",
        height="100%",
        overflow="hidden",  # No scrollbars on left panel container
        padding="0.6em",
        background_color="#f5f5f5",
    ),
)
right_panel = VBox(
    [summary_section, fig1_panel, fig2_panel],
    layout=Layout(
        width="70%",
        height="100%",
        padding="0.6em",
        background_color="white",
        overflow="hidden",  # NO scrollbars allowed
        display="flex",  # Explicit flexbox
        flex_direction="column",  # Stack children vertically
        gap="0.2em",  # Small gap between elements
    ),
)

# Split horizontally: Options left (30%) -> Figures right (70%)
main_content = HBox(
    [left_panel, right_panel],
    layout=Layout(
        width="100%",
        height="90vh",  # Use more of viewport height
        overflow="hidden",  # Prevent outer scrollbars
        margin="0",
        padding="0",
    ),
)

# Complete dashboard
dashboard = VBox(
    [main_content],
    layout=Layout(
        width="100%",
        height="90vh",  # Use more of viewport height
        overflow="hidden",  # Master overflow control - NO SCROLLBARS
        margin="0",
        padding="0",
    ),
)

# %%
# Initialize with welcome message
with fig1_output:
    pass

# %%
# Display dashboard
dashboard
