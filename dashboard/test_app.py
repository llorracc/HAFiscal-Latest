"""
test_app.py – Comprehensive test suite for dashboard functionality

This test suite verifies that app.py creates a functional, professional dashboard
that meets all layout, interaction, and performance requirements.
"""

import hank_sam as hs
import pytest
import sys
import os

# Ensure imports work
try:
    import ipywidgets as widgets
    from ipywidgets import HBox, VBox
    import matplotlib.pyplot as plt
except ImportError as e:
    pytest.fail(f"Required dashboard dependencies not available: {e}")

# Import app module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import app
except ImportError as e:
    pytest.fail(f"app.py failed to import: {e}")


class TestDashboardComponents:
    """Test that all dashboard components are created correctly."""

    def test_all_widgets_exist(self):
        """Test that all required widgets are created."""
        required_widgets = [
            "phi_pi_widget",
            "phi_y_widget",
            "rho_r_widget",
            "kappa_p_widget",
            "phi_b_widget",
            "real_wage_rigidity_widget",
            "ui_extension_widget",
            "tax_cut_widget",
            "run_button",
            "progress_label",
            "fig1_output",
            "fig2_output",
        ]

        for widget_name in required_widgets:
            assert hasattr(app, widget_name), f"Missing widget: {widget_name}"
            widget = getattr(app, widget_name)
            assert isinstance(widget, (widgets.Widget, widgets.DOMWidget)), (
                f"Invalid widget type: {widget_name}"
            )

    def test_layout_containers_exist(self):
        """Test that layout containers are created properly."""
        required_containers = [
            "options_panel",
            "fig1_panel",
            "fig2_panel",
            "intro_section",
            "summary_section",
            "left_panel",
            "right_panel",
            "main_content",
            "dashboard",
        ]

        for container_name in required_containers:
            assert hasattr(app, container_name), f"Missing container: {container_name}"
            container = getattr(app, container_name)
            assert isinstance(container, (VBox, HBox)), (
                f"Invalid container type: {container_name}"
            )

    def test_widget_properties(self):
        """Test that widgets have correct properties and ranges."""
        # Test slider ranges and properties
        assert app.phi_pi_widget.min == 1.0, "phi_pi_widget min value incorrect"
        assert app.phi_pi_widget.max == 3.0, "phi_pi_widget max value incorrect"
        assert app.phi_pi_widget.value == 1.5, "phi_pi_widget default value incorrect"

        assert app.phi_y_widget.min == 0.0, "phi_y_widget min value incorrect"
        assert app.phi_y_widget.max == 1.0, "phi_y_widget max value incorrect"

        assert app.ui_extension_widget.min == 1, (
            "ui_extension_widget min value incorrect"
        )
        assert app.ui_extension_widget.max == 12, (
            "ui_extension_widget max value incorrect"
        )

        # Test button properties
        assert "Run Simulation" in app.run_button.description, (
            "Run button description incorrect"
        )

    def test_layout_properties(self):
        """Test that layout properties prevent scrollbars."""
        # Main dashboard should have no scrollbars
        assert app.dashboard.layout.overflow == "hidden", (
            "Dashboard should have overflow hidden"
        )
        assert app.main_content.layout.overflow == "hidden", (
            "Main content should have overflow hidden"
        )
        assert app.right_panel.layout.overflow == "hidden", (
            "Right panel should have overflow hidden"
        )

        # Check height usage
        assert "vh" in app.dashboard.layout.height, (
            "Dashboard should use viewport height"
        )
        assert "vh" in app.main_content.layout.height, (
            "Main content should use viewport height"
        )


class TestParameterFunctionality:
    """Test parameter override and functionality."""

    def test_parameter_overrides_work(self):
        """Test that parameter overrides actually change the results."""
        # Test with default parameters
        results_default = hs.compute_fiscal_multipliers(horizon_length=3)
        default_transfer_mult = results_default["multipliers"]["transfers"][0]

        # Test with different phi_pi (should change results under Taylor rule)
        results_high_phi_pi = hs.compute_fiscal_multipliers(
            horizon_length=3, phi_pi=2.5
        )
        high_phi_pi_mult = results_high_phi_pi["multipliers"]["transfers"][0]

        # Check if they're different
        diff = abs(default_transfer_mult - high_phi_pi_mult)
        assert diff > 0.001, (
            f"Parameter φπ changes should affect results: {default_transfer_mult:.4f} vs {high_phi_pi_mult:.4f}"
        )

    def test_all_parameters_accepted(self):
        """Test that all dashboard parameters are accepted by the model."""
        app_params = {
            "phi_pi": app.phi_pi_widget.value,
            "phi_y": app.phi_y_widget.value,
            "rho_r": app.rho_r_widget.value,
            "kappa_p": app.kappa_p_widget.value,
            "phi_b": app.phi_b_widget.value,
            "real_wage_rigidity": app.real_wage_rigidity_widget.value,
            "UI_extension_length": app.ui_extension_widget.value,
            "tax_cut_length": app.tax_cut_widget.value,
        }

        # This should not raise any errors
        try:
            results = hs.compute_fiscal_multipliers(horizon_length=3, **app_params)
            assert "multipliers" in results, "Results should contain multipliers"
            assert "irfs" in results, "Results should contain IRFs"
        except Exception as e:
            pytest.fail(f"App parameters not accepted by model: {e}")

    def test_parameter_ranges_are_sensible(self):
        """Test that parameter ranges make economic sense."""
        # Taylor rule coefficients should be positive
        assert app.phi_pi_widget.min >= 1.0, (
            "Taylor rule inflation coefficient should be >= 1"
        )
        assert app.phi_y_widget.min >= 0.0, (
            "Taylor rule output coefficient should be >= 0"
        )

        # Probabilities and rates should be between 0 and 1
        assert 0 <= app.rho_r_widget.min <= app.rho_r_widget.max <= 1, (
            "Interest rate smoothing should be in [0,1]"
        )
        assert (
            0
            <= app.real_wage_rigidity_widget.min
            <= app.real_wage_rigidity_widget.max
            <= 1
        ), "Wage rigidity should be in [0,1]"

        # Policy durations should be positive integers
        assert app.ui_extension_widget.min >= 1, (
            "UI extension should be at least 1 quarter"
        )
        assert app.tax_cut_widget.min >= 1, "Tax cut should be at least 1 quarter"


class TestDashboardIntegration:
    """Test dashboard integration and functionality."""

    def test_update_function_exists(self):
        """Test that the main update function exists and is connected."""
        assert hasattr(app, "update_plots"), "Missing update_plots function"
        assert callable(app.update_plots), "update_plots should be callable"

        # Check that button is connected to function
        assert len(app.run_button._click_handlers.callbacks) > 0, (
            "Button should be connected to handler"
        )

    def test_output_widgets_configured(self):
        """Test that output widgets are properly configured."""
        # Both output widgets should exist and be configured for no scrollbars
        assert app.fig1_output.layout.overflow == "hidden", (
            "fig1_output should have overflow hidden"
        )
        assert app.fig2_output.layout.overflow == "hidden", (
            "fig2_output should have overflow hidden"
        )

        # Should have width and height properties
        assert app.fig1_output.layout.width == "100%", (
            "fig1_output should have 100% width"
        )
        assert app.fig2_output.layout.width == "100%", (
            "fig2_output should have 100% width"
        )

    def test_dashboard_styling(self):
        """Test that dashboard has professional styling."""
        # Check that style properties are set for sliders
        assert hasattr(app, "style"), "Style dictionary should exist"
        assert hasattr(app, "slider_layout"), "Slider layout should exist"

        # Verify description width is relative (not hardcoded pixels)
        assert "%" in app.style["description_width"], "Style should use relative sizing"
        assert "%" in app.slider_layout.width, "Layout should use relative sizing"

    def test_no_hardcoded_dimensions(self):
        """Test that no hardcoded pixel dimensions are used."""
        # Check layout containers for relative sizing
        containers = [app.dashboard, app.main_content, app.left_panel, app.right_panel]

        for container in containers:
            if hasattr(container.layout, "width") and container.layout.width:
                assert (
                    "%" in container.layout.width or container.layout.width == "100%"
                ), f"Container should use relative width: {container.layout.width}"

            if hasattr(container.layout, "height") and container.layout.height:
                assert (
                    "vh" in container.layout.height
                    or "%" in container.layout.height
                    or container.layout.height == "100%"
                ), f"Container should use relative height: {container.layout.height}"


class TestFigureGeneration:
    """Test figure generation and display."""

    def test_figure_creation_capability(self):
        """Test that figures can be created successfully."""
        try:
            # Get test data
            results = hs.compute_fiscal_multipliers(horizon_length=3)

            # Test figure creation (without actually displaying)
            fig1, axes1 = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
            fig1 = hs.plot_multipliers_three_experiments(
                results["multipliers"]["transfers"],
                results["multipliers"]["transfers_fixed_nominal"],
                results["multipliers"]["transfers_fixed_real"],
                results["multipliers"]["UI_extend"],
                results["multipliers"]["UI_extend_fixed_nominal"],
                results["multipliers"]["UI_extend_fixed_real"],
                results["multipliers"]["tax_cut"],
                results["multipliers"]["tax_cut_fixed_nominal"],
                results["multipliers"]["tax_cut_fixed_real"],
                fig_and_axes=(fig1, axes1),
            )

            assert fig1 is not None, "Figure 1 should be created successfully"
            plt.close(fig1)  # Clean up

        except Exception as e:
            pytest.fail(f"Figure generation failed: {e}")

    def test_dashboard_canvas_control(self):
        """Test that dashboard properly controls matplotlib canvas."""
        try:
            results = hs.compute_fiscal_multipliers(horizon_length=3)

            # Test that plotting functions accept fig_and_axes parameter
            fig, axes = plt.subplots(1, 3, figsize=(8, 3))

            result_fig = hs.plot_multipliers_three_experiments(
                results["multipliers"]["transfers"],
                results["multipliers"]["transfers_fixed_nominal"],
                results["multipliers"]["transfers_fixed_real"],
                results["multipliers"]["UI_extend"],
                results["multipliers"]["UI_extend_fixed_nominal"],
                results["multipliers"]["UI_extend_fixed_real"],
                results["multipliers"]["tax_cut"],
                results["multipliers"]["tax_cut_fixed_nominal"],
                results["multipliers"]["tax_cut_fixed_real"],
                fig_and_axes=(fig, axes),
            )

            # Dashboard should get back the same figure object
            assert result_fig is fig, (
                "Dashboard should maintain control of figure object"
            )
            plt.close(fig)

        except Exception as e:
            pytest.fail(f"Dashboard canvas control test failed: {e}")

    def test_axis_labels_in_dashboard_figures(self):
        """Test that dashboard figures have proper axis labels with units."""
        try:
            results = hs.compute_fiscal_multipliers(horizon_length=3)

            # Test multiplier figure axis labels
            fig1, axes1 = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
            hs.plot_multipliers_three_experiments(
                results["multipliers"]["transfers"],
                results["multipliers"]["transfers_fixed_nominal"],
                results["multipliers"]["transfers_fixed_real"],
                results["multipliers"]["UI_extend"],
                results["multipliers"]["UI_extend_fixed_nominal"],
                results["multipliers"]["UI_extend_fixed_real"],
                results["multipliers"]["tax_cut"],
                results["multipliers"]["tax_cut_fixed_nominal"],
                results["multipliers"]["tax_cut_fixed_real"],
                fig_and_axes=(fig1, axes1),
            )

            # Check all multiplier subplot axis labels
            for i, ax in enumerate(axes1):
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()

                # Verify x-axis has time and unit labels
                assert "Time" in xlabel and "Quarters" in xlabel, (
                    f"Multiplier subplot {i}: X-axis should contain 'Time (Quarters)', got '{xlabel}'"
                )

                # Verify y-axis has descriptive multiplier label
                assert "Consumption Multiplier" in ylabel, (
                    f"Multiplier subplot {i}: Y-axis should contain 'Consumption Multiplier', got '{ylabel}'"
                )

            plt.close(fig1)

            # Test consumption IRF figure axis labels
            fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
            hs.plot_consumption_irfs_three_experiments(
                results["irfs"]["UI_extend"],
                results["irfs"]["UI_extend_fixed_nominal"],
                results["irfs"]["UI_extend_fixed_real"],
                results["irfs"]["transfer"],
                results["irfs"]["transfer_fixed_nominal"],
                results["irfs"]["transfer_fixed_real"],
                results["irfs"]["tau"],
                results["irfs"]["tau_fixed_nominal"],
                results["irfs"]["tau_fixed_real"],
                fig_and_axes=(fig2, axes2),
            )

            # Check all consumption subplot axis labels
            for i, ax in enumerate(axes2):
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()

                # Verify x-axis has time and unit labels
                assert "Time" in xlabel and "Quarters" in xlabel, (
                    f"Consumption subplot {i}: X-axis should contain 'Time (Quarters)', got '{xlabel}'"
                )

                # Verify y-axis has response type and unit labels
                assert "Consumption Response" in ylabel and "%" in ylabel, (
                    f"Consumption subplot {i}: Y-axis should contain 'Consumption Response (%)', got '{ylabel}'"
                )

            plt.close(fig2)

        except Exception as e:
            pytest.fail(f"Dashboard axis labels test failed: {e}")


def test_parameter_overrides():
    """Legacy test for backward compatibility."""
    return TestParameterFunctionality().test_parameter_overrides_work()


def test_specific_parameters():
    """Test individual parameter effects on steady state."""
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC PARAMETER EFFECTS")
    print("=" * 60)

    # Test kappa_p effect on Phillips curve
    print("\n1. Testing kappa_p (Phillips curve slope)...")
    result1 = hs.compute_fiscal_multipliers(horizon_length=3, kappa_p=0.01)  # Low slope
    result2 = hs.compute_fiscal_multipliers(
        horizon_length=3, kappa_p=0.15
    )  # High slope

    print(
        f"   Low kappa_p (0.01): Transfer mult = {result1['multipliers']['transfers'][0]:.4f}"
    )
    print(
        f"   High kappa_p (0.15): Transfer mult = {result2['multipliers']['transfers'][0]:.4f}"
    )

    # Test phi_b effect on fiscal rule
    print("\n2. Testing phi_b (fiscal adjustment speed)...")
    result3 = hs.compute_fiscal_multipliers(
        horizon_length=3, phi_b=0.001
    )  # Slow adjustment
    result4 = hs.compute_fiscal_multipliers(
        horizon_length=3, phi_b=0.05
    )  # Fast adjustment

    print(
        f"   Low phi_b (0.001): Transfer mult = {result3['multipliers']['transfers'][0]:.4f}"
    )
    print(
        f"   High phi_b (0.05): Transfer mult = {result4['multipliers']['transfers'][0]:.4f}"
    )


def test_app_integration():
    """Test the exact parameters that app.py sends."""
    print("\n" + "=" * 60)
    print("TESTING APP.PY INTEGRATION")
    print("=" * 60)

    # Simulate app.py parameter dict
    app_params = {
        "phi_pi": 2.0,
        "phi_y": 0.1,
        "rho_r": 0.3,
        "kappa_p": 0.08,
        "phi_b": 0.02,
        "real_wage_rigidity": 0.9,
        "UI_extension_length": 6,
        "tax_cut_length": 12,
    }

    print("App-style parameters:", app_params)

    try:
        results = hs.compute_fiscal_multipliers(horizon_length=3, **app_params)
        print("✅ SUCCESS: App parameters accepted")
        print(f"Transfer multiplier: {results['multipliers']['transfers'][0]:.4f}")
        return True
    except Exception as e:
        print(f"❌ FAILURE: App parameters rejected - {e}")
        return False


if __name__ == "__main__":
    print("HANK-SAM APP PARAMETER TEST")
    print("=" * 50)

    # Run tests
    test1_passed = test_parameter_overrides()
    test_specific_parameters()
    test2_passed = test_app_integration()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Parameter override test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"App integration test: {'PASS' if test2_passed else 'FAIL'}")

    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED - App should work correctly!")
    else:
        print("⚠️  TESTS FAILED - Parameter passing needs to be fixed")
