"""Test file to verify app.py parameter functionality works correctly."""

import hank_sam as hs


def test_parameter_overrides():
    """Test that parameter overrides actually change the results."""
    print("Testing parameter override functionality...")

    # Test 1: Run with default parameters
    print("\n1. Running with default parameters...")
    results_default = hs.compute_fiscal_multipliers(horizon_length=5)
    default_transfer_mult = results_default["multipliers"]["transfers"][0]
    print(f"   Default transfer multiplier (impact): {default_transfer_mult:.4f}")

    # Test 2: Run with different phi_pi (should change results under Taylor rule)
    print("\n2. Running with phi_pi=2.5 (higher than default 1.5)...")
    results_high_phi_pi = hs.compute_fiscal_multipliers(horizon_length=5, phi_pi=2.5)
    high_phi_pi_mult = results_high_phi_pi["multipliers"]["transfers"][0]
    print(f"   High phi_pi transfer multiplier (impact): {high_phi_pi_mult:.4f}")

    # Test 3: Check if they're different
    diff = abs(default_transfer_mult - high_phi_pi_mult)
    print(f"\n3. Difference in multipliers: {diff:.4f}")

    if diff > 0.001:
        print("   ✅ SUCCESS: Parameters are being applied (multipliers changed)")
        return True
    else:
        print("   ❌ FAILURE: Parameters are NOT being applied (multipliers unchanged)")
        return False


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
