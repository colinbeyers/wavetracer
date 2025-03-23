def compute_trajectory(H, T, k, omega, time_array):
    """
    Computes the particle trajectory in ocean waves.

    Parameters:
        H (float): Water depth.
        T (float): Wave period.
        k (float): Wavenumber.
        omega (float): Angular frequency.
        time_array (array-like): Array of time values.

    Returns:
        dict: Dictionary containing xi and zeta trajectories.
    """
    # Check if parameters are valid
    validate_parameters(H, T, k, omega)

    # Determine wave regime
    regime = categorize_wave_regime(H, k)

    # Compute excursion vectors based on regime
    if regime == "deep":
        xi, zeta = compute_deep_excursion(H, k, omega, time_array)
    elif regime == "shallow":
        xi, zeta = compute_shallow_excursion(H, k, omega, time_array)
    else:  # Intermediate
        xi, zeta = compute_intermediate_excursion(H, k, omega, time_array)

    return {"xi": xi, "zeta": zeta}
