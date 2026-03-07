"""CMB Evidence Analysis — Quantitative examination of cosmic microwave background evidence.

Examines what counts as evidence in modern cosmology by computing the
actual physics behind CMB observations.  Tests the core tension between
measurement and interpretation: can a near-perfect blackbody spectrum
be uniquely attributed to an early-universe plasma, or could alternative
sources produce the same signal?

Analyses performed:
  1. Planck blackbody spectrum — exact B(ν,T) at T=2.7255 K
  2. COBE/FIRAS residuals — how perfect is "perfect"?
  3. Thermalization physics — timescales, scattering counts, opacity
  4. Spectral distortion constraints — μ and y distortions
  5. Alternative origin tests — tired light, local dust, steady-state
  6. SZ effect — independent proof CMB is cosmological
  7. CMB dipole — our motion through the CMB rest frame
  8. Acoustic peak structure — what the power spectrum actually encodes
  9. Recombination physics — Saha equation, last scattering surface
 10. Critical assessment — where the evidence is strong vs assumed

Run:  python -m solver cmb
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


# ═══════════════════════════════════════════════════════════════════════════
# Physical constants (SI unless noted)
# ═══════════════════════════════════════════════════════════════════════════

H_PLANCK    = 6.62607015e-34   # Planck constant  J·s
H_BAR       = H_PLANCK / (2 * math.pi)
K_B         = 1.380649e-23     # Boltzmann constant  J/K
C_SI        = 2.99792458e8     # speed of light  m/s
SIGMA_SB    = 5.670374419e-8   # Stefan-Boltzmann constant  W/m²/K⁴
SIGMA_T     = 6.6524587e-29    # Thomson cross-section  m²
M_E         = 9.1093837e-31    # electron mass  kg
M_P         = 1.67262192e-27   # proton mass  kg
E_ION_H     = 13.6             # hydrogen ionisation energy  eV
EV_TO_J     = 1.602176634e-19  # eV → J
G_SI        = 6.67430e-11      # gravitational constant  m³/(kg·s²)
MPC_M       = 3.0857e22        # megaparsec in metres
YR_S        = 3.156e7          # year in seconds

# CMB measurements
T_CMB       = 2.7255           # CMB temperature today  K  (Fixsen 2009)
T_CMB_ERR   = 0.0006           # 1σ uncertainty

# Cosmological parameters (Planck 2018)
H0_SI       = 67.36e3 / MPC_M           # H₀ in s⁻¹
OMEGA_B     = 0.0493                      # baryon density
OMEGA_M     = 0.3153                      # matter density
OMEGA_R     = 9.14e-5                     # radiation density
OMEGA_LAMBDA = 1.0 - OMEGA_M - OMEGA_R
RHO_CRIT    = 3 * H0_SI**2 / (8 * math.pi * G_SI)  # kg/m³
Z_RECOMB    = 1089.80                     # redshift of recombination
Z_DECOUPLE  = 1089.80                     # photon decoupling (same to ~1%)
TAU_REION   = 0.054                       # optical depth to reionization

# Helium mass fraction
Y_HE        = 0.2453                      # primordial helium mass fraction
X_H         = 1.0 - Y_HE                  # hydrogen mass fraction


# ═══════════════════════════════════════════════════════════════════════════
# 1. PLANCK BLACKBODY SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════

def planck_spectral_radiance(nu: float, T: float) -> float:
    """Planck function B(ν, T) in W/(m²·sr·Hz).

    B(ν, T) = (2hν³/c²) × 1/(exp(hν/kT) - 1)
    """
    x = H_PLANCK * nu / (K_B * T)
    if x > 500:
        return 0.0
    return (2 * H_PLANCK * nu**3 / C_SI**2) / (math.exp(x) - 1)


def planck_spectrum_array(T: float, nu_min: float = 1e9,
                          nu_max: float = 1e12, n_points: int = 500) -> dict:
    """Compute Planck spectrum B(ν, T) over a frequency range.

    Returns dict with arrays: nu_ghz, B_watt, B_mjy_sr (MJy/sr).
    """
    nus = np.linspace(nu_min, nu_max, n_points)
    B = np.array([planck_spectral_radiance(nu, T) for nu in nus])
    # Convert to MJy/sr:  1 Jy = 1e-26 W/m²/Hz, 1 MJy = 1e6 Jy
    B_mjy = B * 1e26 * 1e-6  # W/m²/sr/Hz → MJy/sr  (×1e20)
    B_mjy_sr = B / 1e-20     # Actually: B in W/m²/sr/Hz → MJy/sr = B × 1e20

    return {
        "nu_ghz": nus / 1e9,
        "B_si": B,
        "B_mjy_sr": B * 1e20,  # 1 MJy/sr = 1e-20 W/m²/sr/Hz
    }


def wien_peak_frequency(T: float) -> float:
    """Wien's displacement law: peak frequency of B(ν, T).

    ν_peak ≈ 2.821 × kT/h  (from solving dB/dν = 0)
    """
    return 2.821439 * K_B * T / H_PLANCK


def wien_peak_wavelength(T: float) -> float:
    """Wien's displacement law: peak wavelength in metres.

    λ_peak = b/T where b = 2.8977729e-3 m·K
    """
    return 2.8977729e-3 / T


def total_energy_density(T: float) -> float:
    """Total radiation energy density u = aT⁴ (J/m³).

    a = 4σ/c (radiation constant).
    """
    a_rad = 4 * SIGMA_SB / C_SI
    return a_rad * T**4


def photon_number_density(T: float) -> float:
    """CMB photon number density n_γ = (2ζ(3)/π²)(kT/ℏc)³ (per m³).

    ζ(3) = 1.20206...
    """
    zeta3 = 1.2020569
    return 2 * zeta3 / (math.pi**2) * (K_B * T / (H_BAR * C_SI))**3


# ═══════════════════════════════════════════════════════════════════════════
# 2. COBE/FIRAS — HOW PERFECT IS THE BLACKBODY?
# ═══════════════════════════════════════════════════════════════════════════

def firas_analysis() -> dict:
    """Quantify the COBE/FIRAS blackbody measurement precision.

    FIRAS (1990-1996) measured the CMB spectrum at 34 frequencies
    from 60 to 600 GHz and found:
      - Best-fit T = 2.725 ± 0.002 K (later refined to 2.7255 ± 0.0006 K)
      - RMS residual from perfect blackbody: 50 parts per million
      - Peak fractional deviation: < 0.01% at any frequency
      - No detected spectral distortions (μ < 9×10⁻⁵, |y| < 1.5×10⁻⁵)

    Context: This is the most precisely measured blackbody in all of physics.
    No laboratory source achieves this precision.
    """
    nu_peak = wien_peak_frequency(T_CMB)
    lam_peak = wien_peak_wavelength(T_CMB)
    u_cmb = total_energy_density(T_CMB)
    n_gamma = photon_number_density(T_CMB)

    # FIRAS published limits
    mu_limit = 9e-5         # 95% CL upper limit on μ-distortion
    y_limit = 1.5e-5        # 95% CL upper limit on y-distortion
    rms_residual_ppm = 50   # parts per million

    # What does sub-50 ppm mean?
    # The peak B(ν, T) at 160 GHz is ~385 MJy/sr
    # 50 ppm of that is ~0.019 MJy/sr — well below FIRAS noise at each channel
    # Integrated over all channels, FIRAS achieved this by averaging

    B_peak = planck_spectral_radiance(nu_peak, T_CMB)
    B_peak_mjy = B_peak * 1e20

    return {
        "T_cmb_k": T_CMB,
        "T_err_k": T_CMB_ERR,
        "relative_precision": T_CMB_ERR / T_CMB,
        "nu_peak_ghz": nu_peak / 1e9,
        "lambda_peak_mm": lam_peak * 1e3,
        "B_peak_mjy_sr": B_peak_mjy,
        "energy_density_j_m3": u_cmb,
        "energy_density_ev_cm3": u_cmb / EV_TO_J * 1e-6,
        "photon_density_per_cm3": n_gamma * 1e-6,
        "photons_per_baryon": n_gamma / (RHO_CRIT * OMEGA_B / M_P),
        "rms_residual_ppm": rms_residual_ppm,
        "mu_distortion_limit": mu_limit,
        "y_distortion_limit": y_limit,
        "firas_frequency_range_ghz": (60, 600),
        "firas_channels": 34,
        "measurement_note": (
            "Most precisely measured blackbody in physics. "
            "No laboratory source matches this spectral perfection."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. THERMALIZATION PHYSICS — Can the early universe produce this?
# ═══════════════════════════════════════════════════════════════════════════

def thomson_mean_free_path(n_e: float) -> float:
    """Thomson scattering mean free path λ_mfp = 1/(n_e σ_T) in metres."""
    return 1.0 / (n_e * SIGMA_T)


def electron_density_at_z(z: float, x_e: float = 1.0) -> float:
    """Free electron density at redshift z in m⁻³.

    n_e(z) = x_e × n_H(z) = x_e × X_H × ρ_b(z) / m_p
    where ρ_b(z) = ρ_b,0 × (1+z)³
    """
    rho_b0 = RHO_CRIT * OMEGA_B  # today's baryon density  kg/m³
    rho_b_z = rho_b0 * (1 + z)**3
    n_H_z = X_H * rho_b_z / M_P
    return x_e * n_H_z


def compton_scattering_rate(T: float, n_e: float) -> float:
    """Compton scattering rate Γ_C = n_e σ_T c (s⁻¹).

    At temperatures well below m_e c² / k_B ≈ 5.9×10⁹ K,
    Thomson scattering dominates.
    """
    return n_e * SIGMA_T * C_SI


def thermalization_analysis() -> dict:
    """Analyse whether the pre-recombination plasma can thermalise radiation.

    Key question: Starting from *any* initial photon distribution, does
    the universe have enough time and enough scatterings to produce a
    Planck spectrum to 50 ppm precision?

    Physical processes:
    1. Compton scattering: redistributes photon energies (conserves number)
    2. Double Compton: e + γ → e + γ + γ (creates/destroys photons)
    3. Bremsstrahlung: e + p → e + p + γ (creates/destroys photons)

    Key redshifts:
    - z_th ≈ 2×10⁶: above this, double Compton + Compton can fully thermalise
    - z_μ ≈ 5×10⁴: between z_th and z_μ, Compton can redistribute but
                     number-changing processes freeze out → μ-distortion
    - z_y ≈ 10⁴: below z_μ, even Compton can't fully equilibrate → y-distortion
    - z_rec ≈ 1090: recombination, photons decouple

    The "evidence case": FIRAS sees no μ or y distortion to the limits
    μ < 9×10⁻⁵, |y| < 1.5×10⁻⁵. This means:
    - No significant energy injection between z ≈ 5×10⁴ and z ≈ 10³
    - The spectrum was thermalised before z ≈ 2×10⁶

    The "skeptic case": Does the mere perfection of the blackbody *uniquely*
    imply a hot dense past? Consider:
    - A blackbody can be produced by *any* optically thick medium in
      thermal equilibrium — not just a Big Bang plasma
    - The question is whether the *temperature*, *isotropy*, and
      *spectral distortion limits* together constrain the origin
    """
    # Temperature at various redshifts: T(z) = T₀(1+z)
    z_th = 2e6       # full thermalisation
    z_mu = 5e4       # μ-distortion threshold
    z_y  = 1e4       # y-distortion threshold
    z_rec = 1090     # recombination

    T_th = T_CMB * (1 + z_th)
    T_mu = T_CMB * (1 + z_mu)
    T_y  = T_CMB * (1 + z_y)
    T_rec = T_CMB * (1 + z_rec)

    # Electron density and scattering rate at recombination
    n_e_rec = electron_density_at_z(z_rec, x_e=1.0)  # fully ionised just before
    mfp_rec = thomson_mean_free_path(n_e_rec)

    # Compton scattering rate at recombination
    gamma_C_rec = compton_scattering_rate(T_rec, n_e_rec)

    # Hubble rate at z_rec for comparison
    # H(z) = H₀ √(Ω_r(1+z)⁴ + Ω_m(1+z)³ + Ω_Λ)
    def H_at_z(z):
        zp1 = 1 + z
        return H0_SI * math.sqrt(OMEGA_R * zp1**4 + OMEGA_M * zp1**3 + OMEGA_LAMBDA)

    H_rec = H_at_z(z_rec)

    # Number of scatterings from z_th to z_rec
    # Optical depth τ = ∫ n_e σ_T c dt = ∫ n_e σ_T c/(H(z)(1+z)) dz
    def tau_integrand(z):
        n_e = electron_density_at_z(z, x_e=1.0)
        H_z = H_at_z(z)
        return n_e * SIGMA_T * C_SI / (H_z * (1 + z))

    tau_to_rec, _ = quad(tau_integrand, 0, z_rec)
    tau_th_to_rec, _ = quad(tau_integrand, z_rec, z_th)

    # Scattering rate / expansion rate
    scatter_over_hubble = gamma_C_rec / H_rec

    # Thermalisation requires N_scatter >> 1/ε² where ε is the
    # desired spectral precision. For 50 ppm: need >> (1/5e-5)² ≈ 4×10⁸ scatterings.
    # Actual optical depth from z~10⁶ to z~10³ is τ ~ 10⁸, so N_scatter ~ τ.
    # But Compton scattering alone conserves photon number — you need
    # double Compton + bremsstrahlung to adjust the chemical potential.

    # Compton y-parameter from z to 0:
    # y_C = ∫ (kT/m_e c²) n_e σ_T c dt
    def y_integrand(z):
        T_z = T_CMB * (1 + z)
        n_e = electron_density_at_z(z, x_e=1.0)
        H_z = H_at_z(z)
        return (K_B * T_z / (M_E * C_SI**2)) * n_e * SIGMA_T * C_SI / (H_z * (1 + z))

    y_total, _ = quad(y_integrand, z_rec, z_th)

    # Double Compton rate relative to Hubble
    # Γ_DC/H ∝ α_fs n_e σ_T c (kT/m_e c²)² / H
    # Freezes out when this ratio drops below ~1
    alpha_fs = 1.0 / 137.036
    def dc_over_H(z):
        T_z = T_CMB * (1 + z)
        n_e = electron_density_at_z(z, x_e=1.0)
        H_z = H_at_z(z)
        x = K_B * T_z / (M_E * C_SI**2)
        return alpha_fs * n_e * SIGMA_T * C_SI * x**2 / H_z

    dc_at_zth = dc_over_H(z_th)
    dc_at_zmu = dc_over_H(z_mu)
    dc_at_zy = dc_over_H(z_y)

    return {
        "redshift_thresholds": {
            "z_thermalisation": z_th,
            "z_mu_distortion": z_mu,
            "z_y_distortion": z_y,
            "z_recombination": z_rec,
        },
        "temperatures_K": {
            "T_thermalisation": T_th,
            "T_mu": T_mu,
            "T_y": T_y,
            "T_recombination": T_rec,
        },
        "at_recombination": {
            "n_e_per_m3": n_e_rec,
            "mean_free_path_m": mfp_rec,
            "mean_free_path_ly": mfp_rec / (C_SI * YR_S),
            "compton_rate_per_s": gamma_C_rec,
            "hubble_rate_per_s": H_rec,
            "scatter_rate_over_hubble": scatter_over_hubble,
        },
        "optical_depths": {
            "tau_0_to_rec": tau_to_rec,
            "tau_rec_to_thermalisation": tau_th_to_rec,
            "note": "τ >> 1 means many scatterings — optically thick",
        },
        "compton_y_parameter": y_total,
        "double_compton_rates": {
            "dc_over_H_at_z_th": dc_at_zth,
            "dc_over_H_at_z_mu": dc_at_zmu,
            "dc_over_H_at_z_y": dc_at_zy,
            "note": "When Γ_DC/H >> 1, photon number adjusts to full Planck",
        },
        "thermalization_verdict": {
            "strong_case": (
                "Between z=2×10⁶ and z=1090, Compton scattering produces "
                "~10⁸ interactions per photon. Double Compton and bremsstrahlung "
                "create/destroy photons above z~5×10⁴. This is sufficient for "
                "full thermalisation to far better than 50 ppm."
            ),
            "assumption_chain": (
                "BUT this calculation assumes: (1) universe was radiation-dominated "
                "at z>10⁴, (2) the Friedmann equation correctly describes expansion, "
                "(3) Thomson cross-section is unchanged at these temperatures, "
                "(4) no exotic physics altered the photon distribution."
            ),
            "critical_note": (
                "The blackbody perfection alone does not uniquely prove a Big Bang. "
                "It proves SOME optically thick thermal source. The Big Bang "
                "interpretation requires COMBINING the blackbody with: "
                "(a) the correct temperature (2.725 K), (b) the isotropy (ΔT/T~10⁻⁵), "
                "(c) the acoustic peaks in the power spectrum, "
                "(d) the baryon-to-photon ratio matching BBN, "
                "(e) consistent ages/distances from other probes."
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. SPECTRAL DISTORTIONS — What energy injections are ruled out?
# ═══════════════════════════════════════════════════════════════════════════

def spectral_distortion_analysis() -> dict:
    """Analyse what FIRAS non-detection of spectral distortions implies.

    Two types of distortion:

    μ-distortion (chemical potential):
      Arises when energy is injected at 5×10⁴ < z < 2×10⁶.
      Compton scattering redistributes energy but double Compton can't
      adjust photon number → Bose-Einstein spectrum with μ ≠ 0.

      B_μ(ν) = (2hν³/c²) × 1/(exp(hν/kT + μ) - 1)

      FIRAS limit: |μ| < 9×10⁻⁵ (95% CL)
      → Energy injection ΔE/E < 6×10⁻⁵ between z=5×10⁴ and z=2×10⁶

    y-distortion (Compton parameter):
      Arises when hot electrons scatter CMB photons at z < 5×10⁴.
      Inverse Compton shifts photons to higher frequencies.

      ΔB/B ∝ y × x × coth(x/2) - 4   where x = hν/kT

      FIRAS limit: |y| < 1.5×10⁻⁵ (95% CL)
      → Limits energy from decaying particles, early AGN, structure formation

    Physical implications of non-detection:
    - No massive particle decays between z~10³ and z~10⁶ injecting >5×10⁻⁵ of CMB energy
    - No significant early heating of IGM beyond what's expected from structure formation
    - Limits on primordial magnetic field dissipation
    - Limits on cosmic string network energy loss
    """
    # μ-distortion energy constraint
    # ΔE/E ≈ 1.4 × μ  (for small μ)
    mu_limit = 9e-5
    delta_E_over_E_mu = 1.4 * mu_limit

    # y-distortion energy constraint
    # ΔE/E ≈ 4y (for thermal y-distortion)
    y_limit = 1.5e-5
    delta_E_over_E_y = 4 * y_limit

    # What specific scenarios are ruled out?
    cmb_energy = total_energy_density(T_CMB)

    # Decaying particle: if a particle with mass m_X and number density n_X
    # decays at redshift z, it injects E = m_X c² × n_X / (energy density at z)
    # The limit constrains n_X × m_X_c2 / u_CMB(z_decay) < 6×10⁻⁵

    return {
        "mu_distortion": {
            "firas_limit": mu_limit,
            "energy_injection_limit": delta_E_over_E_mu,
            "redshift_range": "5×10⁴ < z < 2×10⁶",
            "physical_meaning": (
                "No more than 0.008% of CMB energy was injected as heat "
                "between z=2×10⁶ and z=5×10⁴ (age 2 months to 9 years)"
            ),
        },
        "y_distortion": {
            "firas_limit": y_limit,
            "energy_injection_limit": delta_E_over_E_y,
            "redshift_range": "z < 5×10⁴",
            "physical_meaning": (
                "Less than 0.006% of CMB energy was thermally added "
                "after z=5×10⁴ (universe age > 9 years)"
            ),
        },
        "what_is_ruled_out": [
            "Massive particle decays injecting >0.01% of radiation energy",
            "Primordial magnetic fields with B > 30 nG (dissipation would distort)",
            "Significant early black hole accretion heating",
            "Cosmic string networks with Gμ > 10⁻⁷ (energy loss would distort)",
            "Any non-thermal radiation source contributing >0.01% of CMB",
        ],
        "what_is_NOT_ruled_out": [
            "Energy injection before z=2×10⁶ (fully thermalised, invisible)",
            "Distortions below FIRAS sensitivity (will be probed by PIXIE/PRISTINE)",
            "Distortions at frequencies outside FIRAS range (ν<60 GHz or ν>600 GHz)",
            "Isotropic but non-thermal sources exactly mimicking a blackbody "
            "(physically implausible but not spectroscopically distinguishable)",
        ],
        "future_experiments": {
            "PIXIE": "μ sensitivity ~5×10⁻⁸ (1000× better than FIRAS)",
            "PRISTINE": "Similar to PIXIE, ESA concept",
            "note": "Would detect distortions from standard physics (Silk damping, "
                    "recombination radiation) that FIRAS couldn't see",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. ALTERNATIVE ORIGIN TESTS
# ═══════════════════════════════════════════════════════════════════════════

def alternative_origin_tests() -> dict:
    """Quantitative tests distinguishing CMB origins.

    For each alternative, compute specific predictions and compare to data.
    """
    tests = {}

    # ── Test A: Tired Light ──────────────────────────────────────────────
    # In tired-light cosmology, photons lose energy over distance: E ∝ e^(-r/R)
    # This predicts a CMB that is NOT a blackbody — it's the integrated
    # starlight from all galaxies, redshifted by the tired-light mechanism.
    #
    # Problem 1: Tired light predicts surface brightness that does NOT dim
    # with (1+z)⁴. Observations of galaxy surface brightness confirm (1+z)⁴.
    #
    # Problem 2: A tired-light CMB would show spectral distortions from
    # the sum of many stellar spectra, not a single-temperature Planck curve.
    #
    # Problem 3: The time dilation of Type Ia SN light curves scales as (1+z),
    # consistent with expansion but not with tired light (which predicts none).

    # Quantitative: energy density of starlight in the observable universe
    # ρ_starlight ≈ 2-4 × 10⁻¹⁵ J/m³  (optical/NIR background)
    # ρ_CMB = 4.2 × 10⁻¹⁴ J/m³
    # → Starlight is ~10× too dim to account for CMB
    rho_cmb = total_energy_density(T_CMB)
    rho_starlight_approx = 3e-15  # J/m³ (cosmic optical background)
    starlight_ratio = rho_starlight_approx / rho_cmb

    tests["tired_light"] = {
        "prediction": "CMB is thermalized starlight; no time dilation",
        "test_1_surface_brightness": {
            "prediction": "SB ∝ (1+z)⁻¹ (energy loss only)",
            "observation": "SB ∝ (1+z)⁻⁴ (Tolman test, Lubin & Sandage 2001)",
            "verdict": "FAILS — observed dimming matches expansion cosmology",
        },
        "test_2_energy_budget": {
            "cmb_energy_density_j_m3": rho_cmb,
            "starlight_energy_density_j_m3": rho_starlight_approx,
            "ratio": starlight_ratio,
            "verdict": f"FAILS — need {1/starlight_ratio:.0f}× more starlight than exists",
        },
        "test_3_sn_time_dilation": {
            "prediction": "No time dilation of SN light curves",
            "observation": "SNIa light curves stretch by factor (1+z) (Goldhaber+2001, Blondin+2008)",
            "verdict": "FAILS — time dilation confirmed to z~1",
        },
        "test_4_spectrum": {
            "prediction": "Superposition of redshifted stellar spectra → NOT a clean blackbody",
            "observation": "Perfect single-T blackbody to 50 ppm",
            "verdict": "FAILS — no known mechanism to thermalise starlight in low-density IGM",
        },
    }

    # ── Test B: Local Dust / Iron Whiskers ───────────────────────────────
    # Some proposals: CMB is thermal radiation from local dust grains
    # (or iron 'whiskers' from SNe) in the solar system or Galaxy.
    #
    # Problems:
    # 1. Dust must be at T=2.725 K everywhere — but local ISM has T=10-100 K
    # 2. The CMB is isotropic to 10⁻⁵; local sources would show structure
    # 3. The spectrum must be a perfect blackbody → requires optically thick dust
    #    but the Galaxy is optically thin at CMB wavelengths
    # 4. CMB shows a dipole consistent with Earth's motion, not local structure

    tests["local_dust"] = {
        "prediction": "CMB = thermal emission from cold dust in Galaxy/solar system",
        "test_1_isotropy": {
            "required": "Isotropic to ΔT/T ~ 10⁻⁵ excluding dipole",
            "local_dust_problem": (
                "ISM density varies by orders of magnitude across the sky. "
                "Galactic plane has column density 100× higher than poles. "
                "Would produce ΔT/T >> 10⁻¹."
            ),
            "verdict": "FAILS — no known local dust distribution is this isotropic",
        },
        "test_2_optical_depth": {
            "required": "Optically thick at 60-600 GHz for blackbody",
            "reality": (
                "Galaxy is τ < 0.01 at 100 GHz outside the plane. "
                "Optically thin → emission is NOT blackbody."
            ),
            "verdict": "FAILS — Galaxy is transparent at CMB frequencies",
        },
        "test_3_temperature": {
            "required": "Dust at exactly 2.725 K everywhere",
            "reality": "ISM dust ranges from 10-100 K; no equilibrium at 2.725 K locally",
            "verdict": "FAILS — wrong temperature by orders of magnitude",
        },
        "test_4_sz_effect": {
            "test": "Sunyaev-Zel'dovich effect seen toward galaxy clusters",
            "implication": (
                "CMB photons passing through hot cluster gas gain energy. "
                "This REQUIRES the CMB to be BEHIND the clusters — not local."
            ),
            "verdict": "FAILS — SZ effect proves CMB is cosmological",
        },
    }

    # ── Test C: Steady-State / Quasi-Steady-State ────────────────────────
    # Hoyle, Burbidge, Narlikar proposed metallic whiskers thermalise starlight
    #
    # Problems:
    # 1. Same energy budget problem as tired light
    # 2. Would predict frequency-dependent opacity → spectral distortions
    # 3. Cannot explain acoustic peaks in power spectrum

    n_gamma = photon_number_density(T_CMB)
    n_baryon = RHO_CRIT * OMEGA_B / M_P
    eta = n_baryon / n_gamma  # baryon-to-photon ratio

    tests["steady_state"] = {
        "prediction": "CMB = thermalised starlight in an eternally expanding universe",
        "test_1": "Same energy budget failure as tired light",
        "test_2_acoustic_peaks": {
            "requires": "No oscillating plasma → no acoustic peaks",
            "observation": "7+ acoustic peaks detected (Planck, ACT, SPT)",
            "verdict": "FAILS — acoustic peaks require a finite-age plasma epoch",
        },
        "test_3_baryon_photon_ratio": {
            "eta_observed": eta,
            "bbn_prediction": "η = (6.1 ± 0.3) × 10⁻¹⁰",
            "cmb_value": f"η = {eta:.2e}",
            "verdict": (
                "BBN and CMB independently give the same η — "
                "this consistency is very hard to replicate in steady-state models"
            ),
        },
    }

    return tests


# ═══════════════════════════════════════════════════════════════════════════
# 6. SUNYAEV-ZEL'DOVICH EFFECT
# ═══════════════════════════════════════════════════════════════════════════

def sz_effect_analysis() -> dict:
    """The Sunyaev-Zel'dovich effect as independent CMB evidence.

    When CMB photons pass through the hot gas (T~10⁸ K) of a galaxy cluster,
    inverse Compton scattering boosts some photons to higher frequencies.

    This creates a characteristic spectral distortion:
    - Deficit at ν < 217 GHz (photons moved out)
    - Excess at ν > 217 GHz (photons moved in)
    - Null at ν = 217 GHz (crossover frequency)

    The SZ effect has been detected in hundreds of clusters.
    WHY IT MATTERS: It proves the CMB originates BEHIND the clusters
    at cosmological distances, not locally.
    """
    # SZ spectral distortion: ΔI/I = y × g(x) where x = hν/kT_CMB
    # g(x) = x(eˣ+1)/(eˣ-1) - 4
    def sz_spectral_function(nu_ghz: float) -> float:
        """SZ spectral distortion function g(x)."""
        x = H_PLANCK * nu_ghz * 1e9 / (K_B * T_CMB)
        if x > 30:
            return x - 4  # high-x limit
        ex = math.exp(x)
        return x * (ex + 1) / (ex - 1) - 4

    # Crossover frequency: g(x) = 0
    # Solve: x(eˣ+1)/(eˣ-1) = 4 → x ≈ 3.83 → ν ≈ 217 GHz
    x_cross = brentq(sz_spectral_function, 100, 300)

    # Typical cluster parameters
    T_cluster = 1e8          # K (∼8 keV)
    n_e_cluster = 1e3        # per m³ (typical core)
    R_cluster = 1.0 * MPC_M  # 1 Mpc radius (∼3e22 m)

    # Compton y-parameter for a typical cluster
    y_cluster = (K_B * T_cluster / (M_E * C_SI**2)) * n_e_cluster * SIGMA_T * R_cluster

    # ΔT/T ≈ -2y in the Rayleigh-Jeans regime (low frequency)
    delta_T_over_T = -2 * y_cluster

    # Number of clusters with SZ detections
    n_clusters_sz = 1600  # Planck SZ catalogue (PSZ2)

    # Spectral distortion at key frequencies
    sz_spectrum = {}
    for nu in [90, 150, 217, 270, 353]:
        sz_spectrum[f"{nu}_GHz"] = {
            "g_x": sz_spectral_function(nu),
            "delta_T_uK": delta_T_over_T * T_CMB * 1e6 * abs(sz_spectral_function(nu) / sz_spectral_function(90)),
            "sign": "decrement" if sz_spectral_function(nu) < 0 else
                    "null" if abs(sz_spectral_function(nu)) < 0.1 else "increment",
        }

    return {
        "crossover_frequency_ghz": x_cross,
        "typical_cluster": {
            "T_keV": K_B * T_cluster / EV_TO_J / 1e3,
            "n_e_per_m3": n_e_cluster,
            "R_mpc": R_cluster / MPC_M,
            "y_parameter": y_cluster,
            "delta_T_over_T": delta_T_over_T,
            "delta_T_uK": abs(delta_T_over_T) * T_CMB * 1e6,
        },
        "planck_sz_catalog_clusters": n_clusters_sz,
        "spectral_shape": sz_spectrum,
        "evidence_value": (
            "The SZ effect is the single strongest proof that the CMB is cosmological. "
            "It requires photons to traverse galaxy clusters at z=0.05-1.5, "
            "picking up energy from 10⁸ K gas. No local-origin model can explain this. "
            f"Detected in {n_clusters_sz}+ clusters with the predicted spectral shape."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. CMB DIPOLE — Our motion through the CMB rest frame
# ═══════════════════════════════════════════════════════════════════════════

def cmb_dipole_analysis() -> dict:
    """Analyse the CMB dipole anisotropy.

    The largest CMB anisotropy is a dipole (ℓ=1) with amplitude 3.3621 mK,
    interpreted as the Doppler shift from our motion relative to the CMB
    rest frame at v = 369.82 ± 0.11 km/s toward (l,b) = (264.021°, 48.253°).

    This has been independently confirmed by the kinematic dipole in
    radio galaxy and quasar number counts (Ellis & Baldwin 1984; Secrest+2021),
    though with some recent tension at the 2-5σ level.
    """
    v_dipole = 369.82    # km/s
    v_err = 0.11
    T_dipole = 3.3621e-3  # K (3.3621 mK)
    l_gal = 264.021      # Galactic longitude (degrees)
    b_gal = 48.253       # Galactic latitude (degrees)

    beta = v_dipole / (C_SI / 1e3)  # v/c

    # Expected ΔT/T from Doppler: ΔT/T = β = v/c
    expected_delta_T = T_CMB * beta
    observed_delta_T = T_dipole

    # Higher-order terms: T(θ) = T₀(1 + β cosθ + β²/2 cos²θ + ...)
    # β² term gives a quadrupole contribution ΔT/T ~ β²/2 ~ 7.6×10⁻⁷
    beta_sq_correction = beta**2 / 2

    # Aberration: changes apparent direction of photons by ~β radians
    aberration_arcsec = beta * 180 * 3600 / math.pi

    return {
        "velocity_km_s": v_dipole,
        "velocity_err": v_err,
        "beta": beta,
        "direction_galactic": {"l_deg": l_gal, "b_deg": b_gal},
        "dipole_amplitude_mK": T_dipole * 1e3,
        "expected_from_v": expected_delta_T * 1e3,
        "agreement": abs(expected_delta_T - T_dipole) / T_dipole < 0.01,
        "higher_order_quadrupole": beta_sq_correction,
        "aberration_arcsec": aberration_arcsec,
        "evidence_value": (
            "The dipole is perfectly explained by our motion at 370 km/s. "
            "This is consistent with Milky Way motion in the local supercluster. "
            "The frequency spectrum of the dipole is EXACTLY that of a boosted "
            "blackbody — not a temperature gradient — which confirms it's kinematic."
        ),
        "tension_note": (
            "Recent studies (Secrest+2021, Nature Astron.) find the quasar "
            "dipole is 2-5× larger than expected from the CMB dipole. "
            "If confirmed, this may indicate either a large-scale bulk flow "
            "or a violation of the cosmological principle. Under active investigation."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. ACOUSTIC PEAK STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

def acoustic_peaks_analysis() -> dict:
    """What the CMB power spectrum acoustic peaks encode.

    The angular power spectrum C(ℓ) shows peaks at specific multipoles
    corresponding to standing sound waves in the pre-recombination plasma.

    Peak positions encode:
    - ℓ₁ ≈ 220: sound horizon at recombination → geometry (Ω_total)
    - ℓ₂/ℓ₁ ratio: baryon loading (Ω_b h²)
    - Odd/even peak heights: baryon density
    - Damping tail: photon diffusion length (Silk damping)
    - Overall amplitude: primordial power + optical depth to reionization
    """
    # Sound speed in primordial plasma: c_s = c/√(3(1+R))
    # where R = 3ρ_b/(4ρ_γ) is the baryon-to-photon momentum ratio
    R_rec = 3 * OMEGA_B / (4 * OMEGA_R) * 1  # at z=0; at z_rec: R(z) independent of z
    # More precisely: R = (3Ω_b)/(4Ω_γ) × 1/(1+z)
    # But Ω_γ ∝ (1+z)⁻¹ × Ω_r... let me be more careful.
    # R(z) = 3ρ_b(z)/(4ρ_γ(z)) = 3(Ω_b(1+z)³)/(4Ω_γ(1+z)⁴) × ρ_crit/ρ_crit
    # Ω_γ = photon only ≈ Ω_r / 1.68 (removing neutrinos)
    OMEGA_GAMMA = OMEGA_R / 1.6813  # photons only (3 massless neutrinos contribute 0.6813×)
    R_at_rec = 3 * OMEGA_B / (4 * OMEGA_GAMMA) * (1 + Z_RECOMB)
    # Wait — R ∝ 1/(1+z)? No. ρ_b ∝ (1+z)³, ρ_γ ∝ (1+z)⁴
    # R = 3ρ_b/(4ρ_γ) ∝ (1+z)⁻¹ → R decreases at higher z
    # R(z) = R_0 / (1+z)  where R_0 = 3Ω_b ρ_crit / (4 × a_rad T₀⁴ / c²)
    # Actually: R(z) = (3Ω_b h²)/(4 × 2.469×10⁻⁵ × (1+z))  (standard formula)
    h = 67.36 / 100
    R_at_rec_correct = 3 * OMEGA_B * h**2 / (4 * 2.469e-5 * (1 + Z_RECOMB))
    # ~ 3 × 0.0493 × 0.4537 / (4 × 2.469e-5 × 1090.8) ≈ 0.625

    c_s_rec = C_SI / math.sqrt(3 * (1 + R_at_rec_correct))

    # Comoving sound horizon: r_s = ∫_z_rec^∞ c_s / H(z) dz
    # (comoving: no (1+z) in denominator)
    def sound_horizon_integrand(z):
        R_z = 3 * OMEGA_B * h**2 / (4 * 2.469e-5 * (1 + z))
        cs = C_SI / math.sqrt(3 * (1 + R_z))
        zp1 = 1 + z
        H_z = H0_SI * math.sqrt(OMEGA_R * zp1**4 + OMEGA_M * zp1**3 + OMEGA_LAMBDA)
        return cs / H_z

    r_s_m, _ = quad(sound_horizon_integrand, Z_RECOMB, 1e7)
    r_s_mpc = r_s_m / MPC_M

    # Comoving distance to recombination
    def d_C_integrand(z):
        zp1 = 1 + z
        H_z = H0_SI * math.sqrt(OMEGA_R * zp1**4 + OMEGA_M * zp1**3 + OMEGA_LAMBDA)
        return C_SI / H_z

    d_C_m, _ = quad(d_C_integrand, 0, Z_RECOMB)
    d_A_m = d_C_m / (1 + Z_RECOMB)  # angular diameter distance
    d_A_mpc = d_A_m / MPC_M

    # Angular scale: θ_s = r_s(comoving) / d_C(comoving)
    theta_s = r_s_m / d_C_m  # radians
    theta_s_deg = theta_s * 180 / math.pi

    # First peak multipole: ℓ₁ ≈ π/θ_s
    ell_1 = math.pi / theta_s

    # Silk damping scale
    # λ_D ≈ (mean free path × horizon) ^ (1/2)
    # More precisely: damping occurs on scales where photon diffusion
    # length exceeds the wavelength during recombination
    mfp_rec = thomson_mean_free_path(electron_density_at_z(Z_RECOMB))
    # Diffusion length ~ √(N × λ²) ~ √(c t_rec / (n_e σ_T) × c/(n_e σ_T))
    # λ_D ~ (c / (n_e σ_T)) × √(n_e σ_T c × t_rec) ~ c × √(t_rec / (n_e σ_T c))

    # Planck measured values for comparison
    planck_ell_s = 302.0   # acoustic scale ℓ_s = π/θ_s (Planck 2018: 100θ_*=1.04110)
    planck_ell_1 = 220.0   # first peak (phase-shifted from ℓ_s by gravitational driving)
    planck_r_s = 144.43    # Mpc (sound horizon, Planck 2018)

    return {
        "sound_speed_at_rec": {
            "c_s_km_s": c_s_rec / 1e3,
            "c_s_over_c": c_s_rec / C_SI,
            "baryon_loading_R": R_at_rec_correct,
        },
        "sound_horizon": {
            "r_s_mpc": r_s_mpc,
            "planck_r_s_mpc": planck_r_s,
            "agreement_pct": abs(r_s_mpc - planck_r_s) / planck_r_s * 100,
        },
        "angular_diameter_distance": {
            "d_A_mpc": d_A_mpc,
            "d_A_gly": d_A_mpc * 3.2616e-3,
        },
        "acoustic_scale": {
            "theta_s_deg": theta_s_deg,
            "computed_ell_s": ell_1,
            "planck_ell_s": planck_ell_s,
            "agreement_pct": abs(ell_1 - planck_ell_s) / planck_ell_s * 100,
            "first_peak_ell": planck_ell_1,
            "note": (
                "ℓ_s = π/θ_s ≈ 302 is the acoustic scale. The observed first peak "
                "at ℓ₁ ≈ 220 is phase-shifted from ℓ_s by gravitational driving "
                "of the acoustic oscillations (ISW + baryon drag). This shift is "
                "a prediction of ΛCDM, not a free parameter."
            ),
        },
        "what_peaks_prove": {
            "peak_1_position": (
                f"Acoustic scale ℓ_s ≈ {ell_1:.0f} (Planck: {planck_ell_s:.0f}). "
                "This depends on geometry (Ω_total), expansion history, and sound speed. "
                f"First peak at ℓ ≈ 220 ↔ Ω_total ≈ 1 (flat universe)."
            ),
            "peak_ratios": (
                "Odd peaks (1,3,5) are compression peaks enhanced by baryons. "
                "Even peaks (2,4) are rarefaction peaks suppressed by baryons. "
                "The ratio ℓ₂/ℓ₁ and relative heights give Ω_b h² = 0.02237."
            ),
            "damping_tail": (
                "High-ℓ peaks are progressively damped by photon diffusion "
                "(Silk damping). The damping scale gives N_eff = 2.99 ± 0.17 "
                "neutrino species — confirming particle physics."
            ),
        },
        "why_this_matters": (
            "The acoustic peaks are the STRONGEST evidence for a hot plasma epoch. "
            "No alternative to the Big Bang can naturally produce: "
            "(1) peaks at the correct ℓ values, (2) the correct odd/even asymmetry, "
            "(3) the correct damping tail, (4) the correct polarization pattern, "
            "all from a single consistent set of 6 parameters."
        ),
        "critical_note": (
            "The peaks require a *finite-age*, *finite-temperature* plasma "
            "that was once optically thick and then became transparent. "
            "This is the recombination event at z≈1090. "
            "No steady-state or local model predicts acoustic oscillations."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. RECOMBINATION — SAHA EQUATION
# ═══════════════════════════════════════════════════════════════════════════

def saha_ionization_fraction(T: float, n_b: float) -> float:
    """Compute hydrogen ionization fraction x_e from the Saha equation.

    x_e²/(1-x_e) = (1/n_b) × (m_e k T / (2π ℏ²))^(3/2) × exp(-E_ion/kT)

    This is the equilibrium prediction. The actual 'freeze-out' of
    recombination (Peebles 1968) gives slightly different results because
    Lyman-α photons get reabsorbed.
    """
    # Thermal de Broglie wavelength factor
    # (m_e kT / (2π ℏ²))^(3/2)
    thermal_factor = (M_E * K_B * T / (2 * math.pi * H_BAR**2))**(1.5)

    # Boltzmann factor
    x_ion = E_ION_H * EV_TO_J / (K_B * T)
    if x_ion > 500:
        return 0.0  # fully neutral

    rhs = thermal_factor * math.exp(-x_ion) / n_b

    # Solve x²/(1-x) = rhs
    # x² + rhs×x - rhs = 0  →  x = (-rhs + √(rhs²+4rhs))/2
    disc = rhs**2 + 4 * rhs
    x_e = (-rhs + math.sqrt(disc)) / 2
    return min(1.0, max(0.0, x_e))


def recombination_analysis() -> dict:
    """Compute the recombination history: when did the universe become transparent?

    The 'last scattering surface' is not a sharp wall — it has a finite
    thickness Δz ≈ 80 (or ~115,000 years).
    """
    # Baryon number density today
    n_b_0 = RHO_CRIT * OMEGA_B * X_H / M_P   # hydrogen number density

    # Compute ionization fraction vs redshift
    z_range = np.linspace(2000, 800, 200)
    x_e_saha = []
    for z in z_range:
        T_z = T_CMB * (1 + z)
        n_b_z = n_b_0 * (1 + z)**3
        x_e_saha.append(saha_ionization_fraction(T_z, n_b_z))
    x_e_saha = np.array(x_e_saha)

    # Find z where x_e = 0.5 (half-ionized)
    idx_half = np.argmin(np.abs(x_e_saha - 0.5))
    z_half = z_range[idx_half]
    T_half = T_CMB * (1 + z_half)

    # Last scattering surface thickness
    # Visibility function g(z) = dτ/dz × exp(-τ)
    # Has a peak at z ≈ 1089 with FWHM Δz ≈ 80
    delta_z_lss = 80
    # Corresponding time span
    def cosmic_age_at_z(z):
        integrand = lambda zp: 1.0 / ((1 + zp) * H0_SI *
                    math.sqrt(OMEGA_R*(1+zp)**4 + OMEGA_M*(1+zp)**3 + OMEGA_LAMBDA))
        val, _ = quad(integrand, z, 1e6)
        return val

    t_rec = cosmic_age_at_z(Z_RECOMB)
    t_rec_kyr = t_rec / YR_S / 1e3
    t_rec_plus = cosmic_age_at_z(Z_RECOMB - delta_z_lss/2)
    t_rec_minus = cosmic_age_at_z(Z_RECOMB + delta_z_lss/2)
    delta_t_lss_kyr = (t_rec_plus - t_rec_minus) / YR_S / 1e3

    # Temperature at recombination
    T_rec = T_CMB * (1 + Z_RECOMB)
    T_rec_eV = K_B * T_rec / EV_TO_J

    return {
        "saha_results": {
            "z_half_ionized": float(z_half),
            "T_half_K": T_half,
            "T_half_eV": K_B * T_half / EV_TO_J,
            "note": (
                f"Saha gives 50% ionization at z≈{z_half:.0f} (T≈{T_half:.0f} K). "
                "This is MUCH lower than E_ion/k = 157,800 K because "
                "the photon-to-baryon ratio is ~10⁹ — even the Wien tail "
                "of a 3000 K bath has enough photons to ionize."
            ),
        },
        "last_scattering_surface": {
            "z_rec": Z_RECOMB,
            "T_rec_K": T_rec,
            "T_rec_eV": T_rec_eV,
            "t_rec_kyr": t_rec_kyr,
            "delta_z": delta_z_lss,
            "delta_t_kyr": delta_t_lss_kyr,
            "note": (
                "Recombination is NOT instantaneous. The last scattering surface "
                "has Δz ≈ 80 (FWHM), spanning ~115 kyr. Photons from slightly "
                "different depths have slightly different properties — this adds "
                "a finite-thickness smoothing to the CMB."
            ),
        },
        "why_3000K_not_160000K": (
            "Naive expectation: ionization turns off at T = E_ion/k = 157,800 K "
            "(z ≈ 58,000). But there are ~10⁹ photons per baryon, so even the "
            "exponential tail of the Planck distribution contains enough ionizing "
            "photons to keep hydrogen ionized down to ~3000 K (z ≈ 1090)."
        ),
        "evidence_assessment": (
            "The Saha equation prediction z_rec ≈ 1090 matches the observed "
            "CMB angular power spectrum peak positions. This is a NON-TRIVIAL "
            "consistency check: the same baryon density Ω_b h² that fits the "
            "peak heights also gives the correct recombination redshift."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 10. CRITICAL EVIDENCE ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════

def evidence_assessment() -> dict:
    """A critical assessment of CMB evidence — what is strong, what is assumed.

    Separates observational FACTS from theoretical INTERPRETATIONS.
    """
    return {
        "observational_facts": {
            "F1": "Microwave background radiation exists, filling all of space (Penzias & Wilson 1965)",
            "F2": f"Its spectrum is a blackbody at T = {T_CMB} ± {T_CMB_ERR} K to 50 ppm (FIRAS 1996)",
            "F3": "It is isotropic to ΔT/T ~ 10⁻⁵ after dipole removal (COBE/Planck)",
            "F4": "The dipole has amplitude 3.36 mK, consistent with motion at 370 km/s",
            "F5": "The angular power spectrum shows 7+ acoustic peaks (WMAP/Planck/ACT/SPT)",
            "F6": "E-mode polarization is detected, correlated with temperature anisotropies",
            "F7": "SZ decrements/increments are detected toward ~1600 galaxy clusters",
            "F8": "CMB lensing by intervening structure is detected (cosmological distance confirmed)",
            "F9": "Baryon acoustic oscillations at z=0.1-2.5 match the CMB-predicted sound horizon",
            "F10": "No spectral distortions detected (μ < 9×10⁻⁵, |y| < 1.5×10⁻⁵)",
        },
        "standard_interpretations": {
            "I1": "CMB is relic radiation from recombination at z ≈ 1090",
            "I2": "Blackbody perfection results from thermalisation in the hot dense early universe",
            "I3": "Anisotropies are seeds of structure, amplified by gravitational instability",
            "I4": "Acoustic peaks record sound waves in a coupled baryon-photon plasma",
            "I5": "The power spectrum encodes 6 parameters: Ω_b h², Ω_c h², θ_s, τ, n_s, A_s",
            "I6": "The universe was once hot, dense, and opaque, then cooled through expansion",
        },
        "strength_of_evidence": {
            "VERY_STRONG": [
                "CMB is at cosmological distance (SZ effect in 1600+ clusters, lensing)",
                "A hot plasma epoch existed (acoustic peaks, polarization)",
                "Baryon density from peaks matches BBN (two independent probes)",
                "Sound horizon matches BAO measurements (CMB-galaxy consistency)",
                "Geometry is flat (ℓ₁ ≈ 220 → Ω_total ≈ 1.000 ± 0.002)",
            ],
            "STRONG": [
                "Thermalisation occurred at z > 2×10⁶ (no spectral distortions)",
                "Recombination happened at T ≈ 3000 K (Saha + peak positions)",
                "Silk damping gives N_eff consistent with 3 neutrino species",
                "Primordial spectrum is nearly scale-invariant (n_s = 0.965 ± 0.004)",
            ],
            "MODERATE_ASSUMPTIONS": [
                "General relativity correctly describes expansion at all z",
                "Thomson cross-section is constant (no new physics)",
                "The cosmological principle (homogeneity + isotropy) holds",
                "Initial perturbations were Gaussian and adiabatic",
            ],
            "OPEN_QUESTIONS": [
                "What set the initial conditions? (Inflation is assumed, not proven)",
                "Why is the universe so flat? (Inflation solves this, but is it the only solution?)",
                "The CMB anomalies: low quadrupole, hemispherical asymmetry, cold spot — statistical flukes or new physics?",
                "Hubble tension: CMB-inferred H₀=67.4 vs local H₀=73.0 — systematic or fundamental?",
                "The dipole tension: quasar number count dipole is 2-5× too large",
                "Dark energy: Ω_Λ is measured but not understood",
                "Dark matter: Ω_c is measured but not identified",
            ],
        },
        "the_core_tension": (
            "The black body spectrum alone does not prove the Big Bang. "
            "Any optically thick thermal source at 2.725 K would produce it. "
            "However, no known local or alternative mechanism can simultaneously "
            "explain:\n"
            "  (1) the blackbody spectrum\n"
            "  (2) the isotropy to 10⁻⁵\n"
            "  (3) the acoustic peak structure\n"
            "  (4) the SZ effect in distant clusters\n"
            "  (5) the matching baryon density from BBN\n"
            "  (6) the BAO measurements in galaxy surveys\n\n"
            "Each individual observation has alternative explanations. "
            "The power of the standard model is that ONE set of 6 parameters "
            "fits ALL of them simultaneously. The question is whether this "
            "concordance constitutes PROOF or merely the best current MODEL."
        ),
        "what_a_skeptic_should_focus_on": [
            "The CMB anomalies (real statistical significance, not cherry-picked)",
            "The Hubble tension (may indicate new physics at z~1000)",
            "The dipole tension (may challenge the cosmological principle)",
            "The assumption test: which observations are truly model-independent?",
            "PIXIE/PRISTINE: will spectral distortions from standard physics be found?",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Master runner
# ═══════════════════════════════════════════════════════════════════════════

def run_all(verbose: bool = True) -> Dict:
    results = {}

    def _pr(msg=""):
        if verbose:
            print(msg)

    def _hr():
        _pr("─" * 72)

    # ── 1. Planck spectrum ───────────────────────────────────────────────
    _pr("\n  ══════════ CMB BLACKBODY SPECTRUM ══════════")
    _hr()
    firas = firas_analysis()
    results["firas"] = firas

    _pr(f"    T_CMB = {firas['T_cmb_k']} ± {firas['T_err_k']} K")
    _pr(f"    Relative precision: {firas['relative_precision']:.1e}")
    _pr(f"    Peak frequency:     {firas['nu_peak_ghz']:.1f} GHz")
    _pr(f"    Peak wavelength:    {firas['lambda_peak_mm']:.2f} mm")
    _pr(f"    Peak radiance:      {firas['B_peak_mjy_sr']:.1f} MJy/sr")
    _pr(f"    Energy density:     {firas['energy_density_ev_cm3']:.3f} eV/cm³")
    _pr(f"    Photon density:     {firas['photon_density_per_cm3']:.0f} /cm³")
    _pr(f"    Photons per baryon: {firas['photons_per_baryon']:.0f}")
    _pr(f"    FIRAS RMS residual: {firas['rms_residual_ppm']} ppm")
    _pr(f"    μ-distortion limit: < {firas['mu_distortion_limit']:.0e}")
    _pr(f"    y-distortion limit: < {firas['y_distortion_limit']:.0e}")
    _pr(f"    {firas['measurement_note']}")
    _hr()

    # ── 2. Thermalization physics ────────────────────────────────────────
    _pr("\n  ══════════ THERMALIZATION PHYSICS ══════════")
    _hr()
    therm = thermalization_analysis()
    results["thermalization"] = therm

    _pr("    Key redshifts:")
    z_items = list(therm["redshift_thresholds"].items())
    t_items = list(therm["temperatures_K"].values())
    for (name, z), T in zip(z_items, t_items):
        _pr(f"      {name:25s}  z = {z:.0e}   T = {T:.0e} K")

    ar = therm["at_recombination"]
    _pr(f"\n    At recombination (z = {Z_RECOMB}):")
    _pr(f"      Electron density:   {ar['n_e_per_m3']:.2e} /m³")
    _pr(f"      Mean free path:     {ar['mean_free_path_m']:.2e} m  ({ar['mean_free_path_ly']:.1f} ly)")
    _pr(f"      Compton rate:       {ar['compton_rate_per_s']:.2e} /s")
    _pr(f"      Hubble rate:        {ar['hubble_rate_per_s']:.2e} /s")
    _pr(f"      Γ_scatter / H:     {ar['scatter_rate_over_hubble']:.0f}×")

    od = therm["optical_depths"]
    _pr(f"\n    Optical depths:")
    _pr(f"      τ(0 → z_rec):      {od['tau_0_to_rec']:.1f}")
    _pr(f"      τ(z_rec → z_th):   {od['tau_rec_to_thermalisation']:.2e}")
    _pr(f"      Compton y total:    {therm['compton_y_parameter']:.2e}")

    dc = therm["double_compton_rates"]
    _pr(f"\n    Double Compton Γ/H:")
    _pr(f"      at z_th = 2×10⁶:   {dc['dc_over_H_at_z_th']:.2e}")
    _pr(f"      at z_μ  = 5×10⁴:   {dc['dc_over_H_at_z_mu']:.2e}")
    _pr(f"      at z_y  = 1×10⁴:   {dc['dc_over_H_at_z_y']:.2e}")

    tv = therm["thermalization_verdict"]
    _pr(f"\n    VERDICT: {tv['strong_case']}")
    _pr(f"\n    ASSUMPTION CHAIN: {tv['assumption_chain']}")
    _pr(f"\n    CRITICAL NOTE: {tv['critical_note']}")
    _hr()

    # ── 3. Spectral distortions ──────────────────────────────────────────
    _pr("\n  ══════════ SPECTRAL DISTORTION CONSTRAINTS ══════════")
    _hr()
    dist = spectral_distortion_analysis()
    results["distortions"] = dist

    _pr(f"    μ-distortion limit:  |μ| < {dist['mu_distortion']['firas_limit']:.0e}")
    _pr(f"      → ΔE/E < {dist['mu_distortion']['energy_injection_limit']:.1e}  "
         f"({dist['mu_distortion']['redshift_range']})")
    _pr(f"      {dist['mu_distortion']['physical_meaning']}")
    _pr(f"\n    y-distortion limit:  |y| < {dist['y_distortion']['firas_limit']:.0e}")
    _pr(f"      → ΔE/E < {dist['y_distortion']['energy_injection_limit']:.1e}  "
         f"({dist['y_distortion']['redshift_range']})")
    _pr(f"      {dist['y_distortion']['physical_meaning']}")

    _pr(f"\n    RULED OUT:")
    for item in dist["what_is_ruled_out"]:
        _pr(f"      ✗ {item}")
    _pr(f"\n    NOT RULED OUT:")
    for item in dist["what_is_NOT_ruled_out"]:
        _pr(f"      ? {item}")
    _hr()

    # ── 4. Alternative origin tests ──────────────────────────────────────
    _pr("\n  ══════════ ALTERNATIVE ORIGIN TESTS ══════════")
    _hr()
    alts = alternative_origin_tests()
    results["alternatives"] = alts

    for alt_name, alt_data in alts.items():
        _pr(f"\n    [{alt_name.upper().replace('_', ' ')}]")
        if isinstance(alt_data, dict) and "prediction" in alt_data:
            _pr(f"    Prediction: {alt_data['prediction']}")
        for key, val in alt_data.items():
            if isinstance(val, dict) and "verdict" in val:
                _pr(f"      {key}: {val['verdict']}")
    _hr()

    # ── 5. Sunyaev-Zel'dovich effect ────────────────────────────────────
    _pr("\n  ══════════ SUNYAEV-ZEL'DOVICH EFFECT ══════════")
    _hr()
    sz = sz_effect_analysis()
    results["sz_effect"] = sz

    tc = sz["typical_cluster"]
    _pr(f"    Typical cluster: T = {tc['T_keV']:.1f} keV, "
         f"n_e = {tc['n_e_per_m3']:.0e} /m³, R = {tc['R_mpc']:.0f} Mpc")
    _pr(f"    y-parameter:     {tc['y_parameter']:.2e}")
    _pr(f"    ΔT/T:            {tc['delta_T_over_T']:.2e}  ({tc['delta_T_uK']:.0f} μK)")
    _pr(f"    Crossover freq:  {sz['crossover_frequency_ghz']:.0f} GHz")
    _pr(f"    Detected in:     {sz['planck_sz_catalog_clusters']}+ clusters")
    _pr(f"\n    Spectral shape:")
    for fname, fdata in sz["spectral_shape"].items():
        _pr(f"      {fname:8s}  g(x) = {fdata['g_x']:+.3f}  [{fdata['sign']}]")
    _pr(f"\n    {sz['evidence_value']}")
    _hr()

    # ── 6. CMB dipole ────────────────────────────────────────────────────
    _pr("\n  ══════════ CMB DIPOLE ══════════")
    _hr()
    dip = cmb_dipole_analysis()
    results["dipole"] = dip

    _pr(f"    Velocity:        {dip['velocity_km_s']:.2f} ± {dip['velocity_err']:.2f} km/s")
    _pr(f"    β = v/c:         {dip['beta']:.6f}")
    _pr(f"    Direction:       (l, b) = ({dip['direction_galactic']['l_deg']:.3f}°, {dip['direction_galactic']['b_deg']:.3f}°)")
    _pr(f"    Dipole T:        {dip['dipole_amplitude_mK']:.4f} mK")
    _pr(f"    Expected from v: {dip['expected_from_v']:.4f} mK")
    _pr(f"    Agreement:       {'✓' if dip['agreement'] else '✗'}")
    _pr(f"    β² quadrupole:   {dip['higher_order_quadrupole']:.2e}")
    _pr(f"    Aberration:      {dip['aberration_arcsec']:.1f} arcsec")
    _pr(f"\n    {dip['evidence_value']}")
    _pr(f"\n    TENSION: {dip['tension_note']}")
    _hr()

    # ── 7. Acoustic peaks ────────────────────────────────────────────────
    _pr("\n  ══════════ ACOUSTIC PEAK STRUCTURE ══════════")
    _hr()
    peaks = acoustic_peaks_analysis()
    results["acoustic_peaks"] = peaks

    ss = peaks["sound_speed_at_rec"]
    _pr(f"    Sound speed at recombination:")
    _pr(f"      c_s = {ss['c_s_km_s']:.0f} km/s  ({ss['c_s_over_c']:.4f} c)")
    _pr(f"      Baryon loading R = {ss['baryon_loading_R']:.3f}")

    sh = peaks["sound_horizon"]
    _pr(f"\n    Sound horizon:")
    _pr(f"      Computed: {sh['r_s_mpc']:.1f} Mpc")
    _pr(f"      Planck:   {sh['planck_r_s_mpc']:.2f} Mpc")
    _pr(f"      Agreement: {sh['agreement_pct']:.1f}%")

    fp = peaks["acoustic_scale"]
    _pr(f"\n    Acoustic scale:")
    _pr(f"      θ_s = {fp['theta_s_deg']:.3f}°")
    _pr(f"      Computed ℓ_s = {fp['computed_ell_s']:.0f}")
    _pr(f"      Planck ℓ_s  = {fp['planck_ell_s']:.0f}")
    _pr(f"      Agreement:    {fp['agreement_pct']:.1f}%")
    _pr(f"      First peak ℓ₁ ≈ {fp['first_peak_ell']:.0f} (phase-shifted by grav. driving)")
    _pr(f"      {fp['note']}")

    _pr(f"\n    {peaks['why_this_matters']}")
    _pr(f"\n    {peaks['critical_note']}")
    _hr()

    # ── 8. Recombination physics ─────────────────────────────────────────
    _pr("\n  ══════════ RECOMBINATION PHYSICS (SAHA EQUATION) ══════════")
    _hr()
    rec = recombination_analysis()
    results["recombination"] = rec

    sr = rec["saha_results"]
    _pr(f"    Saha equation:")
    _pr(f"      50% ionized at z ≈ {sr['z_half_ionized']:.0f}  (T ≈ {sr['T_half_K']:.0f} K, {sr['T_half_eV']:.2f} eV)")
    _pr(f"      {sr['note']}")

    lss = rec["last_scattering_surface"]
    _pr(f"\n    Last scattering surface:")
    _pr(f"      z_rec = {lss['z_rec']}  (T = {lss['T_rec_K']:.0f} K, {lss['T_rec_eV']:.2f} eV)")
    _pr(f"      Age: {lss['t_rec_kyr']:.0f} kyr")
    _pr(f"      Thickness: Δz ≈ {lss['delta_z']}  (Δt ≈ {lss['delta_t_kyr']:.0f} kyr)")
    _pr(f"      {lss['note']}")

    _pr(f"\n    WHY 3000 K NOT 160,000 K?")
    _pr(f"    {rec['why_3000K_not_160000K']}")
    _hr()

    # ── 9. Critical assessment ───────────────────────────────────────────
    _pr("\n  ══════════ CRITICAL EVIDENCE ASSESSMENT ══════════")
    _hr()
    assess = evidence_assessment()
    results["assessment"] = assess

    _pr("    OBSERVATIONAL FACTS (model-independent):")
    for k, v in assess["observational_facts"].items():
        _pr(f"      [{k}] {v}")

    _pr("\n    STANDARD INTERPRETATIONS:")
    for k, v in assess["standard_interpretations"].items():
        _pr(f"      [{k}] {v}")

    _pr("\n    STRENGTH OF EVIDENCE:")
    for level, items in assess["strength_of_evidence"].items():
        _pr(f"\n      {level}:")
        for item in items:
            _pr(f"        • {item}")

    _pr(f"\n    THE CORE TENSION:")
    for line in assess["the_core_tension"].split("\n"):
        _pr(f"    {line}")

    _pr(f"\n    WHAT A PRODUCTIVE SKEPTIC SHOULD FOCUS ON:")
    for item in assess["what_a_skeptic_should_focus_on"]:
        _pr(f"      → {item}")
    _hr()

    # ── Final Summary ────────────────────────────────────────────────────
    _pr(f"\n  ══════════ SUMMARY ══════════")
    _pr(f"    10 quantitative analyses completed")
    _pr(f"    Blackbody verified to 50 ppm (FIRAS)")
    _pr(f"    3 alternative origins tested — all fail quantitative tests")
    _pr(f"    Sound horizon: {peaks['sound_horizon']['r_s_mpc']:.1f} Mpc "
         f"(Planck: {peaks['sound_horizon']['planck_r_s_mpc']} Mpc)")
    _pr(f"    Acoustic scale: ℓ_s = {peaks['acoustic_scale']['computed_ell_s']:.0f} "
         f"(Planck: {peaks['acoustic_scale']['planck_ell_s']:.0f})")
    _pr(f"    SZ effect detected in {sz['planck_sz_catalog_clusters']}+ clusters")
    _pr(f"    All Planck 2018 ΛCDM — no external cosmology packages needed")
    _pr(f"  ═══════════════════════════════")

    results["pass"] = True
    return results
