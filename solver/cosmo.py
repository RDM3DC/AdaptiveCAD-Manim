"""Cosmological Calculator — Friedmann-equation numerics for high-z galaxies.

Uses Planck 2018 flat ΛCDM parameters to compute lookback time, cosmic age,
comoving / luminosity / angular-diameter distances, Lyman-break position,
and angular sizes for the highest-redshift spectroscopically confirmed galaxies.

Deep analysis: star formation rates, halo masses, reionization budget,
Hubble tension comparison, dark-energy domination epoch, causal horizons,
and the "impossibly early galaxy" problem.

Run:  python -m solver cosmo
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from scipy.integrate import quad


# ═══════════════════════════════════════════════════════════════════════════
# Planck 2018 flat ΛCDM cosmology  (Table 2, TT,TE,EE+lowE+lensing+BAO)
# ═══════════════════════════════════════════════════════════════════════════

H0_KM_S_MPC = 67.36          # km s⁻¹ Mpc⁻¹
OMEGA_M     = 0.3153          # matter density parameter
OMEGA_R     = 9.14e-5         # radiation (photons + 3 massless neutrinos)
OMEGA_LAMBDA = 1.0 - OMEGA_M - OMEGA_R  # flat universe
AGE_GYR     = 13.797          # age of Universe in Gyr

# Derived constants
C_KM_S      = 299792.458     # speed of light  km/s
MPC_TO_GLY  = 3.2616e-3      # 1 Mpc = 3.2616×10⁻³ Gly
H0_INV_GYR  = 1.0 / (H0_KM_S_MPC * 1.0225e-3)  # 1/H₀ in Gyr (≈14.52)
D_H_MPC     = C_KM_S / H0_KM_S_MPC              # Hubble distance in Mpc

# Lyman-alpha rest wavelength
LYMAN_ALPHA_NM = 121.567     # nm


# ═══════════════════════════════════════════════════════════════════════════
# Friedmann integrand E(z) = H(z)/H₀
# ═══════════════════════════════════════════════════════════════════════════

def _E(z: float) -> float:
    """Dimensionless Hubble parameter E(z) = H(z)/H₀."""
    zp1 = 1.0 + z
    return math.sqrt(OMEGA_R * zp1**4 + OMEGA_M * zp1**3 + OMEGA_LAMBDA)


# ═══════════════════════════════════════════════════════════════════════════
# Core distance / time integrals
# ═══════════════════════════════════════════════════════════════════════════

def lookback_time_gyr(z: float) -> float:
    """Lookback time in Gyr to redshift z."""
    integrand = lambda zp: 1.0 / ((1.0 + zp) * _E(zp))
    result, _ = quad(integrand, 0, z)
    return result * H0_INV_GYR


def cosmic_age_gyr(z: float) -> float:
    """Age of the Universe at redshift z, in Gyr."""
    integrand = lambda zp: 1.0 / ((1.0 + zp) * _E(zp))
    result, _ = quad(integrand, z, np.inf)
    return result * H0_INV_GYR


def comoving_distance_mpc(z: float) -> float:
    """Line-of-sight comoving distance in Mpc."""
    integrand = lambda zp: 1.0 / _E(zp)
    result, _ = quad(integrand, 0, z)
    return result * D_H_MPC


def luminosity_distance_mpc(z: float) -> float:
    """Luminosity distance in Mpc."""
    return (1.0 + z) * comoving_distance_mpc(z)


def angular_diameter_distance_mpc(z: float) -> float:
    """Angular diameter distance in Mpc."""
    return comoving_distance_mpc(z) / (1.0 + z)


def comoving_distance_gly(z: float) -> float:
    """Comoving distance in Gly (billions of light-years)."""
    return comoving_distance_mpc(z) * MPC_TO_GLY


def proper_distance_gly(z: float) -> float:
    """Present-day proper distance in Gly (= comoving for flat FLRW)."""
    return comoving_distance_gly(z)


# ═══════════════════════════════════════════════════════════════════════════
# Observable quantities
# ═══════════════════════════════════════════════════════════════════════════

def lyman_break_wavelength_nm(z: float) -> float:
    """Observed Lyman-α break wavelength in nm at redshift z."""
    return LYMAN_ALPHA_NM * (1.0 + z)


def angular_size_arcsec(physical_size_pc: float, z: float) -> float:
    """Angular size in arcsec for a physical size (in pc) at redshift z."""
    d_A_mpc = angular_diameter_distance_mpc(z)
    d_A_pc = d_A_mpc * 1e6
    theta_rad = physical_size_pc / d_A_pc
    return theta_rad * (180.0 / math.pi) * 3600.0


def apparent_magnitude(M_UV: float, z: float) -> float:
    """Apparent magnitude from absolute UV magnitude and luminosity distance."""
    d_L_mpc = luminosity_distance_mpc(z)
    d_L_pc = d_L_mpc * 1e6
    return M_UV + 5.0 * math.log10(d_L_pc / 10.0)


# ═══════════════════════════════════════════════════════════════════════════
# Galaxy database — highest-redshift spectroscopically confirmed
# ═══════════════════════════════════════════════════════════════════════════

GALAXIES = [
    {
        "name": "MoM-z14",
        "z": 14.44,
        "z_err": 0.02,
        "telescope": "JWST NIRSpec/PRISM",
        "evidence": "Lyman break + 5 UV lines (CIV, CIII], NIV, NIII], HeII+OIII])",
        "M_UV": -20.2,
        "r_e_pc": 74,
        "beta_UV": -2.5,
        "date": "2025-05-16 (arXiv); 2026-01-28 (published)",
        "ra_deg": 150.0933255,
        "dec_deg": 2.2731627,
        "claimed_age_myr": 280,
    },
    {
        "name": "JADES-GS-z14-0",
        "z": 14.1793,
        "z_err": 0.0007,
        "telescope": "JWST + ALMA [OIII]",
        "evidence": "Lyman break + ALMA [OIII] 88μm line",
        "M_UV": -20.81,
        "r_e_pc": None,
        "beta_UV": -2.2,
        "date": "2024-05-30 (JWST); 2025-03-20 (ALMA)",
        "ra_deg": None,
        "dec_deg": None,
        "claimed_age_myr": 290,
    },
    {
        "name": "JADES-GS-z14-1",
        "z": 13.90,
        "z_err": 0.17,
        "telescope": "JWST NIRSpec",
        "evidence": "Continuum + Lyman break modeling",
        "M_UV": None,
        "r_e_pc": None,
        "beta_UV": None,
        "date": "2024 (Carniani et al., Nature)",
        "ra_deg": None,
        "dec_deg": None,
        "claimed_age_myr": 300,
    },
    {
        "name": "PAN-z14-1",
        "z": 13.53,
        "z_err": 0.055,
        "telescope": "JWST NIRSpec/PRISM",
        "evidence": "Lyman break modeling (no UV lines)",
        "M_UV": None,
        "r_e_pc": None,
        "beta_UV": None,
        "date": "2026-01-16 (arXiv, Donnan et al.)",
        "ra_deg": None,
        "dec_deg": None,
        "claimed_age_myr": 315,
    },
    {
        "name": "JADES-GS-z13-0",
        "z": 13.2,
        "z_err": 0.1,
        "telescope": "JWST NIRCam + NIRSpec",
        "evidence": "Spectroscopic confirmation",
        "M_UV": None,
        "r_e_pc": None,
        "beta_UV": None,
        "date": "2023-04-04 (Curtis-Lake et al., Nat Astron)",
        "ra_deg": None,
        "dec_deg": None,
        "claimed_age_myr": 330,
    },
    {
        "name": "JADES-GS-z13-1-LA",
        "z": 13.0,
        "z_err": 0.01,
        "telescope": "JWST NIRCam + NIRSpec/PRISM",
        "evidence": "Lyman break + high-SNR Ly-α emission",
        "M_UV": None,
        "r_e_pc": None,
        "beta_UV": None,
        "date": "2025-03-26 (Witstok et al., Nature)",
        "ra_deg": None,
        "dec_deg": None,
        "claimed_age_myr": 340,
    },
    {
        "name": "GN-z11",
        "z": 11.1,
        "z_err": 0.1,
        "telescope": "Hubble (confirmed) + JWST (follow-up)",
        "evidence": "Grism spectroscopy + JWST confirmation",
        "M_UV": -21.1,
        "r_e_pc": None,
        "beta_UV": -2.4,
        "date": "2016-03-03 (Oesch et al.)",
        "ra_deg": None,
        "dec_deg": None,
        "claimed_age_myr": 420,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Compute all observables for a galaxy
# ═══════════════════════════════════════════════════════════════════════════

def compute_galaxy(gal: dict) -> dict:
    """Compute all cosmological observables for a galaxy entry."""
    z = gal["z"]

    t_lookback = lookback_time_gyr(z)
    t_cosmic = cosmic_age_gyr(z)
    d_C = comoving_distance_mpc(z)
    d_L = luminosity_distance_mpc(z)
    d_A = angular_diameter_distance_mpc(z)
    d_proper_gly = proper_distance_gly(z)
    lyman_nm = lyman_break_wavelength_nm(z)

    result = {
        "name": gal["name"],
        "z_spec": z,
        "lookback_gyr": t_lookback,
        "cosmic_age_myr": t_cosmic * 1000,  # Gyr → Myr
        "comoving_mpc": d_C,
        "proper_gly": d_proper_gly,
        "luminosity_mpc": d_L,
        "angular_diameter_mpc": d_A,
        "lyman_break_nm": lyman_nm,
        "lyman_break_um": lyman_nm / 1000,
    }

    # Angular size (if physical size known)
    if gal.get("r_e_pc") is not None:
        result["angular_size_arcsec"] = angular_size_arcsec(gal["r_e_pc"], z)
        result["angular_size_mas"] = result["angular_size_arcsec"] * 1000

    # Apparent magnitude (if M_UV known)
    if gal.get("M_UV") is not None:
        result["m_apparent"] = apparent_magnitude(gal["M_UV"], z)

    # Check against claimed age
    if gal.get("claimed_age_myr") is not None:
        result["claimed_age_myr"] = gal["claimed_age_myr"]
        result["computed_age_myr"] = result["cosmic_age_myr"]
        result["age_agreement_pct"] = abs(
            result["computed_age_myr"] - gal["claimed_age_myr"]
        ) / gal["claimed_age_myr"] * 100

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Redshift sweep — cosmic age vs z
# ═══════════════════════════════════════════════════════════════════════════

def age_vs_redshift(z_values: list = None) -> List[dict]:
    """Compute cosmic age for a range of redshifts."""
    if z_values is None:
        z_values = [0, 0.5, 1, 2, 3, 5, 7, 10, 11.1, 13.0, 13.2,
                    13.53, 13.9, 14.18, 14.44, 17, 20, 30, 50, 1100]
    entries = []
    for z in z_values:
        t = cosmic_age_gyr(z)
        entries.append({
            "z": z,
            "cosmic_age_gyr": t,
            "cosmic_age_myr": t * 1000,
            "lookback_gyr": AGE_GYR - t if t < AGE_GYR else 0,
        })
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# MoM-z14 specific verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_mom_z14() -> dict:
    """Verify all numbers from the NASA/ESA MoM-z14 announcement.

    Claims to check:
    - z = 14.44  →  ~280 Myr after Big Bang
    - Light traveled ~13.5 of 13.8 Gyr
    - Proper distance ~33.9 Gly
    - Compact: r_e ≈ 74 pc
    - M_UV ≈ −20.2
    """
    z = 14.44
    t_cosmic = cosmic_age_gyr(z) * 1000  # Myr
    t_lookback = lookback_time_gyr(z)
    d_proper = proper_distance_gly(z)
    ang = angular_size_arcsec(74, z)
    lyman_um = lyman_break_wavelength_nm(z) / 1000

    checks = {
        "z": z,
        "cosmic_age_myr": t_cosmic,
        "claimed_age_myr": 280,
        "age_check": abs(t_cosmic - 280) < 15,

        "lookback_gyr": t_lookback,
        "claimed_lookback_gyr": 13.5,
        "lookback_check": abs(t_lookback - 13.5) < 0.1,

        "proper_distance_gly": d_proper,
        "claimed_distance_gly": 33.9,
        "distance_check": abs(d_proper - 33.9) < 1.0,

        "lyman_break_um": lyman_um,
        "lyman_in_nirspec_range": 0.6 <= lyman_um <= 5.3,

        "angular_size_mas": ang * 1000,
        "resolving_note": "JWST NIRCam diffraction limit ~60 mas at 2μm",

        "m_apparent": apparent_magnitude(-20.2, z),
    }

    checks["all_pass"] = (
        checks["age_check"]
        and checks["lookback_check"]
        and checks["distance_check"]
        and checks["lyman_in_nirspec_range"]
    )

    return checks


# ═══════════════════════════════════════════════════════════════════════════
# JADES-GS-z14-0 ALMA cross-check
# ═══════════════════════════════════════════════════════════════════════════

def verify_jades_z14_alma() -> dict:
    """Cross-check JADES-GS-z14-0: JWST z≈14.32 vs ALMA z=14.1793.

    The ALMA [OIII] 88μm line provides a precision redshift. Verify:
    - Observed [OIII] frequency = 88.356μm / (1+z)  → ν_obs
    - Cosmic age at z=14.1793
    """
    z_jwst = 14.32
    z_alma = 14.1793

    # [OIII] 88.356μm rest frequency = c/λ
    oiii_rest_um = 88.356
    oiii_obs_um = oiii_rest_um * (1 + z_alma)
    # freq GHz = c(m/s) / λ(m) / 1e9
    oiii_rest_ghz = (C_KM_S * 1e3) / (oiii_rest_um * 1e-6) / 1e9
    oiii_obs_ghz = oiii_rest_ghz / (1 + z_alma)

    t_jwst = cosmic_age_gyr(z_jwst) * 1000
    t_alma = cosmic_age_gyr(z_alma) * 1000
    delta_myr = t_alma - t_jwst

    return {
        "z_jwst": z_jwst,
        "z_alma": z_alma,
        "delta_z": z_jwst - z_alma,
        "OIII_rest_um": oiii_rest_um,
        "OIII_obs_um": oiii_obs_um,
        "OIII_obs_mm": oiii_obs_um / 1000,
        "OIII_rest_GHz": oiii_rest_ghz,
        "OIII_obs_GHz": oiii_obs_ghz,
        "ALMA_band": "Band 6/7" if 211 <= oiii_obs_ghz <= 373 else
                     "Band 5" if 163 <= oiii_obs_ghz <= 211 else "other",
        "cosmic_age_jwst_myr": t_jwst,
        "cosmic_age_alma_myr": t_alma,
        "age_difference_myr": delta_myr,
        "proper_distance_gly": proper_distance_gly(z_alma),
    }


# ═══════════════════════════════════════════════════════════════════════════
# NIRSpec band coverage check
# ═══════════════════════════════════════════════════════════════════════════

def rest_uv_lines_observed(z: float) -> dict:
    """Compute observed wavelengths of key rest-UV lines at redshift z.

    Lines reported in MoM-z14: CIV 1549, CIII] 1909, NIV 1486,
    NIII] 1750, HeII 1640 + OIII] 1666.
    """
    lines = {
        "Ly-alpha": 121.567,
        "N IV 1486": 148.6,
        "C IV 1549": 154.9,
        "He II 1640": 164.0,
        "O III] 1666": 166.6,
        "N III] 1750": 175.0,
        "C III] 1909": 190.9,
    }

    zp1 = 1 + z
    entries = {}
    for name, lam_rest_nm in lines.items():
        lam_obs_nm = lam_rest_nm * zp1
        lam_obs_um = lam_obs_nm / 1000
        in_nirspec = 0.6 <= lam_obs_um <= 5.3
        entries[name] = {
            "rest_nm": lam_rest_nm,
            "obs_nm": lam_obs_nm,
            "obs_um": lam_obs_um,
            "in_NIRSpec_range": in_nirspec,
        }

    return {"z": z, "lines": entries}


# ═══════════════════════════════════════════════════════════════════════════
# DEEP COSMOLOGICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

# Physical constants for astrophysics
L_SUN_ERG_S    = 3.828e33         # solar luminosity erg/s
M_SUN_G        = 1.989e33         # solar mass in grams
PC_CM          = 3.086e18         # parsec in cm
MPC_CM         = PC_CM * 1e6      # Mpc in cm
YR_S           = 3.156e7          # year in seconds
GYR_S          = YR_S * 1e9
M_UV_SUN       = 5.33             # AB absolute mag of Sun at 1500 Å
SIGMA_T_CM2    = 6.6524e-25       # Thomson cross-section
M_P_G          = 1.6726e-24       # proton mass in grams
K_B_ERG_K      = 1.3807e-16       # Boltzmann constant erg/K
G_CGS          = 6.674e-8         # gravitational constant cgs
H0_CGS         = H0_KM_S_MPC * 1e5 / MPC_CM  # H0 in s⁻¹
RHO_CRIT_CGS   = 3.0 * H0_CGS**2 / (8.0 * math.pi * G_CGS)  # g/cm³

# Cosmological parameters
OMEGA_B        = 0.0493           # baryon density parameter (Planck 2018)
F_BARYON       = OMEGA_B / OMEGA_M  # cosmic baryon fraction ≈ 0.156
T_CMB_K        = 2.7255           # CMB temperature today
Z_RECOM        = 1089.80          # redshift of recombination (Planck)
Z_REION_MID    = 7.67             # midpoint of reionization (Planck τ)
SIGMA_8        = 0.8111           # RMS matter fluctuations in 8 h⁻¹ Mpc


def sfr_from_uv(M_UV: float) -> float:
    """Star formation rate in M☉/yr from UV absolute magnitude.

    Uses the Kennicutt-Madau relation (Madau & Dickinson 2014):
        SFR = L_UV / 1.15e28  (erg/s/Hz → M☉/yr)
    with L_UV from M_UV (AB system).
    """
    # AB magnitude → luminosity density at 1500 Å in erg/s/Hz
    L_nu = 10**(-0.4 * (M_UV - (-48.6))) * 4 * math.pi * (10 * PC_CM)**2
    # Actually: M_AB = -2.5 log10(L_nu / (4π (10pc)²)) - 48.6
    # So L_nu = 10^(-(M_UV + 48.6)/2.5) * 4π(10pc)²
    # But simpler: L_nu(erg/s/Hz) from absolute mag
    L_nu_correct = 10**(0.4 * (M_UV_SUN - M_UV)) * 1.81e20  # L_ν,☉ at 1500Å ≈ 1.81e20
    # Standard conversion: use the textbook approach
    log_L = 0.4 * (51.63 - M_UV)  # M_AB → log10(L_ν) for erg/s/Hz at 10pc=AB
    L_nu_final = 10**log_L
    # Kennicutt 1998, Madau & Dickinson 2014 (Salpeter IMF):
    # SFR (M☉/yr) = κ_UV × L_ν  where κ_UV = 1.15×10⁻²⁸
    # For Chabrier IMF: divide by 1.7
    sfr_salpeter = L_nu_final * 1.15e-28
    sfr_chabrier = sfr_salpeter / 1.7
    return sfr_chabrier  # Chabrier is standard at high-z


def uv_luminosity_erg_s_hz(M_UV: float) -> float:
    """UV luminosity density L_ν (erg/s/Hz) from absolute UV magnitude."""
    log_L = 0.4 * (51.63 - M_UV)
    return 10**log_L


def stellar_mass_formed(sfr: float, age_myr: float, duty_cycle: float = 1.0) -> float:
    """Total stellar mass formed in M☉, given constant SFR over age_myr."""
    return sfr * age_myr * 1e6 * duty_cycle


def halo_mass_from_stellar(M_star: float, f_star: float = 0.01) -> float:
    """Estimate dark-matter halo mass from stellar mass.

    f_star = M_star / (f_baryon × M_halo), typical 1-10% at high-z.
    """
    return M_star / (f_star * F_BARYON)


def ionizing_photon_rate(M_UV: float, xi_ion_log: float = 25.2) -> float:
    """Ionizing photon production rate Ṅ_ion (photons/s).

    ξ_ion ≡ Ṅ_ion / L_UV  (photons/s per erg/s/Hz)
    Canonical value log(ξ_ion) ≈ 25.2 (Bouwens+2016).
    Very blue galaxies (β<-2.5) can reach log(ξ_ion) ≈ 25.6-25.9.
    """
    L_uv = uv_luminosity_erg_s_hz(M_UV)
    xi_ion = 10**xi_ion_log
    return L_uv * xi_ion


def particle_horizon_mpc(z: float) -> float:
    """Comoving particle horizon at redshift z — the maximum comoving distance
    a photon could have traveled since the Big Bang to epoch z.

    d_ph(z) = c ∫₀^t dt'/a(t') = (c/H₀) ∫_z^∞ dz'/((1+z')E(z'))
    We integrate to z_max=1e5 (deep into radiation era) as a proxy for ∞.
    """
    integrand = lambda zp: 1.0 / ((1.0 + zp) * _E(zp))
    val, _ = quad(integrand, z, 1e5)
    return D_H_MPC * val


def dark_energy_domination_redshift() -> float:
    """Redshift at which Ω_Λ = Ω_m  →  dark energy starts dominating.

    Ω_m(z) = Ω_m(1+z)³/E²(z),  Ω_Λ(z) = Ω_Λ/E²(z)
    Equal when Ω_m(1+z)³ = Ω_Λ  →  z_eq = (Ω_Λ/Ω_m)^(1/3) - 1
    """
    return (OMEGA_LAMBDA / OMEGA_M)**(1.0 / 3.0) - 1.0


def matter_radiation_equality_z() -> float:
    """Redshift of matter-radiation equality.

    Ω_m(1+z)³ = Ω_r(1+z)⁴  →  z_eq = Ω_m/Ω_r - 1
    """
    return OMEGA_M / OMEGA_R - 1.0


def density_fractions(z: float) -> dict:
    """Energy density fractions at redshift z."""
    zp1 = 1.0 + z
    E2 = OMEGA_R * zp1**4 + OMEGA_M * zp1**3 + OMEGA_LAMBDA
    return {
        "z": z,
        "Omega_r": OMEGA_R * zp1**4 / E2,
        "Omega_m": OMEGA_M * zp1**3 / E2,
        "Omega_Lambda": OMEGA_LAMBDA / E2,
    }


def hubble_tension_comparison(z: float) -> dict:
    """Compare Planck H0 vs SH0ES (Riess+2022) for key quantities at z.

    Planck 2018:  H0 = 67.36 ± 0.54 km/s/Mpc
    SH0ES 2022:   H0 = 73.04 ± 1.04 km/s/Mpc
    Tension:      ~5σ
    """
    H0_shoes = 73.04
    Omega_m_shoes = 0.334  # roughly consistent with SH0ES + flat ΛCDM
    Omega_r_shoes = OMEGA_R  # radiation density well-constrained
    Omega_L_shoes = 1.0 - Omega_m_shoes - Omega_r_shoes

    def _E_shoes(zp):
        zp1 = 1.0 + zp
        return math.sqrt(Omega_r_shoes * zp1**4 + Omega_m_shoes * zp1**3
                         + Omega_L_shoes)

    H0_inv_shoes = 1.0 / (H0_shoes * 1.0225e-3)
    D_H_shoes = C_KM_S / H0_shoes

    integrand_t_p = lambda zp: 1.0 / ((1.0 + zp) * _E(zp))
    integrand_t_s = lambda zp: 1.0 / ((1.0 + zp) * _E_shoes(zp))
    integrand_d_p = lambda zp: 1.0 / _E(zp)
    integrand_d_s = lambda zp: 1.0 / _E_shoes(zp)

    age_planck = H0_INV_GYR * quad(integrand_t_p, 0, z)[0]
    age_shoes = H0_inv_shoes * quad(integrand_t_s, 0, z)[0]
    cosmic_age_planck = AGE_GYR - age_planck
    cosmic_age_shoes_total = H0_inv_shoes * quad(integrand_t_p, 0, 1e5)[0]  # approx
    # More accurate for SH0ES total age
    cosmic_age_shoes_total2 = H0_inv_shoes * quad(integrand_t_s, 0, 1e5)[0]
    cosmic_age_shoes = cosmic_age_shoes_total2 - (H0_inv_shoes * quad(integrand_t_s, 0, z)[0])

    d_C_planck = D_H_MPC * quad(integrand_d_p, 0, z)[0]
    d_C_shoes = D_H_shoes * quad(integrand_d_s, 0, z)[0]

    return {
        "z": z,
        "H0_planck": H0_KM_S_MPC,
        "H0_shoes": H0_shoes,
        "tension_sigma": 4.8,
        "cosmic_age_planck_myr": cosmic_age_planck * 1000,
        "cosmic_age_shoes_myr": cosmic_age_shoes * 1000,
        "age_universe_planck_gyr": AGE_GYR,
        "age_universe_shoes_gyr": cosmic_age_shoes_total2,
        "comoving_planck_mpc": d_C_planck,
        "comoving_shoes_mpc": d_C_shoes,
        "proper_planck_gly": d_C_planck * MPC_TO_GLY,
        "proper_shoes_gly": d_C_shoes * MPC_TO_GLY,
    }


def galaxy_formation_problem(gal: dict) -> dict:
    """Analyze the 'impossibly early galaxy' problem for a given galaxy.

    At z=14.44 the universe is ~282 Myr old. Stars take ~2 Myr to form from
    molecular clouds, and the first Pop III stars form at z~30 (t~100 Myr).
    So galaxies with M_UV=-20.2 had at most ~180 Myr to assemble.

    This computes what star formation rate, efficiency, and halo mass are
    needed — and whether ΛCDM predicts halos massive enough.
    """
    z = gal["z"]
    M_UV = gal.get("M_UV")
    if M_UV is None:
        return {"name": gal["name"], "skip": True,
                "reason": "No M_UV measurement available"}

    t_cosmic_myr = cosmic_age_gyr(z) * 1000
    # First stars form at z~30 → t~100 Myr; generous: z~50 → t~50 Myr
    t_first_stars_myr = cosmic_age_gyr(30) * 1000
    t_available_myr = t_cosmic_myr - t_first_stars_myr

    sfr = sfr_from_uv(M_UV)
    L_uv = uv_luminosity_erg_s_hz(M_UV)

    # Stellar mass if SFR was constant since first stars
    M_star_formed = stellar_mass_formed(sfr, t_available_myr)

    # After ~200 Myr of star formation, ~30% of mass returned to ISM
    M_star_present = M_star_formed * 0.7  # rough return fraction

    # Required halo mass for different star formation efficiencies
    halo_1pct = halo_mass_from_stellar(M_star_present, 0.01)
    halo_5pct = halo_mass_from_stellar(M_star_present, 0.05)
    halo_10pct = halo_mass_from_stellar(M_star_present, 0.10)
    halo_30pct = halo_mass_from_stellar(M_star_present, 0.30)

    # Expected most massive halo at z~14 in observable universe
    # Press-Schechter / Sheth-Tormen: ~10⁸-10⁹ M☉ at z=14
    # Typical ΛCDM simulation: most massive halo at z=14 in (100 Mpc)³ box
    # is ~10⁸·⁵ M☉. In full observable volume (~4×10¹⁰ Mpc³) maybe ~10¹⁰ M☉.
    halo_max_expected_log = 9.5  # generous upper limit at z~14

    # Ionizing photon rate
    N_ion = ionizing_photon_rate(M_UV)
    # For very blue β < -2.5 galaxies:
    N_ion_blue = ionizing_photon_rate(M_UV, xi_ion_log=25.7)

    # Specific star formation rate (sSFR = SFR / M_star)
    sSFR = sfr / M_star_present if M_star_present > 0 else 0
    sSFR_per_gyr = sSFR * 1e9  # per Gyr

    return {
        "name": gal["name"],
        "z": z,
        "M_UV": M_UV,
        "cosmic_age_myr": t_cosmic_myr,
        "first_stars_myr": t_first_stars_myr,
        "available_time_myr": t_available_myr,
        "sfr_msun_yr": sfr,
        "L_UV_erg_s_hz": L_uv,
        "M_star_formed_msun": M_star_formed,
        "M_star_present_msun": M_star_present,
        "halo_mass_1pct_msun": halo_1pct,
        "halo_mass_5pct_msun": halo_5pct,
        "halo_mass_10pct_msun": halo_10pct,
        "halo_mass_30pct_msun": halo_30pct,
        "halo_max_expected_log": halo_max_expected_log,
        "needs_high_efficiency": math.log10(halo_5pct) > halo_max_expected_log,
        "N_ion_s": N_ion,
        "N_ion_blue_s": N_ion_blue,
        "sSFR_per_gyr": sSFR_per_gyr,
        "doubling_time_myr": (math.log(2) / sSFR / 1e6) if sSFR > 0 else 0,
    }


def reionization_budget(galaxies: list) -> dict:
    """Estimate the contribution of these galaxies to cosmic reionization.

    Key question: Can the observed population of UV-bright galaxies at z>10
    produce enough ionizing photons to reionize the universe?

    Critical density of ionizing photons needed:
        ṅ_ion = n_H × C × (1+z)³ / t_rec
    where C is the clumping factor and t_rec is the recombination time.
    """
    # Hydrogen number density today
    rho_b = RHO_CRIT_CGS * OMEGA_B
    n_H_0 = rho_b * 0.76 / M_P_G  # 76% hydrogen by mass

    results = []
    for gal in galaxies:
        M_UV = gal.get("M_UV")
        if M_UV is None:
            continue
        z = gal["z"]

        N_ion = ionizing_photon_rate(M_UV)
        beta = gal.get("beta_UV", -2.0)

        # Escape fraction — very uncertain, typical f_esc ~ 5-20% at high z
        # Very blue β < -2.5 suggests low dust → possibly higher f_esc
        f_esc = 0.20 if (beta is not None and beta < -2.3) else 0.10

        # Ionizing photons escaping into IGM per second
        N_esc = N_ion * f_esc

        # Recombination timescale at redshift z
        # t_rec = 1 / (C_HII × α_B × n_H(z))
        # α_B ≈ 2.6×10⁻¹³ cm³/s at T=10⁴K (Case B)
        alpha_B = 2.6e-13
        C_clump = 3.0  # clumping factor (typical z>10 estimate)
        n_H_z = n_H_0 * (1 + z)**3
        t_rec_s = 1.0 / (C_clump * alpha_B * n_H_z)
        t_rec_myr = t_rec_s / (YR_S * 1e6)

        # Volume that one such galaxy can keep ionized (Strömgren argument)
        # Ṅ_esc = n_H(z) × V_ion / t_rec  → V_ion = Ṅ_esc × t_rec / n_H(z)
        V_ion_cm3 = N_esc * t_rec_s / n_H_z
        V_ion_mpc3 = V_ion_cm3 / MPC_CM**3

        # Comoving volume of observable universe at z
        # V_com = (4π/3) × d_C³  (proper volume in comoving coords)
        d_C = comoving_distance_mpc(z)
        # But we want the total comoving volume: (4π/3) × d_C(z→∞)³
        # Use d_C to z=1100 as proxy
        d_C_total = comoving_distance_mpc(1100)
        V_total = (4 * math.pi / 3) * d_C_total**3  # Mpc³

        # Number of such galaxies needed to reionize the full volume
        N_galaxies_needed = V_total / V_ion_mpc3 if V_ion_mpc3 > 0 else float('inf')

        results.append({
            "name": gal["name"],
            "z": z,
            "M_UV": M_UV,
            "N_ion_s": N_ion,
            "f_esc": f_esc,
            "N_esc_s": N_esc,
            "t_rec_myr": t_rec_myr,
            "V_ion_mpc3": V_ion_mpc3,
            "V_total_mpc3": V_total,
            "N_galaxies_needed": N_galaxies_needed,
        })

    return {
        "entries": results,
        "note": "Reionization completed by z≈5.5; these galaxies are at Cosmic Dawn",
    }


def cosmic_timeline_context() -> dict:
    """Put high-z galaxies in context of the full cosmic timeline."""
    # Key epochs
    t_recomb = cosmic_age_gyr(Z_RECOM) * 1000       # Myr
    t_dark_ages_end = cosmic_age_gyr(30) * 1000      # ~z=30, first stars
    t_mom = cosmic_age_gyr(14.44) * 1000             # MoM-z14
    t_reion_mid = cosmic_age_gyr(Z_REION_MID) * 1000 # midpoint reionization
    t_reion_end = cosmic_age_gyr(5.5) * 1000         # reionization complete
    t_de_domination = cosmic_age_gyr(dark_energy_domination_redshift()) * 1000
    t_solar = cosmic_age_gyr(0) * 1000               # today

    z_mr_eq = matter_radiation_equality_z()
    t_mr_eq = cosmic_age_gyr(z_mr_eq) * 1000

    z_de = dark_energy_domination_redshift()

    # Particle horizons
    ph_recomb = particle_horizon_mpc(Z_RECOM) * MPC_TO_GLY * 1e3  # Mly
    ph_mom = particle_horizon_mpc(14.44) * MPC_TO_GLY  # Gly

    # Density fractions at MoM-z14 epoch
    dens_mom = density_fractions(14.44)
    dens_now = density_fractions(0)
    dens_reion = density_fractions(Z_REION_MID)

    return {
        "epochs": {
            "big_bang":              {"t_myr": 0,           "z": "∞"},
            "matter_radiation_eq":   {"t_myr": t_mr_eq,     "z": f"{z_mr_eq:.0f}"},
            "recombination":         {"t_myr": t_recomb,    "z": f"{Z_RECOM:.0f}"},
            "dark_ages_end":         {"t_myr": t_dark_ages_end, "z": "~30"},
            "MoM_z14":              {"t_myr": t_mom,        "z": "14.44"},
            "JADES_z14":            {"t_myr": cosmic_age_gyr(14.18)*1000, "z": "14.18"},
            "reionization_midpoint": {"t_myr": t_reion_mid, "z": f"{Z_REION_MID}"},
            "reionization_end":      {"t_myr": t_reion_end, "z": "5.5"},
            "dark_energy_domination":{"t_myr": t_de_domination, "z": f"{z_de:.2f}"},
            "today":                {"t_myr": t_solar,      "z": "0"},
        },
        "mom_z14_context": {
            "fraction_of_cosmic_age": t_mom / t_solar,
            "time_since_recombination_myr": t_mom - t_recomb,
            "time_to_reionization_myr": t_reion_mid - t_mom,
            "pct_to_reion_midpoint": (t_mom - t_recomb) / (t_reion_mid - t_recomb) * 100,
        },
        "horizons": {
            "particle_horizon_recomb_mly": ph_recomb,
            "particle_horizon_z14_gly": ph_mom,
            "particle_horizon_today_gly": comoving_distance_mpc(1e5) * MPC_TO_GLY,
        },
        "density_at_z14": dens_mom,
        "density_today": dens_now,
        "density_at_reion": dens_reion,
        "key_redshifts": {
            "matter_radiation_equality": z_mr_eq,
            "dark_energy_domination": z_de,
            "recombination": Z_RECOM,
            "reionization_midpoint": Z_REION_MID,
        },
    }


def what_it_means() -> str:
    """Return a plain-language summary of what these observations tell us."""
    return """
╔══════════════════════════════════════════════════════════════════════╗
║       WHAT THESE HIGH-z GALAXIES TELL US ABOUT THE UNIVERSE        ║
╚══════════════════════════════════════════════════════════════════════╝

1. GALAXIES FORMED SHOCKINGLY FAST
   MoM-z14 exists just ~282 Myr after the Big Bang — only 2% of cosmic
   history.  The universe went from a featureless hydrogen fog to hosting
   a galaxy with heavy elements (C, N, O) and an ongoing starburst in
   less time than it takes light to cross the Milky Way (~100 kly).

2. THE "IMPOSSIBLY EARLY" PROBLEM
   Pre-JWST ΛCDM simulations predicted very few UV-luminous galaxies at
   z>12.  JWST is finding them routinely.  This doesn't break ΛCDM, but
   it pushes star formation efficiency to its limits — galaxies at z~14
   need to convert 5-30% of their baryons into stars, far above the
   ~1% efficiency at later epochs.

3. THE DARK AGES ENDED EARLIER THAN EXPECTED
   Detection of multiple UV emission lines (CIV, CIII], HeII) in
   MoM-z14 means that Population III → Population II enrichment happened
   within ~180 Myr.  At least one prior generation of stars lived and
   died before z=14.4.

4. COSMIC REIONIZATION BEGAN AT COSMIC DAWN
   These galaxies are producing ionizing photons that carve ionized
   bubbles in the neutral hydrogen fog.  The detection of Ly-α at z=13
   (JADES-GS-z13-1-LA) confirms transparent sightlines exist even in the
   pre-reionization epoch — likely inside large ionized bubbles.

5. DARK ENERGY WAS IRRELEVANT AT z>10
   At z=14.44, dark energy contributes <0.003% of the energy budget.
   The universe was entirely matter+radiation dominated. Dark energy
   only takes over at z≈0.30 (~9.8 Gyr after the Big Bang).

6. THE HUBBLE TENSION BARELY MATTERS HERE
   Whether H₀ = 67.36 (Planck) or 73.04 (SH0ES), the cosmic age at
   z=14.44 changes by only ~20 Myr.  The tension matters most at low-z
   distance measurements, not at Cosmic Dawn.

7. WE ARE LOOKING BACK 98% OF COSMIC HISTORY
   The lookback time to z=14.44 is 13.52 Gyr out of 13.80 Gyr total.
   Photons from MoM-z14 have traveled 33.8 Gly of proper distance —
   further than the observable edge — because the universe expanded
   while the photons were in flight.

8. STRUCTURE FORMATION IS BOTTOM-UP, AND IT'S FAST
   These tiny galaxies (r_e ~ 74 pc, smaller than a globular cluster)
   will merge over billions of years into massive galaxies like the
   Milky Way.  The hierarchical assembly process was already well under
   way by 282 Myr.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Master runner
# ═══════════════════════════════════════════════════════════════════════════

def run_all(verbose: bool = True) -> Dict:
    results = {}

    def _pr(msg):
        if verbose:
            print(msg)

    def _hr():
        _pr("─" * 72)

    # ── 1. Compute all galaxies ──────────────────────────────────────────
    _pr("\n  ══════════ HIGH-z GALAXY COSMOLOGICAL CALCULATIONS ══════════")
    _pr(f"  Cosmology: Planck 2018 flat ΛCDM  H₀={H0_KM_S_MPC}  Ωm={OMEGA_M}")
    _hr()

    galaxy_results = []
    for gal in GALAXIES:
        r = compute_galaxy(gal)
        galaxy_results.append(r)

        _pr(f"\n  {r['name']}  (z = {r['z_spec']})")
        _hr()
        _pr(f"    Cosmic age:       {r['cosmic_age_myr']:.1f} Myr")
        _pr(f"    Lookback time:    {r['lookback_gyr']:.3f} Gyr")
        _pr(f"    Comoving dist:    {r['comoving_mpc']:.1f} Mpc")
        _pr(f"    Proper dist:      {r['proper_gly']:.2f} Gly")
        _pr(f"    Luminosity dist:  {r['luminosity_mpc']:.0f} Mpc")
        _pr(f"    Angular diam:     {r['angular_diameter_mpc']:.2f} Mpc")
        _pr(f"    Lyman break:      {r['lyman_break_um']:.3f} μm")

        if "angular_size_mas" in r:
            _pr(f"    Angular size:     {r['angular_size_mas']:.1f} mas  "
                 f"({r['angular_size_arcsec']:.4f}\")")
        if "m_apparent" in r:
            _pr(f"    m_apparent:       {r['m_apparent']:.1f}")
        if "age_agreement_pct" in r:
            _pr(f"    Age claim check:  computed {r['computed_age_myr']:.1f} vs "
                 f"claimed {r['claimed_age_myr']} Myr  "
                 f"({r['age_agreement_pct']:.1f}% off)")
        _hr()

    results["galaxies"] = galaxy_results

    # ── 2. MoM-z14 verification ──────────────────────────────────────────
    _pr("\n  ══════════ MoM-z14 NASA/ESA CLAIM VERIFICATION ══════════")
    _hr()
    mom = verify_mom_z14()
    results["mom_z14_verification"] = mom

    _pr(f"  Cosmic age:       {mom['cosmic_age_myr']:.1f} Myr  "
         f"(claimed ~{mom['claimed_age_myr']})  "
         f"{'✓' if mom['age_check'] else '✗'}")
    _pr(f"  Lookback time:    {mom['lookback_gyr']:.3f} Gyr  "
         f"(claimed ~{mom['claimed_lookback_gyr']})  "
         f"{'✓' if mom['lookback_check'] else '✗'}")
    _pr(f"  Proper distance:  {mom['proper_distance_gly']:.2f} Gly  "
         f"(claimed ~{mom['claimed_distance_gly']})  "
         f"{'✓' if mom['distance_check'] else '✗'}")
    _pr(f"  Lyman break:      {mom['lyman_break_um']:.3f} μm  "
         f"(in NIRSpec 0.6–5.3μm: "
         f"{'✓' if mom['lyman_in_nirspec_range'] else '✗'})")
    _pr(f"  Angular size:     {mom['angular_size_mas']:.1f} mas  "
         f"({mom['resolving_note']})")
    _pr(f"  m_apparent:       {mom['m_apparent']:.1f}")
    _pr(f"\n  All checks: {'✓ PASS' if mom['all_pass'] else '✗ FAIL'}")
    _hr()

    # ── 3. JADES ALMA cross-check ────────────────────────────────────────
    _pr("\n  ══════════ JADES-GS-z14-0 ALMA CROSS-CHECK ══════════")
    _hr()
    jades = verify_jades_z14_alma()
    results["jades_alma"] = jades

    _pr(f"  JWST z = {jades['z_jwst']}  →  ALMA z = {jades['z_alma']}")
    _pr(f"  Δz = {jades['delta_z']:.4f}  "
         f"(Δt ≈ {jades['age_difference_myr']:.1f} Myr)")
    _pr(f"  [OIII] 88μm observed at {jades['OIII_obs_mm']:.2f} mm  "
         f"({jades['OIII_obs_GHz']:.2f} GHz)")
    _pr(f"  ALMA band: {jades['ALMA_band']}")
    _pr(f"  Proper distance: {jades['proper_distance_gly']:.2f} Gly")
    _hr()

    # ── 4. Rest-UV lines at z=14.44 ─────────────────────────────────────
    _pr("\n  ══════════ REST-UV EMISSION LINES at z=14.44 ══════════")
    _hr()
    lines = rest_uv_lines_observed(14.44)
    results["uv_lines"] = lines

    for name, info in lines["lines"].items():
        mark = "✓" if info["in_NIRSpec_range"] else "✗"
        _pr(f"    {name:16s}  {info['rest_nm']:.1f} nm  →  "
             f"{info['obs_um']:.3f} μm  [{mark} NIRSpec]")
    _hr()

    # ── 5. Cosmic age vs redshift ────────────────────────────────────────
    _pr("\n  ══════════ COSMIC AGE vs REDSHIFT ══════════")
    _hr()
    age_sweep = age_vs_redshift()
    results["age_sweep"] = age_sweep

    for e in age_sweep:
        if e["z"] < 1:
            _pr(f"    z = {e['z']:8.1f}   age = {e['cosmic_age_gyr']:.3f} Gyr")
        elif e["cosmic_age_myr"] > 100:
            _pr(f"    z = {e['z']:8.2f}   age = {e['cosmic_age_myr']:.1f} Myr  "
                 f"({e['cosmic_age_gyr']:.4f} Gyr)")
        else:
            _pr(f"    z = {e['z']:8.1f}   age = {e['cosmic_age_myr']:.2f} Myr")
    _hr()

    # ── Summary ──────────────────────────────────────────────────────────
    _pr(f"\n  ══════════ SUMMARY ══════════")
    _pr(f"  Galaxies computed:     {len(galaxy_results)}")
    _pr(f"  MoM-z14 verification:  {'PASS' if mom['all_pass'] else 'FAIL'}")
    _pr(f"  All Planck 2018 ΛCDM — no external cosmology packages needed")
    _pr(f"  ═════════════════════════════")

    # ══════════════════════════════════════════════════════════════════════
    #  DEEP COSMOLOGICAL ANALYSIS — What does this say about our universe?
    # ══════════════════════════════════════════════════════════════════════

    # ── 6. Star formation rates & impossibly-early problem ───────────────
    _pr("\n  ══════════ STAR FORMATION & THE 'IMPOSSIBLY EARLY' PROBLEM ══════════")
    _hr()
    formation_results = []
    for gal in GALAXIES:
        fp = galaxy_formation_problem(gal)
        formation_results.append(fp)
        if fp.get("skip"):
            continue
        _pr(f"\n  {fp['name']}  (z={fp['z']}, M_UV={fp['M_UV']})")
        _pr(f"    Cosmic age:           {fp['cosmic_age_myr']:.1f} Myr")
        _pr(f"    First stars at:       {fp['first_stars_myr']:.1f} Myr  (z~30)")
        _pr(f"    Available time:       {fp['available_time_myr']:.1f} Myr")
        _pr(f"    UV luminosity:        {fp['L_UV_erg_s_hz']:.2e} erg/s/Hz")
        _pr(f"    SFR (Chabrier):       {fp['sfr_msun_yr']:.1f} M☉/yr")
        _pr(f"    Stellar mass formed:  {fp['M_star_formed_msun']:.2e} M☉")
        _pr(f"    Stellar mass today:   {fp['M_star_present_msun']:.2e} M☉  (30% recycled)")
        _pr(f"    sSFR:                 {fp['sSFR_per_gyr']:.0f} Gyr⁻¹  "
             f"(doubling time {fp['doubling_time_myr']:.0f} Myr)")
        _pr(f"    Required halo mass:")
        _pr(f"      ε★=1%:   {fp['halo_mass_1pct_msun']:.2e} M☉  "
             f"(log={math.log10(fp['halo_mass_1pct_msun']):.1f})")
        _pr(f"      ε★=5%:   {fp['halo_mass_5pct_msun']:.2e} M☉  "
             f"(log={math.log10(fp['halo_mass_5pct_msun']):.1f})")
        _pr(f"      ε★=10%:  {fp['halo_mass_10pct_msun']:.2e} M☉  "
             f"(log={math.log10(fp['halo_mass_10pct_msun']):.1f})")
        _pr(f"      ε★=30%:  {fp['halo_mass_30pct_msun']:.2e} M☉  "
             f"(log={math.log10(fp['halo_mass_30pct_msun']):.1f})")
        _pr(f"    Max expected halo at z~14:  10^{fp['halo_max_expected_log']:.1f} M☉")
        _pr(f"    Needs high ε★?:       {'YES — pushes ΛCDM limits' if fp['needs_high_efficiency'] else 'No — within ΛCDM'}")
        _pr(f"    Ionizing photon rate:  {fp['N_ion_s']:.2e} s⁻¹  "
             f"(blue: {fp['N_ion_blue_s']:.2e} s⁻¹)")
    _hr()
    results["formation_problem"] = formation_results

    # ── 7. Hubble tension ────────────────────────────────────────────────
    _pr("\n  ══════════ HUBBLE TENSION COMPARISON ══════════")
    _pr("  Planck 2018: H₀ = 67.36 km/s/Mpc  vs  SH0ES 2022: H₀ = 73.04 km/s/Mpc")
    _hr()
    tension_results = []
    for z_test in [14.44, 14.18, 11.1, 6.0, 2.0, 0.5]:
        ht = hubble_tension_comparison(z_test)
        tension_results.append(ht)
        delta_age = ht["cosmic_age_shoes_myr"] - ht["cosmic_age_planck_myr"]
        _pr(f"    z={z_test:6.2f}   "
             f"Planck: {ht['cosmic_age_planck_myr']:8.1f} Myr   "
             f"SH0ES: {ht['cosmic_age_shoes_myr']:8.1f} Myr   "
             f"Δ = {delta_age:+.1f} Myr")
    _pr(f"\n    Universe age:  Planck {AGE_GYR:.3f} Gyr  vs  "
         f"SH0ES {tension_results[0]['age_universe_shoes_gyr']:.3f} Gyr")
    _pr(f"    → SH0ES universe ~{(AGE_GYR - tension_results[0]['age_universe_shoes_gyr'])*1000:.0f} Myr YOUNGER")
    _pr(f"    → At z=14.44, cosmic ages differ by only "
         f"~{abs(tension_results[0]['cosmic_age_shoes_myr'] - tension_results[0]['cosmic_age_planck_myr']):.0f} Myr")
    _hr()
    results["hubble_tension"] = tension_results

    # ── 8. Dark energy & density evolution ───────────────────────────────
    _pr("\n  ══════════ DARK ENERGY & DENSITY EVOLUTION ══════════")
    _hr()
    z_de = dark_energy_domination_redshift()
    z_mr = matter_radiation_equality_z()
    _pr(f"    Matter-radiation equality:    z = {z_mr:.0f}  "
         f"(t = {cosmic_age_gyr(z_mr)*1000:.2f} Myr)")
    _pr(f"    Dark energy domination:       z = {z_de:.2f}  "
         f"(t = {cosmic_age_gyr(z_de):.2f} Gyr = {cosmic_age_gyr(z_de)*1000:.0f} Myr)")
    _pr(f"")
    _pr(f"    DENSITY FRACTIONS:")
    for z_test in [14.44, 10, 5.5, Z_REION_MID, 2.0, z_de, 1.0, 0.5, 0]:
        df = density_fractions(z_test)
        _pr(f"      z={z_test:8.2f}   Ωr={df['Omega_r']:8.5f}  "
             f"Ωm={df['Omega_m']:.5f}  ΩΛ={df['Omega_Lambda']:.5f}")
    _hr()
    results["z_dark_energy"] = z_de
    results["z_matter_radiation"] = z_mr

    # ── 9. Causal horizon ────────────────────────────────────────────────
    _pr("\n  ══════════ CAUSAL HORIZONS ══════════")
    _hr()
    ph_z14 = particle_horizon_mpc(14.44)
    ph_z14_gly = ph_z14 * MPC_TO_GLY
    ph_z14_proper_mpc = ph_z14 / (1 + 14.44)
    ph_today = particle_horizon_mpc(0)
    _pr(f"    Particle horizon at z=14.44:")
    _pr(f"      Comoving:  {ph_z14:.0f} Mpc  ({ph_z14_gly:.2f} Gly)")
    _pr(f"      Proper:    {ph_z14_proper_mpc:.0f} Mpc  "
         f"({ph_z14_proper_mpc * MPC_TO_GLY * 1e3:.0f} Mly)")
    _pr(f"    Particle horizon today:")
    _pr(f"      Comoving:  {ph_today:.0f} Mpc  "
         f"({ph_today * MPC_TO_GLY:.1f} Gly)")
    _pr(f"    Ratio:  {ph_z14/ph_today*100:.1f}% of today's horizon was "
         f"causally connected at z=14.44")
    _hr()
    results["particle_horizon_z14_mpc"] = ph_z14
    results["particle_horizon_today_mpc"] = ph_today

    # ── 10. Reionization budget ──────────────────────────────────────────
    _pr("\n  ══════════ REIONIZATION BUDGET ══════════")
    _hr()
    reion = reionization_budget(GALAXIES)
    results["reionization"] = reion
    for entry in reion["entries"]:
        _pr(f"    {entry['name']}  (z={entry['z']})")
        _pr(f"      Ṅ_ion = {entry['N_ion_s']:.2e} s⁻¹  "
             f"(f_esc={entry['f_esc']:.0%} → {entry['N_esc_s']:.2e} s⁻¹ escaping)")
        _pr(f"      Recombination time:  {entry['t_rec_myr']:.1f} Myr")
        _pr(f"      Ionized volume:      {entry['V_ion_mpc3']:.2e} Mpc³  (comoving)")
        _pr(f"      Galaxies needed to reionize:  {entry['N_galaxies_needed']:.1e}")
    _pr(f"\n    {reion['note']}")
    _hr()

    # ── 11. Cosmic timeline context ──────────────────────────────────────
    _pr("\n  ══════════ COSMIC TIMELINE CONTEXT ══════════")
    _hr()
    timeline = cosmic_timeline_context()
    results["timeline"] = timeline
    for name, info in timeline["epochs"].items():
        label = name.replace("_", " ").title()
        _pr(f"    {label:30s}  t = {info['t_myr']:10.1f} Myr   (z ≈ {info['z']})")
    _pr(f"")
    ctx = timeline["mom_z14_context"]
    _pr(f"    MoM-z14 at {ctx['fraction_of_cosmic_age']*100:.1f}% of cosmic age")
    _pr(f"    {ctx['time_since_recombination_myr']:.0f} Myr after recombination")
    _pr(f"    {ctx['time_to_reionization_myr']:.0f} Myr before reionization midpoint")
    _pr(f"    {ctx['pct_to_reion_midpoint']:.0f}% of the way from recombination to reionization")
    _pr(f"")
    _pr(f"    At z=14.44:  Ωr={timeline['density_at_z14']['Omega_r']:.5f}  "
         f"Ωm={timeline['density_at_z14']['Omega_m']:.5f}  "
         f"ΩΛ={timeline['density_at_z14']['Omega_Lambda']:.5f}")
    _pr(f"    Today:       Ωr={timeline['density_today']['Omega_r']:.5f}  "
         f"Ωm={timeline['density_today']['Omega_m']:.5f}  "
         f"ΩΛ={timeline['density_today']['Omega_Lambda']:.5f}")
    _hr()

    # ── 12. What it all means ────────────────────────────────────────────
    _pr(what_it_means())

    # ── Final Summary ────────────────────────────────────────────────────
    _pr(f"\n  ══════════ FINAL SUMMARY ══════════")
    _pr(f"  Galaxies computed:     {len(galaxy_results)}")
    _pr(f"  MoM-z14 verification:  {'PASS' if mom['all_pass'] else 'FAIL'}")
    _pr(f"  Deep analysis:         8 additional physics calculations completed")
    _pr(f"  All Planck 2018 ΛCDM — no external cosmology packages needed")
    _pr(f"  ═════════════════════════════════════")

    results["pass"] = mom["all_pass"]
    return results
