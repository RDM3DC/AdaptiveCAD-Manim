"""Cosmological Calculator — Friedmann-equation numerics for high-z galaxies.

Uses Planck 2018 flat ΛCDM parameters to compute lookback time, cosmic age,
comoving / luminosity / angular-diameter distances, Lyman-break position,
and angular sizes for the highest-redshift spectroscopically confirmed galaxies.

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

    results["pass"] = mom["all_pass"]
    return results
