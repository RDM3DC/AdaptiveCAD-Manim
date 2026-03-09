"""Photon Sphere & Light Bending — Schwarzschild Null Geodesics.

In Schwarzschild spacetime the *photon sphere* sits at r = 1.5 rₛ
where photons travel on unstable circular orbits.  The critical
impact parameter is b_crit = (3√3/2) rₛ.

Rays with b > b_crit are deflected and escape.
Rays with b = b_crit spiral at r = 1.5 rₛ (unstable).
Rays with b < b_crit plunge through the horizon.

The effective potential for null geodesics is:

    V_eff(r) = (1 − rₛ/r) L²/r²

This animation traces many light rays at different impact parameters
around a black hole, showing deflection, capture, and the photon
sphere.

Acts
----
1. Title card with photon-sphere equation
2. Draw BH (dark disk + horizon ring) and photon-sphere ring
3. Trace rays that pass (large b) — mild deflection
4. Trace rays near critical b — strong bending, winding
5. Trace captured rays (b < b_crit) — spiral into BH
6. Camera tilt to show 3-D photon sphere shell
7. Summary card

Run
---
    manim -pql examples/photon_sphere.py PhotonSphere
    manim -qh  examples/photon_sphere.py PhotonSphere
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Dot,
    Dot3D,
    Circle,
    VMobject,
    VGroup,
    Surface,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    SurroundingRectangle,
    PI,
    TAU,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    YELLOW,
    RED,
    ORANGE,
    GOLD,
    TEAL,
    GREY_A,
    GREY_D,
    BLUE,
    BLUE_D,
    BLUE_E,
    GREEN,
    PURPLE,
    interpolate_color,
)

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Physics constants (geometric units, rₛ = 1)
# ═══════════════════════════════════════════════════════════════════════════
RS = 1.0                           # Schwarzschild radius
R_PHOTON = 1.5 * RS                # photon sphere radius
B_CRIT = (3 * np.sqrt(3) / 2) * RS  # critical impact parameter ≈ 2.598

# Display scaling — we multiply physical radii by SCALE so the BH
# is big enough on screen.  All ray coordinates use this scale.
SCALE = 1.2
RS_D = RS * SCALE
R_PH_D = R_PHOTON * SCALE
B_CRIT_D = B_CRIT * SCALE


# ═══════════════════════════════════════════════════════════════════════════
# Numerical ray tracer  (Schwarzschild, equatorial plane)
#
# We integrate the orbit equation in (r, φ) using the substitution
# u = 1/r:
#     (du/dφ)² = 1/b² − u²(1 − rₛ u)
#
# which yields:
#     d²u/dφ² = −u + (3/2) rₛ u²
#
# We integrate in φ from the start angle.
# ═══════════════════════════════════════════════════════════════════════════
def _trace_null_geodesic(
    b: float,
    phi_span: float = 3 * PI,
    n_steps: int = 2000,
    r_start: float = 12.0,
    rs: float = RS,
):
    """Trace a null geodesic with impact parameter *b* around a
    Schwarzschild mass.

    Returns arrays (x, y) in scaled display coordinates.
    The ray enters from the right with y-offset = b.
    """
    # u = 1/r,  u' = du/dφ
    u = 1.0 / r_start
    # At large r the ray is nearly straight: du/dφ ≈ −1/b  (minus
    # because r decreases as φ increases for ingoing ray).
    # Sign: ray comes from +x heading in −x direction, so φ increases
    # from 0 (right) counter-clockwise.
    du = -np.sqrt(max(1.0 / b**2 - u**2 * (1 - rs * u), 0))

    dphi = phi_span / n_steps
    xs, ys = [], []
    for _ in range(n_steps):
        r = 1.0 / u if u > 0 else 1e6
        if r < rs:
            break  # captured
        if r > 20:
            # If the ray has turned around and is heading outward past
            # our starting radius, we can stop.
            if len(xs) > 100:
                break
        phi = len(xs) * dphi
        xs.append(r * np.cos(phi) * SCALE)
        ys.append(r * np.sin(phi) * SCALE)

        # Leapfrog / Störmer-Verlet step for the orbit equation
        d2u = -u + 1.5 * rs * u * u
        du += d2u * dphi
        u += du * dphi
        if u <= 0:
            break  # ray escaping to infinity

    return np.array(xs), np.array(ys)


def _ray_vmobject(xs, ys, color, width=1.8):
    """Build a VMobject curve from (xs, ys)."""
    pts = np.column_stack([xs, ys, np.zeros_like(xs)])
    vm = VMobject(color=color, stroke_width=width)
    # Down-sample for performance: keep every 4th point
    pts_ds = pts[::4]
    if len(pts_ds) < 3:
        pts_ds = pts
    vm.set_points_smoothly(pts_ds)
    return vm


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class PhotonSphere(ThreeDScene):
    """Light bending and photon sphere around a Schwarzschild BH."""

    def construct(self):
        self.camera.background_color = "#050510"
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        # ─── Act 1  Title ────────────────────────────────────────────────
        ttl = Text("Photon Sphere  &  Light Bending",
                    font_size=42, color=GOLD)
        sub = Text("Schwarzschild null geodesics",
                    font_size=22, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"r_{\rm ph} = \tfrac{3}{2}\,r_s",
            r"\qquad",
            r"b_{\rm crit} = \tfrac{3\sqrt{3}}{2}\,r_s",
            font_size=24, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Black hole + photon sphere ──────────────────────────
        # Dark disk for the BH interior
        bh_disk = Circle(radius=RS_D, color="#050510",
                         fill_color="#050510", fill_opacity=1,
                         stroke_width=0)
        # Horizon ring
        horizon = Circle(radius=RS_D, color=RED,
                         stroke_width=2, stroke_opacity=0.8)
        # Photon sphere ring (dashed feel via opacity)
        ph_ring = Circle(radius=R_PH_D, color=ORANGE,
                         stroke_width=1.5, stroke_opacity=0.6)
        # Glow
        bh_glow = Circle(radius=RS_D * 1.6, color=RED,
                         fill_color=RED, fill_opacity=0.06,
                         stroke_width=0)

        # Labels
        lbl_rs = MathTex(r"r_s", font_size=16, color=RED)
        lbl_rs.move_to([RS_D + 0.25, -0.25, 0])
        lbl_ph = MathTex(r"r_{\rm ph}=\tfrac{3}{2}r_s",
                         font_size=14, color=ORANGE)
        lbl_ph.move_to([R_PH_D + 0.55, 0.25, 0])

        for lbl in [lbl_rs, lbl_ph]:
            self.add_fixed_in_frame_mobjects(lbl)

        self.play(
            FadeIn(bh_glow), FadeIn(bh_disk), Create(horizon),
            Create(ph_ring),
            FadeIn(lbl_rs), FadeIn(lbl_ph),
            run_time=1.2,
        )
        self.wait(0.3)

        # ─── Act 3  Passing rays (b >> b_crit) — mild deflection ───────
        pass_lbl = Text("Deflected rays  (b > b_crit)", font_size=16,
                        color=YELLOW)
        pass_lbl.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(pass_lbl)
        self.play(FadeIn(pass_lbl), run_time=0.4)

        b_pass = [4.0, 5.0, 6.5, 8.0, 10.0]
        pass_rays = VGroup()
        for b in b_pass:
            xs, ys = _trace_null_geodesic(b)
            col = interpolate_color(YELLOW, WHITE,
                                    (b - 4) / 6)
            ray = _ray_vmobject(xs, ys, color=col, width=1.4)
            pass_rays.add(ray)
            # Mirror ray below axis
            ray_m = _ray_vmobject(xs, -ys, color=col, width=1.4)
            pass_rays.add(ray_m)

        self.play(Create(pass_rays, lag_ratio=0.05), run_time=2.5)
        self.wait(0.5)

        # ─── Act 4  Near-critical rays — strong bending ─────────────────
        self.play(FadeOut(pass_lbl), run_time=0.3)
        crit_lbl = Text("Near-critical rays  (b ≈ b_crit)",
                        font_size=16, color=ORANGE)
        crit_lbl.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(crit_lbl)
        self.play(FadeIn(crit_lbl), run_time=0.4)

        # Rays very close to b_crit wind around the BH
        b_near = [B_CRIT * 1.02, B_CRIT * 1.05, B_CRIT * 1.10,
                  B_CRIT * 1.20]
        near_rays = VGroup()
        for b in b_near:
            xs, ys = _trace_null_geodesic(b, phi_span=4 * PI)
            frac = (b / B_CRIT - 1.0) / 0.2
            col = interpolate_color(ORANGE, YELLOW,
                                    min(frac, 1.0))
            ray = _ray_vmobject(xs, ys, color=col, width=2.0)
            near_rays.add(ray)
            ray_m = _ray_vmobject(xs, -ys, color=col, width=2.0)
            near_rays.add(ray_m)

        self.play(Create(near_rays, lag_ratio=0.08), run_time=3)
        self.wait(0.5)

        # ─── Act 5  Captured rays (b < b_crit) — spiral in ─────────────
        self.play(FadeOut(crit_lbl), run_time=0.3)
        cap_lbl = Text("Captured rays  (b < b_crit)", font_size=16,
                        color=RED)
        cap_lbl.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(cap_lbl)
        self.play(FadeIn(cap_lbl), run_time=0.4)

        b_cap = [B_CRIT * 0.98, B_CRIT * 0.90, B_CRIT * 0.70,
                 B_CRIT * 0.50]
        cap_rays = VGroup()
        for b in b_cap:
            xs, ys = _trace_null_geodesic(b, phi_span=4 * PI)
            frac = (B_CRIT - b) / (B_CRIT * 0.5)
            col = interpolate_color(ORANGE, RED, min(frac, 1.0))
            ray = _ray_vmobject(xs, ys, color=col, width=1.6)
            cap_rays.add(ray)
            ray_m = _ray_vmobject(xs, -ys, color=col, width=1.6)
            cap_rays.add(ray_m)

        self.play(Create(cap_rays, lag_ratio=0.08), run_time=2.5)
        self.wait(0.5)

        # ─── Act 6  Camera tilt — 3-D photon-sphere shell ──────────────
        self.play(FadeOut(cap_lbl), run_time=0.3)
        shell_lbl = Text("Photon sphere  (3-D shell)", font_size=16,
                         color=ORANGE)
        shell_lbl.to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(shell_lbl)
        self.play(FadeIn(shell_lbl), run_time=0.4)

        # Translucent sphere at the photon radius
        ph_sphere = Surface(
            lambda u, v: np.array([
                R_PH_D * np.sin(u) * np.cos(v),
                R_PH_D * np.sin(u) * np.sin(v),
                R_PH_D * np.cos(u),
            ]),
            u_range=[0, PI], v_range=[0, TAU],
            resolution=(16, 24),
            fill_color=ORANGE, fill_opacity=0.08,
            stroke_width=0.3, stroke_color=ORANGE,
            stroke_opacity=0.15,
        )

        # Small dark sphere for BH
        bh_sphere = Surface(
            lambda u, v: np.array([
                RS_D * np.sin(u) * np.cos(v),
                RS_D * np.sin(u) * np.sin(v),
                RS_D * np.cos(u),
            ]),
            u_range=[0, PI], v_range=[0, TAU],
            resolution=(12, 18),
            fill_color="#050510", fill_opacity=1.0,
            stroke_width=0.4, stroke_color=RED,
            stroke_opacity=0.25,
        )

        self.play(FadeIn(ph_sphere), FadeIn(bh_sphere), run_time=1)

        # Camera orbit
        self.move_camera(phi=65 * DEG, theta=-50 * DEG, run_time=3)
        self.wait(0.5)
        self.move_camera(phi=35 * DEG, theta=-90 * DEG, run_time=2)
        self.wait(0.5)

        # ─── Act 7  Summary ──────────────────────────────────────────────
        self.move_camera(phi=0, theta=-PI / 2, run_time=1.5)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            *[FadeOut(lbl) for lbl in [lbl_rs, lbl_ph, shell_lbl]],
            run_time=0.8,
        )

        card = Text("Photon Sphere", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"r_{\rm ph}", r"=", r"\tfrac{3}{2}\,r_s",
                r"\quad\text{(unstable circular photon orbit)}",
                font_size=19,
            ),
            MathTex(
                r"b_{\rm crit}", r"=",
                r"\tfrac{3\sqrt{3}}{2}\,r_s",
                r"\;\approx 2.60\,r_s",
                font_size=19, color=ORANGE,
            ),
            MathTex(
                r"b > b_{\rm crit}:",
                r"\;\text{ray deflected and escapes}",
                font_size=18, color=YELLOW,
            ),
            MathTex(
                r"b < b_{\rm crit}:",
                r"\;\text{ray captured by BH}",
                font_size=18, color=RED,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        bullets.next_to(card, DOWN, buff=0.35)
        box = SurroundingRectangle(
            VGroup(card, bullets), color=GOLD, buff=0.25,
            corner_radius=0.1,
        )
        self.add_fixed_in_frame_mobjects(card, box, *bullets)
        self.play(Write(card), FadeIn(box), run_time=0.8)
        for b in bullets:
            self.play(FadeIn(b), run_time=0.5)
        self.wait(2)
        self.play(
            FadeOut(card), FadeOut(box),
            *[FadeOut(b) for b in bullets],
            run_time=1,
        )
