"""Binary Inspiral Gravitational Waves — +/× Strains on a Test Ring.

Two compact objects spiral inward emitting GWs.  The two polarisations
deform a ring of free test particles:

    h₊ : stretch along x, squeeze along y  (and vice versa)
    h×  : stretch along 45°, squeeze along 135°

The chirp signal has increasing frequency and amplitude:
    Φ(t) ∝ (t_c − t)^{5/8}   →   f_GW ∝ (t_c − t)^{−3/8}

Acts
----
1. Title card with strain equations
2. Draw ring of test particles + central binary schematic
3. Animate h₊ deformation (chirp: amplitude & frequency ramp)
4. Switch to h× deformation
5. Show combined +/× oscillation
6. Summary card

Run
---
    manim -pql examples/binary_gw.py BinaryGW
    manim -qh  examples/binary_gw.py BinaryGW
"""

from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene,
    Dot,
    VMobject,
    VGroup,
    MathTex,
    Text,
    Write,
    FadeIn,
    FadeOut,
    Create,
    SurroundingRectangle,
    ValueTracker,
    PI,
    TAU,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    YELLOW,
    ORANGE,
    GOLD,
    TEAL,
    GREY_D,
    PURPLE,
)

DEG = PI / 180

# ═══════════════════════════════════════════════════════════════════════════
# Physics
# ═══════════════════════════════════════════════════════════════════════════
RING_R = 2.0         # unperturbed ring radius
N_PARTICLES = 24     # test particles in ring
CHIRP_CYCLES = 6     # total GW cycles in the chirp
H_MAX = 0.25         # max strain amplitude (exaggerated for visibility)


def _chirp_phase(t_frac):
    s = max(1 - t_frac, 1e-4)
    return CHIRP_CYCLES * TAU * (1.0 - s ** (5 / 8))


def _chirp_amplitude(t_frac):
    s = max(1 - t_frac, 0.02)
    return H_MAX * (0.15 / s) ** 0.25


ANGLES = np.linspace(0, TAU, N_PARTICLES, endpoint=False)


def _update_ring_plus(ring, h):
    for dot, ang in zip(ring, ANGLES):
        r = RING_R * (1 + 0.5 * h * np.cos(2 * ang))
        dot.move_to([r * np.cos(ang), r * np.sin(ang), 0])


def _update_ring_cross(ring, h):
    for dot, ang in zip(ring, ANGLES):
        r = RING_R * (1 + 0.5 * h * np.sin(2 * ang))
        dot.move_to([r * np.cos(ang), r * np.sin(ang), 0])


def _update_ring_combined(ring, hp, hx):
    for dot, ang in zip(ring, ANGLES):
        r = RING_R * (1 + 0.5 * hp * np.cos(2 * ang)
                        + 0.5 * hx * np.sin(2 * ang))
        dot.move_to([r * np.cos(ang), r * np.sin(ang), 0])


def _update_outline(outline, strain_func, h):
    pts = np.array([strain_func(a, h) for a in
                    np.linspace(0, TAU, 100)])
    pts_3d = np.column_stack([pts[:, :2],
                               np.zeros(len(pts))])
    outline.set_points_smoothly(pts_3d)


def _h_plus_pt(angle, h):
    r = RING_R * (1 + 0.5 * h * np.cos(2 * angle))
    return np.array([r * np.cos(angle), r * np.sin(angle), 0.0])


def _h_cross_pt(angle, h):
    r = RING_R * (1 + 0.5 * h * np.sin(2 * angle))
    return np.array([r * np.cos(angle), r * np.sin(angle), 0.0])


def _make_ring(color=TEAL):
    return VGroup(*[
        Dot(point=[RING_R * np.cos(a), RING_R * np.sin(a), 0],
            radius=0.06, color=color)
        for a in ANGLES
    ])


def _make_outline(color=GREY_D):
    ol = VMobject(color=color, stroke_width=1.5)
    pts = np.array([[RING_R * np.cos(a), RING_R * np.sin(a), 0]
                    for a in np.linspace(0, TAU, 100)])
    ol.set_points_smoothly(pts)
    return ol


def _make_binary(phase=0, sep=0.4):
    x1 = sep * np.cos(phase)
    y1 = sep * np.sin(phase)
    d1 = Dot(point=[x1, y1, 0], radius=0.08, color=ORANGE)
    d2 = Dot(point=[-x1, -y1, 0], radius=0.08, color=ORANGE)
    return VGroup(d1, d2)


class BinaryGW(ThreeDScene):
    """Binary inspiral GWs deforming a test-particle ring."""

    def construct(self):
        self.camera.background_color = "#050510"
        self.set_camera_orientation(phi=0, theta=-PI / 2)

        # ─── Act 1  Title ────────────────────────────────────────────────
        ttl = Text("Binary Inspiral  —  Gravitational Waves",
                    font_size=40, color=GOLD)
        sub = Text("+/× polarisations deforming a test ring",
                    font_size=20, color=TEAL)
        sub.next_to(ttl, DOWN, buff=0.3)
        eq = MathTex(
            r"h_+", r"=", r"A\,(1+\cos^2\!\iota)\cos\Phi",
            r"\quad",
            r"h_\times", r"=", r"2A\cos\iota\,\sin\Phi",
            font_size=20, color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(ttl, sub, eq)
        self.play(Write(ttl), run_time=1)
        self.play(FadeIn(sub), run_time=0.5)
        self.play(Write(eq), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(ttl), FadeOut(sub), FadeOut(eq))

        # ─── Act 2  Unperturbed ring + binary ───────────────────────────
        ring = _make_ring(TEAL)
        outline = _make_outline(GREY_D)
        binary = _make_binary(0)
        tracker = ValueTracker(0)

        mode_lbl = MathTex(r"h_+\;\text{mode}", font_size=20,
                           color=YELLOW)
        mode_lbl.to_corner(UP + LEFT, buff=0.4)
        self.add_fixed_in_frame_mobjects(mode_lbl)

        self.play(
            Create(outline), FadeIn(ring), FadeIn(binary),
            FadeIn(mode_lbl),
            run_time=1,
        )
        self.wait(0.3)

        # ─── Act 3  h₊ chirp (single animation via updaters) ───────────
        def _updater_plus(mob, dt=None):
            t = tracker.get_value()
            phi = _chirp_phase(t)
            amp = _chirp_amplitude(t)
            h = amp * np.cos(phi)
            _update_ring_plus(ring, h)
            _update_outline(outline, _h_plus_pt, h)
            sep = 0.4 * max(1 - t, 0.1)
            bp = phi * 0.5
            binary[0].move_to([sep * np.cos(bp), sep * np.sin(bp), 0])
            binary[1].move_to([-sep * np.cos(bp), -sep * np.sin(bp), 0])

        ring.add_updater(_updater_plus)
        self.play(tracker.animate.set_value(1.0), run_time=6,
                  rate_func=lambda t: t)
        ring.remove_updater(_updater_plus)
        self.wait(0.3)

        # ─── Act 4  Switch to h× ────────────────────────────────────────
        new_mode_lbl = MathTex(r"h_\times\;\text{mode}",
                                font_size=20, color=PURPLE)
        new_mode_lbl.to_corner(UP + LEFT, buff=0.4)
        self.add_fixed_in_frame_mobjects(new_mode_lbl)

        # Reset ring and tracker
        _update_ring_cross(ring, 0)
        _update_outline(outline, _h_cross_pt, 0)
        for d in ring:
            d.set_color(PURPLE)
        outline.set_color(GREY_D)
        tracker.set_value(0)

        self.play(FadeIn(new_mode_lbl), FadeOut(mode_lbl), run_time=0.4)
        mode_lbl = new_mode_lbl

        def _updater_cross(mob, dt=None):
            t = tracker.get_value()
            phi = _chirp_phase(t)
            amp = _chirp_amplitude(t)
            h = amp * np.sin(phi)
            _update_ring_cross(ring, h)
            _update_outline(outline, _h_cross_pt, h)
            sep = 0.4 * max(1 - t, 0.1)
            bp = phi * 0.5
            binary[0].move_to([sep * np.cos(bp), sep * np.sin(bp), 0])
            binary[1].move_to([-sep * np.cos(bp), -sep * np.sin(bp), 0])

        ring.add_updater(_updater_cross)
        self.play(tracker.animate.set_value(1.0), run_time=6,
                  rate_func=lambda t: t)
        ring.remove_updater(_updater_cross)
        self.wait(0.3)

        # ─── Act 5  Combined ────────────────────────────────────────────
        comb_lbl = MathTex(r"h_+ + h_\times", font_size=20,
                           color=GOLD)
        comb_lbl.to_corner(UP + LEFT, buff=0.4)
        self.add_fixed_in_frame_mobjects(comb_lbl)
        self.play(FadeOut(mode_lbl), FadeIn(comb_lbl), run_time=0.3)

        for d in ring:
            d.set_color(GOLD)
        tracker.set_value(0)

        def _updater_combined(mob, dt=None):
            t = tracker.get_value()
            phi = _chirp_phase(t)
            amp = _chirp_amplitude(t)
            hp = amp * np.cos(phi)
            hx = amp * 0.6 * np.sin(phi)
            _update_ring_combined(ring, hp, hx)
            # outline
            def _comb_pt(a, _):
                r = RING_R * (1 + 0.5 * hp * np.cos(2 * a)
                                + 0.5 * hx * np.sin(2 * a))
                return np.array([r * np.cos(a), r * np.sin(a), 0])
            _update_outline(outline, _comb_pt, 0)
            outline.set_color(GOLD)
            sep = 0.4 * max(1 - t, 0.1)
            bp = phi * 0.5
            binary[0].move_to([sep * np.cos(bp), sep * np.sin(bp), 0])
            binary[1].move_to([-sep * np.cos(bp), -sep * np.sin(bp), 0])

        ring.add_updater(_updater_combined)
        self.play(tracker.animate.set_value(1.0), run_time=4,
                  rate_func=lambda t: t)
        ring.remove_updater(_updater_combined)
        self.wait(0.5)

        # ─── Act 6  Summary ──────────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            FadeOut(comb_lbl),
            run_time=0.8,
        )

        card = Text("Gravitational Waves", font_size=36, color=GOLD)
        card.to_edge(UP, buff=0.6)
        bullets = VGroup(
            MathTex(
                r"h_+:", r"\;\text{stretch } x,\;\text{squeeze } y",
                font_size=19,
            ),
            MathTex(
                r"h_\times:", r"\;\text{rotated } 45^\circ"
                r"\;\text{from } h_+",
                font_size=19, color=PURPLE,
            ),
            MathTex(
                r"f_{\rm GW}", r"\propto (t_c - t)^{-3/8}",
                font_size=18, color=ORANGE,
            ),
            MathTex(
                r"\text{Coalescence}", r"\to", r"\text{merger}",
                r"\to", r"\text{ringdown}",
                font_size=18, color=YELLOW,
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
