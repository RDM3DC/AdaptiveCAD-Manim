"""Rubin Observatory — When the bottleneck shifts from photons to attention.

For 400 years we built telescopes to see further.
On Feb 24 2025, VRubinObs produced 800,000 alerts in a single night.
Expected: 7 million alerts/night when fully operational.

Scene:
    RubinAlertFlood — The firehose era of astronomy

Run:
    manim -pqh examples/rubin_alert_flood.py RubinAlertFlood
"""

from __future__ import annotations

import math
import numpy as np
from manim import (
    Scene, VGroup, VMobject,
    Axes, DashedLine,
    Text, MathTex, DecimalNumber,
    Rectangle, RoundedRectangle, Line, Arrow, Dot, Circle,
    Annulus, AnnularSector, Arc, Sector,
    FadeIn, FadeOut, Create, Write, Transform,
    ReplacementTransform, AnimationGroup,
    Indicate, Flash, GrowFromCenter,
    ShrinkToCenter, ApplyMethod,
    LEFT, RIGHT, UP, DOWN, ORIGIN, UL, UR, DL, DR,
    BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    RED, RED_A, RED_B, RED_C, RED_D, RED_E,
    GREEN, GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E,
    YELLOW, WHITE, GREY, ORANGE, PURPLE, PINK,
    GOLD, GOLD_A, GOLD_B, GOLD_C, GOLD_E,
    TEAL, TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E,
    MAROON, MAROON_A, MAROON_B, MAROON_C,
    GREY_A, GREY_B, GREY_C, GREY_D, GREY_E,
    rate_functions, config,
    ValueTracker,
    always_redraw,
    linear,
    smooth,
    there_and_back,
    PI, TAU,
)


# ═══════════════════════════════════════════════════════════════════════════
# Colour palette
# ═══════════════════════════════════════════════════════════════════════════

ALERT_COLORS = {
    "Supernova":     "#ff4444",
    "Asteroid":      "#44aaff",
    "Variable Star": "#ffaa00",
    "AGN Flare":     "#cc44ff",
    "Microlensing":  "#44ffaa",
    "Unknown":       "#888888",
}

DEEP_BG   = "#050510"
PANEL_BG  = "#0a0a1e"
RUBIN_TEAL = "#00b4d8"


# ═══════════════════════════════════════════════════════════════════════════
# Helper: star field
# ═══════════════════════════════════════════════════════════════════════════

def make_starfield(n=200, seed=12345):
    """Scatter faint dots across the frame as a background star field."""
    rng = np.random.default_rng(seed)
    stars = VGroup()
    for _ in range(n):
        x = rng.uniform(-7.2, 7.2)
        y = rng.uniform(-4.1, 4.1)
        r = rng.uniform(0.008, 0.025)
        opacity = rng.uniform(0.15, 0.55)
        star = Dot(
            point=[x, y, 0], radius=r,
            color=WHITE, fill_opacity=opacity,
            stroke_width=0,
        )
        stars.add(star)
    return stars


# ═══════════════════════════════════════════════════════════════════════════
# Helper: timeline telescope entry
# ═══════════════════════════════════════════════════════════════════════════

def telescope_card(year: str, name: str, detail: str, color):
    """Small card for a telescope milestone."""
    bg = RoundedRectangle(
        width=2.6, height=1.1, corner_radius=0.08,
        fill_color=PANEL_BG, fill_opacity=0.9,
        stroke_color=color, stroke_width=1.2,
    )
    yr = Text(year, font_size=20, color=color, weight="BOLD")
    nm = Text(name, font_size=14, color=WHITE)
    dt = Text(detail, font_size=11, color=GREY_A)
    content = VGroup(yr, nm, dt).arrange(DOWN, buff=0.06)
    content.move_to(bg.get_center())
    return VGroup(bg, content)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: alert burst (dots erupting from a point)
# ═══════════════════════════════════════════════════════════════════════════

def make_alert_burst(n, center, spread, seed=0):
    """Create a VGroup of scattered alert dots around *center*."""
    rng = np.random.default_rng(seed)
    cats = list(ALERT_COLORS.keys())
    colors = list(ALERT_COLORS.values())
    dots = VGroup()
    for i in range(n):
        angle = rng.uniform(0, TAU)
        radius = rng.exponential(spread * 0.4)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        # clamp to frame
        x = max(-6.8, min(6.8, x))
        y = max(-3.6, min(3.6, y))
        cat_idx = rng.integers(0, len(cats))
        dot = Dot(
            point=[x, y, 0],
            radius=rng.uniform(0.02, 0.05),
            color=colors[cat_idx],
            fill_opacity=rng.uniform(0.5, 1.0),
            stroke_width=0,
        )
        dots.add(dot)
    return dots


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class RubinAlertFlood(Scene):
    """The bottleneck in astronomy is no longer observation — it's attention."""

    def construct(self):
        # Black background
        self.camera.background_color = DEEP_BG

        # ── Star field ───────────────────────────────────────────────────
        stars = make_starfield(300)
        self.add(stars)

        # ══════════════════════════════════════════════════════════════════
        # ACT 0: Opening Text
        # ══════════════════════════════════════════════════════════════════
        line1 = Text(
            "We built telescopes to see further.",
            font_size=36, color=WHITE,
        )
        line2 = Text(
            "That was the story for 400 years.",
            font_size=28, color=GREY_A,
        )
        line2.next_to(line1, DOWN, buff=0.3)

        self.play(Write(line1), run_time=2)
        self.play(FadeIn(line2), run_time=1)
        self.wait(2)
        self.play(FadeOut(line1), FadeOut(line2))

        # ══════════════════════════════════════════════════════════════════
        # ACT 1: Telescope Timeline — 400 years in 12 seconds
        # ══════════════════════════════════════════════════════════════════
        timeline_label = Text(
            "400 Years of Seeing Further",
            font_size=24, color=GREY_A,
        ).to_edge(UP, buff=0.3)
        self.play(Write(timeline_label))

        # Timeline axis
        tl_line = Line(
            [-6, -0.5, 0], [6, -0.5, 0],
            stroke_width=1.5, color=GREY_D,
        )
        self.play(Create(tl_line), run_time=0.8)

        milestones = [
            ("1609",  "Galileo",     "4 moons of Jupiter",     BLUE_C),
            ("1668",  "Newton",      "Reflecting telescope",    TEAL_C),
            ("1845",  "Rosse",       "Spiral nebulae",          GREEN_C),
            ("1917",  "Hooker 100\"","Island universes",        GOLD_C),
            ("1990",  "Hubble ST",   "Deep field, 10B ly",      ORANGE),
            ("2022",  "JWST",        "First galaxies, z > 13",  RED_C),
        ]

        cards = []
        x_positions = np.linspace(-5.2, 5.2, len(milestones))
        for i, (yr, name, detail, color) in enumerate(milestones):
            card = telescope_card(yr, name, detail, color)
            y_offset = 1.2 if i % 2 == 0 else -2.2
            card.move_to([x_positions[i], y_offset, 0])
            tick = Line(
                [x_positions[i], -0.5, 0],
                [x_positions[i], -0.5 + (0.3 if i % 2 == 0 else -0.3), 0],
                stroke_width=1, color=color,
            )
            cards.append((card, tick))

        for card, tick in cards:
            self.play(
                FadeIn(card, shift=UP * 0.3),
                Create(tick),
                run_time=0.7,
            )

        self.wait(1.5)

        # Insight text
        insight = Text(
            "Each one: bigger mirror → see further",
            font_size=18, color=GREY_A,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(insight))
        self.wait(1)

        # Wipe the timeline
        tl_group = VGroup(
            tl_line, timeline_label, insight,
            *[c for c, t in cards], *[t for c, t in cards],
        )
        self.play(FadeOut(tl_group), run_time=1)

        # ══════════════════════════════════════════════════════════════════
        # ACT 2: Rubin Changes the Story
        # ══════════════════════════════════════════════════════════════════
        rubin_title = Text(
            "Vera C. Rubin Observatory",
            font_size=38, color=RUBIN_TEAL, weight="BOLD",
        )
        rubin_sub = Text(
            "Cerro Pachón, Chile  ·  8.4m mirror  ·  3.2 gigapixel camera",
            font_size=18, color=GREY_A,
        ).next_to(rubin_title, DOWN, buff=0.2)

        self.play(Write(rubin_title), run_time=1.5)
        self.play(FadeIn(rubin_sub))
        self.wait(1)

        date_line = Text(
            "February 24, 2025 — First Light",
            font_size=22, color=GOLD,
        ).next_to(rubin_sub, DOWN, buff=0.4)
        self.play(Write(date_line))
        self.wait(1)

        # "Not an image — a flood"
        not_image = Text(
            "It didn't produce a breathtaking image.",
            font_size=24, color=WHITE,
        ).next_to(date_line, DOWN, buff=0.5)
        self.play(Write(not_image), run_time=1.5)
        self.wait(0.8)

        it_produced = Text(
            "It produced 800,000 alerts — in a single night.",
            font_size=28, color=YELLOW, weight="BOLD",
        ).next_to(not_image, DOWN, buff=0.3)
        self.play(Write(it_produced), run_time=2)
        self.wait(1.5)

        self.play(
            FadeOut(rubin_title), FadeOut(rubin_sub),
            FadeOut(date_line), FadeOut(not_image), FadeOut(it_produced),
        )

        # ══════════════════════════════════════════════════════════════════
        # ACT 3: The Firehose — alert dots flooding the screen
        # ══════════════════════════════════════════════════════════════════

        # Counter at top
        counter_label = Text(
            "Alerts tonight:", font_size=20, color=GREY_A,
        ).to_corner(UL, buff=0.4)

        counter_tracker = ValueTracker(0)
        counter_num = always_redraw(
            lambda: DecimalNumber(
                counter_tracker.get_value(),
                num_decimal_places=0,
                font_size=36,
                color=YELLOW,
            ).next_to(counter_label, RIGHT, buff=0.2)
        )

        # Classification panel (right side)
        panel_bg = RoundedRectangle(
            width=2.8, height=3.5, corner_radius=0.12,
            fill_color=PANEL_BG, fill_opacity=0.9,
            stroke_color=RUBIN_TEAL, stroke_width=1,
        ).to_corner(DR, buff=0.3)

        panel_title = Text(
            "Live Classification", font_size=14, color=RUBIN_TEAL,
        )
        panel_title.next_to(panel_bg.get_top(), DOWN, buff=0.15)

        category_entries = VGroup()
        for i, (cat, col) in enumerate(ALERT_COLORS.items()):
            dot = Dot(color=col, radius=0.04)
            label = Text(cat, font_size=12, color=WHITE)
            row = VGroup(dot, label).arrange(RIGHT, buff=0.1)
            category_entries.add(row)
        category_entries.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        category_entries.next_to(panel_title, DOWN, buff=0.2)

        # Processing time badge
        proc_badge = VGroup(
            Text("Processing latency:", font_size=11, color=GREY_A),
            Text("< 120 seconds", font_size=14, color=GREEN),
        ).arrange(DOWN, buff=0.04)
        proc_badge.next_to(category_entries, DOWN, buff=0.25)

        panel = VGroup(panel_bg, panel_title, category_entries, proc_badge)

        self.play(
            FadeIn(counter_label), FadeIn(counter_num),
            FadeIn(panel),
            run_time=1,
        )

        # Burst 1: 200 dots
        burst1 = make_alert_burst(200, [0, 0, 0], 3.5, seed=100)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(d, run_time=0.4) for d in burst1],
                lag_ratio=0.005,
            ),
            counter_tracker.animate.set_value(50000),
            run_time=2,
        )

        # Burst 2: 300 more
        burst2 = make_alert_burst(300, [-1, 0.5, 0], 4.0, seed=200)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(d, run_time=0.3) for d in burst2],
                lag_ratio=0.003,
            ),
            counter_tracker.animate.set_value(200000),
            run_time=2,
        )

        # Burst 3: 400 more — screen gets crowded
        burst3 = make_alert_burst(400, [1, -0.3, 0], 4.5, seed=300)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(d, run_time=0.2) for d in burst3],
                lag_ratio=0.002,
            ),
            counter_tracker.animate.set_value(500000),
            run_time=2,
        )

        # Final surge to 800,000
        burst4 = make_alert_burst(300, [0, 0, 0], 5.0, seed=400)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(d, run_time=0.15) for d in burst4],
                lag_ratio=0.002,
            ),
            counter_tracker.animate.set_value(800000),
            run_time=2,
        )

        # Flash the counter
        self.play(Indicate(counter_num, color=YELLOW, scale_factor=1.3))
        self.wait(1)

        # ── "And this is just the beginning" ─────────────────────────────
        future_text = Text(
            "Expected at full capacity: 7,000,000 alerts per night",
            font_size=22, color=ORANGE,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(future_text), run_time=1.5)
        self.wait(1.5)

        # Surge counter to 7M with more dots
        burst5 = make_alert_burst(500, [0, 0, 0], 5.5, seed=500)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(d, run_time=0.1) for d in burst5],
                lag_ratio=0.001,
            ),
            counter_tracker.animate(rate_func=smooth).set_value(7000000),
            run_time=3,
        )
        self.play(Indicate(counter_num, color=ORANGE, scale_factor=1.4))
        self.wait(1)

        # Clear the chaos
        all_bursts = VGroup(burst1, burst2, burst3, burst4, burst5)
        self.play(
            FadeOut(all_bursts),
            FadeOut(future_text),
            FadeOut(panel),
            FadeOut(counter_label), FadeOut(counter_num),
            run_time=1.5,
        )

        # ══════════════════════════════════════════════════════════════════
        # ACT 4: The Bottleneck Shift
        # ══════════════════════════════════════════════════════════════════

        # Before: the old bottleneck
        old_title = Text(
            "The Old Bottleneck", font_size=28, color=GREY_A,
        ).shift(UP * 2)

        funnel_wide = Rectangle(
            width=5, height=0.6, fill_color=BLUE_E, fill_opacity=0.5,
            stroke_color=BLUE, stroke_width=1.5,
        ).shift(UP * 0.8)
        funnel_narrow = Rectangle(
            width=1.2, height=0.6, fill_color=BLUE_E, fill_opacity=0.5,
            stroke_color=BLUE, stroke_width=1.5,
        ).shift(DOWN * 0.2)
        funnel_arrow = Arrow(
            funnel_wide.get_bottom(), funnel_narrow.get_top(),
            color=GREY, stroke_width=2,
        )

        lbl_photons = Text("Photons from the sky", font_size=16, color=BLUE)
        lbl_photons.next_to(funnel_wide, UP, buff=0.1)
        lbl_telescope = Text("Telescope aperture", font_size=16, color=YELLOW)
        lbl_telescope.next_to(funnel_narrow, DOWN, buff=0.1)

        solution_old = Text(
            'Solution: build a bigger mirror →',
            font_size=14, color=GREY_A,
        ).shift(DOWN * 1.5)

        self.play(
            Write(old_title),
            FadeIn(funnel_wide), FadeIn(funnel_narrow),
            Create(funnel_arrow),
            Write(lbl_photons), Write(lbl_telescope),
            run_time=1.5,
        )
        self.play(Write(solution_old))
        self.wait(1.5)

        # Transform to new bottleneck
        new_title = Text(
            "The New Bottleneck", font_size=28, color=YELLOW,
        ).shift(UP * 2)

        funnel_wide2 = Rectangle(
            width=5, height=0.6, fill_color="#1a0a2e", fill_opacity=0.5,
            stroke_color=ORANGE, stroke_width=1.5,
        ).shift(UP * 0.8)
        funnel_narrow2 = Rectangle(
            width=1.2, height=0.6, fill_color="#1a0a2e", fill_opacity=0.5,
            stroke_color=RED, stroke_width=1.5,
        ).shift(DOWN * 0.2)
        funnel_arrow2 = Arrow(
            funnel_wide2.get_bottom(), funnel_narrow2.get_top(),
            color=GREY, stroke_width=2,
        )

        lbl_alerts = Text("7 million alerts / night", font_size=16, color=ORANGE)
        lbl_alerts.next_to(funnel_wide2, UP, buff=0.1)
        lbl_attention = Text("Human attention", font_size=16, color=RED)
        lbl_attention.next_to(funnel_narrow2, DOWN, buff=0.1)

        solution_new = Text(
            'Solution: build smarter filters →',
            font_size=14, color=GREY_A,
        ).shift(DOWN * 1.5)

        self.play(
            Transform(old_title, new_title),
            Transform(funnel_wide, funnel_wide2),
            Transform(funnel_narrow, funnel_narrow2),
            Transform(funnel_arrow, funnel_arrow2),
            Transform(lbl_photons, lbl_alerts),
            Transform(lbl_telescope, lbl_attention),
            Transform(solution_old, solution_new),
            run_time=2,
        )
        self.wait(2)

        self.play(
            *[FadeOut(m) for m in [
                old_title, funnel_wide, funnel_narrow, funnel_arrow,
                lbl_photons, lbl_telescope, solution_old,
            ]],
        )

        # ══════════════════════════════════════════════════════════════════
        # ACT 5: The Numbers
        # ══════════════════════════════════════════════════════════════════
        stats_title = Text(
            "Rubin / LSST by the Numbers",
            font_size=28, color=RUBIN_TEAL,
        ).to_edge(UP, buff=0.4)
        self.play(Write(stats_title))

        stats = [
            ("3.2",    "gigapixels",          "Largest digital camera ever built",   BLUE),
            ("20",     "terabytes / night",   "Raw image data per observing night",  TEAL),
            ("800K",   "alerts / night",      "Current (Feb 2025 first light)",      YELLOW),
            ("7M",     "alerts / night",      "Expected at full survey speed",       ORANGE),
            ("< 120s", "latency",             "Alert to global distribution",        GREEN),
            ("37B",    "objects",             "10-year Legacy Survey catalogue",      GOLD),
            ("20B",    "galaxies",            "Mapped over the full southern sky",   PURPLE),
            ("~6M",    "asteroids",           "Solar system objects tracked",         RED),
        ]

        stat_mobjects = VGroup()
        for num, unit, desc, color in stats:
            row = VGroup(
                Text(num, font_size=28, color=color, weight="BOLD"),
                Text(unit, font_size=16, color=WHITE),
                Text(desc, font_size=12, color=GREY_A),
            ).arrange(RIGHT, buff=0.15)
            stat_mobjects.add(row)
        stat_mobjects.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        stat_mobjects.next_to(stats_title, DOWN, buff=0.4)
        stat_mobjects.shift(LEFT * 1.5)

        for row in stat_mobjects:
            self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(2.5)

        self.play(FadeOut(stat_mobjects), FadeOut(stats_title), run_time=1)

        # ══════════════════════════════════════════════════════════════════
        # ACT 6: The Metaphor — Sound Familiar?
        # ══════════════════════════════════════════════════════════════════
        quote1 = Text(
            "The rarest resource in the universe",
            font_size=30, color=WHITE,
        ).shift(UP * 1)
        quote2 = Text(
            "isn't photons.",
            font_size=30, color=GREY_A,
        ).next_to(quote1, DOWN, buff=0.3)

        self.play(Write(quote1), run_time=1.5)
        self.play(Write(quote2), run_time=1)
        self.wait(1)

        quote3 = Text(
            "It's the ability to focus on what matters",
            font_size=32, color=YELLOW, weight="BOLD",
        ).next_to(quote2, DOWN, buff=0.5)
        quote4 = Text(
            "among an incomprehensible flood of signal.",
            font_size=26, color=GOLD,
        ).next_to(quote3, DOWN, buff=0.2)

        self.play(Write(quote3), run_time=2)
        self.play(FadeIn(quote4), run_time=1)
        self.wait(2)

        # The punchline
        self.play(FadeOut(quote1), FadeOut(quote2))

        punchline = Text(
            "That's not just a cosmic problem.",
            font_size=24, color=GREY_A,
        ).shift(UP * 1)
        punchline2 = Text(
            "It's your problem every morning.",
            font_size=28, color=WHITE, weight="BOLD",
        ).next_to(punchline, DOWN, buff=0.3)

        self.play(
            FadeOut(quote3), FadeOut(quote4),
            Write(punchline),
            run_time=1.5,
        )
        self.play(Write(punchline2), run_time=1.5)
        self.wait(3)

        # ── Final fade ───────────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=2)
