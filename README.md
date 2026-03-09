# CADManim — AdaptiveCAD × Manim

Animate AdaptiveCAD geometry with Manim's powerful rendering engine.

## What it does

| AdaptiveCAD provides | Manim provides | CADManim bridges them |
|---|---|---|
| SDF shapes (sphere, torus, gyroid…) | Scene/camera/lighting | `SDFSurface`, `AdaptiveCADMesh` |
| Bézier / B-spline curves | Animations (Create, Transform…) | `BezierCurveMobject`, `AnimateCurveGrowth` |
| 2-D sketch entities | Text, LaTeX, labels | `SketchMobject` |
| G-code / toolpaths | Video/GIF export | `ToolpathMobject`, `AnimateToolpath` |
| π_a adaptive geometry | 3-D camera orbits | `PiAdaptiveSurface` scene |

## Quick start

```bash
pip install -r requirements.txt
# Render the SDF shapes demo
manim -pql examples/sdf_shapes_demo.py SDFShapesDemo
# Render curves & toolpaths
manim -pql examples/curves_and_toolpaths.py CurvesAndToolpaths
```

## Package structure

```
cadmanim/
├── __init__.py          # public API
├── mobjects.py          # Manim Mobjects wrapping AdaptiveCAD geometry
├── animations.py        # Custom animations (exploded view, morph, toolpath…)
└── utils.py             # Conversion helpers (SDF→mesh, Vec3↔array)
examples/
├── sdf_shapes_demo.py   # SDF meshing, morphing sphere→torus, gyroid, exploded view
└── curves_and_toolpaths.py  # Bézier growth, CNC toolpath trace, assembly
```

## Custom Mobjects

### `SDFSurface(sdf_func, bounds, resolution)`
Samples an SDF on a grid, runs marching cubes, and renders as a Manim Surface.

### `AdaptiveCADMesh(vertices, faces)`
Renders a triangle mesh as a `VGroup` of polygons. Classmethods:
- `.from_stl(path)` — load an STL file through AdaptiveCAD
- `.from_sdf(func)` — marching-cubes extraction in one call

### `BezierCurveMobject(curve)`
Evaluates an AdaptiveCAD `BezierCurve` and renders it as a smooth Manim path.

### `SketchMobject(sketch_doc)`
Converts AdaptiveCAD `SketchDocument` entities (lines, arcs, circles) into 2D Manim objects.

### `ToolpathMobject(points)`
Colour-gradient path for G-code / CNC visualisation.

## Custom Animations

| Animation | Purpose |
|---|---|
| `AnimateExplodedView(group)` | Push parts outward from centre |
| `AnimateAssembly(group)` | Snap scattered parts to assembled positions |
| `AnimateToolpath(path)` | Progressive reveal of a CNC/3D-print path |
| `AnimateCurveGrowth(curve)` | Draw a parametric curve from t=0→1 |
| `MorphBetweenSDFs(mob, sdf_a, sdf_b)` | Geometrically correct SDF blending |

## Tips

- Use `-pql` for quick 480p preview, `-pqh` for 1080p.
- For 3D scenes, inherit from `ThreeDScene` and call `set_camera_orientation()`.
- `MorphBetweenSDFs` is compute-heavy; lower `resolution` and `keyframes` for faster previews.
- If AdaptiveCAD is not installed, the examples fall back to pure-Manim stand-ins.

---

## EGATL Benchmark Suite

This repo also contains the **Adaptive Chern Self-Healing Conductance Law (EGATL)** benchmark harness and verified solver modules. See [SUBMISSION.md](SUBMISSION.md) for the full writeup.

### Quick run

```bash
pip install -e .                               # install arp-topology package
python benchmarks/run_full_pipeline.py         # run all 4 stages → outputs/manifest.json
```

### Benchmark stages

| Stage | Script | Output |
|-------|--------|--------|
| Recovery demo | `benchmarks/run_recovery_demo.py` | 4-variant damage/recovery traces |
| Matched-present | `benchmarks/run_matched_present.py` | Ablation from shared snapshot |
| Solver verification | `benchmarks/run_solver_verification.py` | 5/5 PASS verdict |
| Onset maps | `benchmarks/run_onset_map.py` | 2D parameter sweep heatmaps |

### Solver modules

| Module | Model |
|--------|-------|
| `solver/egatl.py` | Block QWZ (2×2 Nambu) |
| `solver/scalar_flux.py` | 2D Harper-Hofstadter flux lattice |
| `solver/ssh.py` | 1D SSH chain |
| `solver/kitaev.py` | Kitaev p-wave chain |
| `solver/haldane.py` | Haldane honeycomb lattice |
| `solver/rice_mele.py` | Rice-Mele chain (Zak phase) |
