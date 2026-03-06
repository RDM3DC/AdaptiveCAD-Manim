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
