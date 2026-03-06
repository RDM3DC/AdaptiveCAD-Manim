"""CADManim — triangle-free bridge between AdaptiveCAD geometry and Manim.

All 3D shapes use analytic parametric surfaces or SDF contour slicing,
aligned with AdaptiveCAD's philosophy of implicit/analytic geometry.
No marching cubes, no triangle meshes.
"""

from .mobjects import (
    SDFSurface,
    SDFContourStack,
    BezierCurveMobject,
    SketchMobject,
    ToolpathMobject,
)
from .animations import (
    AnimateAssembly,
    AnimateCurveGrowth,
    AnimateExplodedView,
    AnimateToolpath,
    MorphBetweenSDFs,
)

__all__ = [
    # Mobjects
    "SDFSurface",
    "SDFContourStack",
    "BezierCurveMobject",
    "SketchMobject",
    "ToolpathMobject",
    # Animations
    "AnimateAssembly",
    "AnimateExplodedView",
    "AnimateToolpath",
    "AnimateCurveGrowth",
    "MorphBetweenSDFs",
]
