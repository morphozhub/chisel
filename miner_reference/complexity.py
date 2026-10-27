"""Deterministic static pre-check for candidates that would stall the CPU renderer.

Observed failure: candidates with ~10k InstancedMesh instances of *transparent* material
pass the validator in 0.4 s but exceed the renderer's 30 s deadline (software rendering
with transparency sorting). A render timeout is an infrastructure error for the contract,
so the prompt stalls. This check runs on the source text only (pure function, replay-safe)
and turns such candidates into a structured rejection that the repair step can fix.
"""

from __future__ import annotations

import json
import re

MAX_TRANSPARENT_INSTANCES = 6000
MAX_TOTAL_INSTANCES = 30000
MAX_SINGLE_GEOMETRY_VERTICES = 400_000

_NUMBER = r"(?:\d+(?:\.\d+)?|[A-Za-z_$][\w$]*)"


def _resolve(token: str, consts: dict[str, float]) -> float | None:
    token = token.strip()
    try:
        return float(token)
    except ValueError:
        return consts.get(token)


def _constants(source: str) -> dict[str, float]:
    consts: dict[str, float] = {}
    for name, value in re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(\d+(?:\.\d+)?)\b", source):
        consts.setdefault(name, float(value))
    return consts


def assess(source: str) -> str | None:
    """Return a human-readable reason when the source is too heavy for the renderer, else None."""
    consts = _constants(source)
    transparent = re.search(r"transparent\s*:\s*true", source) is not None or "opacity" in source
    instances = 0.0
    for args in re.findall(r"new\s+THREE\.InstancedMesh\s*\(([^)]*)\)", source):
        parts = [p for p in args.split(",")]
        if len(parts) >= 3:
            count = _resolve(parts[2].split("(")[0], consts)
            if count is not None:
                instances += count
    # Fallback: large numeric loop bounds feeding setMatrixAt are instance counts too.
    if "setMatrixAt" in source and instances == 0:
        for bound in re.findall(r"for\s*\(\s*(?:let|var|const)?\s*\w+\s*=\s*0\s*;\s*\w+\s*<\s*(\d+)", source):
            instances = max(instances, float(bound))
    if transparent and instances > MAX_TRANSPARENT_INSTANCES:
        return (
            f"too_heavy_for_renderer: about {int(instances)} instanced objects with transparent materials; "
            f"the software renderer cannot sort that many transparent instances within its time limit. "
            f"Keep transparent instances under {MAX_TRANSPARENT_INSTANCES} (merge bubbles/particles into a few "
            f"opaque or lightly textured meshes) and keep total instances under {MAX_TOTAL_INSTANCES}."
        )
    if instances > MAX_TOTAL_INSTANCES:
        return (
            f"too_heavy_for_renderer: about {int(instances)} instanced objects; keep total InstancedMesh "
            f"instances under {MAX_TOTAL_INSTANCES}."
        )
    for name, args in re.findall(r"new\s+THREE\.(Sphere|Cylinder|Cone|Torus|TorusKnot|Lathe|Tube|Plane|Capsule)Geometry\s*\(([^)]*)\)", source):
        nums = [_resolve(p.split("(")[0], consts) for p in args.split(",")]
        nums = [n for n in nums if n is not None]
        segs = {"Sphere": (1, 2), "Cylinder": (3, 4), "Cone": (2, 3), "Torus": (2, 3), "TorusKnot": (2, 3),
                "Lathe": (1, 1), "Tube": (1, 3), "Plane": (2, 3), "Capsule": (2, 3)}[name]
        try:
            a = nums[segs[0]] if segs[0] < len(nums) else 32
            b = nums[segs[1]] if segs[1] < len(nums) else 16
        except IndexError:
            continue
        if a * b > MAX_SINGLE_GEOMETRY_VERTICES:
            return (
                f"too_heavy_for_renderer: {name}Geometry with about {int(a * b)} segment vertices; "
                f"keep segment counts moderate (<=64 x <=64)."
            )
    return None


def rejection_json(reason: str) -> str:
    return json.dumps({"error": reason, "failures": [{"stage": "pre_render", "rule": "TOO_HEAVY_FOR_RENDERER", "detail": reason}]})
