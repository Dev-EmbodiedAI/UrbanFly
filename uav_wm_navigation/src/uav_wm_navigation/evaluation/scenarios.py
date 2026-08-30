from __future__ import annotations

SCENARIOS = (
    "StreetCanyon", "IntersectionTurn", "OccludedCrossing", "DynamicCrossing", "NarrowPassage", "DenseMixedUrban"
)


def validate_scenario(name: str) -> str:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose one of {SCENARIOS}")
    return name
