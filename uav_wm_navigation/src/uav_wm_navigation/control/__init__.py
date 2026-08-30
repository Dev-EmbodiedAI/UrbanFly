"""Control components.

Imports are lazy so the lightweight safety/runtime path does not require the
recording stack (h5py) to be binary-compatible on a deployment machine.
"""

from importlib import import_module

_EXPORTS = {
    "ned_to_nwu": (".coordinate_transform", "ned_to_nwu"),
    "nwu_to_ned": (".coordinate_transform", "nwu_to_ned"),
    "RiskReranker": (".reranker", "RiskReranker"),
    "CandidateRerankerV3": (".reranker", "CandidateRerankerV3"),
    "PolylineRoute": (".route_manager", "PolylineRoute"),
    "RouteManager": (".route_manager", "RouteManager"),
    "RouteProjection": (".route_manager", "RouteProjection"),
    "rank_route_consistent_candidates": (
        ".route_guard",
        "rank_route_consistent_candidates",
    ),
    "SafetyFilter": (".safety_filter", "SafetyFilter"),
    "TransparentSafetyLayer": (".transparent_safety", "TransparentSafetyLayer"),
    "TrajectoryExecutor": (".trajectory_executor", "TrajectoryExecutor"),
    "LatestValue": (".realtime_yopo", "LatestValue"),
    "RealtimeYOPORunner": (".realtime_yopo", "RealtimeYOPORunner"),
    "WAMMPCController": (".wam_mpc_controller", "WAMMPCController"),
    "WAMMPCDecision": (".wam_mpc_controller", "WAMMPCDecision"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
