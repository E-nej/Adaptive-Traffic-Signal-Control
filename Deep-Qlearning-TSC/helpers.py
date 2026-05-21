"""Shared utilities used by both single-agent and multi-agent code."""

import bisect
import os
import sys


# ---------------------------------------------------------------------------
# SUMO path bootstrap  (run at import time so traci is importable afterwards)
# ---------------------------------------------------------------------------

def _setup_sumo_path():
    candidates = [
        '/usr/share/sumo/tools',
        'C:\\Program Files (x86)\\Eclipse\\Sumo\\tools',
        'C:\\Program Files\\Sumo\\tools',
    ]
    s_h = os.environ.get('SUMO_HOME')
    if s_h:
        candidates.insert(0, os.path.join(s_h, 'tools'))
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


_setup_sumo_path()

import traci  # type: ignore  # noqa: E402  (must come after path setup)


# ---------------------------------------------------------------------------
# Traffic light phase indices  (match the order in the .net.xml file)
# ---------------------------------------------------------------------------

PHASE_NS_GREEN   = 0
PHASE_NS_YELLOW  = 1
PHASE_NSL_GREEN  = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN   = 4
PHASE_EW_YELLOW  = 5
PHASE_EWL_GREEN  = 6
PHASE_EWL_YELLOW = 7

_GREEN_PHASES = [PHASE_NS_GREEN, PHASE_NSL_GREEN, PHASE_EW_GREEN, PHASE_EWL_GREEN]


def set_green_phase(tl_id: str, action: int) -> None:
    traci.trafficlight.setPhase(tl_id, _GREEN_PHASES[action])


def set_yellow_phase(tl_id: str, old_action: int) -> None:
    traci.trafficlight.setPhase(tl_id, old_action * 2 + 1)


# ---------------------------------------------------------------------------
# Vehicle position → discrete lane cell
# ---------------------------------------------------------------------------

# Upper bounds (exclusive) for each distance cell from the stop line.
# bisect_right maps a lane_pos float to a cell index 0–9.
LANE_CELL_BOUNDS = [7, 14, 21, 28, 40, 60, 100, 200, 350]


def get_lane_cell(lane_pos: float) -> int:
    """Map distance-from-stop-line (m) to a discrete cell index 0-9."""
    return bisect.bisect_right(LANE_CELL_BOUNDS, lane_pos)


# ---------------------------------------------------------------------------
# TraCI helpers
# ---------------------------------------------------------------------------

def get_edge_waiting_time(roads) -> float:
    """Sum of accumulated vehicle waiting times across the given edges.

    Uses edge-level TraCI (one call per edge) instead of iterating all
    vehicles, giving O(|roads|) complexity instead of O(n vehicles).
    """
    return sum(traci.edge.getWaitingTime(road) for road in roads)
