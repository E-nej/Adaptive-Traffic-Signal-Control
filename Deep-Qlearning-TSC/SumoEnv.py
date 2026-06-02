# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:20:28 2019

@author: Ritu Pande
"""
import os
import numpy as np
from helpers import get_lane_cell, get_edge_waiting_time  # also runs SUMO path setup
import traci  # type: ignore


class SumoEnv:
    INCOMING_ROADS = ["E2TL", "N2TL", "W2TL", "S2TL"]

    # Shared LANE_CELL_BOUNDS and get_lane_cell() live in helpers.py

    LANE_GROUP_MAP = {
        "W2TL_0": 0, "W2TL_1": 0, "W2TL_2": 0,
        "W2TL_3": 1,
        "N2TL_0": 2, "N2TL_1": 2, "N2TL_2": 2,
        "N2TL_3": 3,
        "E2TL_0": 4, "E2TL_1": 4, "E2TL_2": 4,
        "E2TL_3": 5,
        "S2TL_0": 6, "S2TL_1": 6, "S2TL_2": 6,
        "S2TL_3": 7,
    }

    def __init__(self, sumoBinary, max_steps):
        os.makedirs("results", exist_ok=True)
        self.sumoCmd = [sumoBinary, "-c", "intersection/ts.4L.sumocfg", "--no-step-log", "true",
                        "--waiting-time-memory", str(max_steps), "--log", "results/logfile.txt"]
        self.SUMO_INT_LANE_LENGTH = 500
        self.num_states = 88  # 80 binary position cells + 8 normalised scalar features
        self.max_steps = max_steps
        self._init()

    def _init(self):
        self.current_state = None
        self.curr_wait_time = 0
        self.steps = 0
        self.last_queue_sum = 0
        self.last_delay_sum = 0
        self.last_stops_sum = 0
        self.last_co2_sum = 0
        self.last_speed_sum = 0
        self.last_vehicle_count_sum = 0
        # Per-step cache: holds values from the most recent simulation step,
        # used in _encode_env_state to avoid redundant TraCI queries.
        self._step_halting = 0
        self._step_count = 0
        self._step_speed = 0.0
        self._step_co2 = 0.0

    def get_state(self):
        return self.current_state

    def start(self):
        try:
            traci.start(self.sumoCmd)
            self.current_state = self._encode_env_state()
            return self.current_state
        except Exception as e:
            print(f"SUMO start FAILED: {type(e).__name__}: {e}")
            raise

    def reset(self):
        try:
            traci.close()
            traci.start(self.sumoCmd)
            self._init()
            self.current_state = self._encode_env_state()
            return self.current_state
        except Exception as e:
            print(f"SUMO reset FAILED: {type(e).__name__}: {e}")
            raise

    def step(self, num_steps=1):
        if self.steps + num_steps > self.max_steps:
            num_steps = self.max_steps - self.steps

        old_wait_time = self.curr_wait_time  # saved before loop for reward calc

        queue_sum = 0
        co2_sum = 0
        speed_sum = 0
        vehicle_count_sum = 0

        for _ in range(num_steps):
            traci.simulationStep()

            halting = self._get_halting_count()
            queue_sum += halting

            co2 = self._get_co2()
            co2_sum += co2
            speed = self._get_average_speed()
            speed_sum += speed
            count = self._get_vehicle_count()
            vehicle_count_sum += count

            # Update cache so _encode_env_state doesn't re-query TraCI
            self._step_halting = halting
            self._step_co2 = co2
            self._step_speed = speed
            self._step_count = count

        new_wait_time = self._get_waiting_time()

        self.last_queue_sum = queue_sum
        self.last_delay_sum = new_wait_time - old_wait_time
        self.last_stops_sum = queue_sum
        self.last_co2_sum = co2_sum
        self.last_speed_sum = speed_sum
        self.last_vehicle_count_sum = vehicle_count_sum

        self.steps += num_steps
        self.current_state = self._encode_env_state()
        self.curr_wait_time = new_wait_time

        # Original reward: reduction in cumulative waiting time across incoming lanes
        reward = 0.9 * old_wait_time - new_wait_time

        is_terminal = (
            self.steps >= self.max_steps
            or traci.simulation.getMinExpectedNumber() == 0
        )

        return (reward, self.current_state, is_terminal)

    def _get_waiting_time(self):
        return get_edge_waiting_time(self.INCOMING_ROADS)

    def _get_halting_count(self):
        return sum(traci.edge.getLastStepHaltingNumber(road) for road in self.INCOMING_ROADS)

    def _get_vehicle_count(self):
        return sum(traci.edge.getLastStepVehicleNumber(r) for r in self.INCOMING_ROADS)

    def _get_average_speed(self):
        speeds = [traci.edge.getLastStepMeanSpeed(r) for r in self.INCOMING_ROADS]
        active = [s for s in speeds if s > 0]
        return sum(active) / len(active) if active else 0.0

    def _get_co2(self):
        return sum(traci.edge.getCO2Emission(r) for r in self.INCOMING_ROADS)

    def _encode_env_state(self):
        state = np.zeros(80)

        for veh_id in traci.vehicle.getIDList():
            lane_pos = self.SUMO_INT_LANE_LENGTH - traci.vehicle.getLanePosition(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)

            lane_group = self.LANE_GROUP_MAP.get(lane_id, -1)
            if lane_group == -1:
                continue

            lane_cell = get_lane_cell(lane_pos)
            veh_position = lane_group * 10 + lane_cell
            state[veh_position] = 1

        extra = np.array([
            self._step_halting / 500,
            self._step_count / 500,
            self._step_speed / 25,
            traci.trafficlight.getPhase("TL") / 7,
            self._step_halting / 500,
            self._step_co2 / 100000,
            self.steps / self.max_steps,
            traci.simulation.getMinExpectedNumber() / 500,
        ])
        return np.concatenate([state, extra])

    def __del__(self):
        try:
            traci.close()
        except Exception:
            pass
