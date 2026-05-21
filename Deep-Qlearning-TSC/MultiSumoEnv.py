import os
import traci
import numpy as np


class MultiSumoEnv:
    """Shared SUMO environment for a multi-agent traffic signal scenario.

    One SUMO process is managed here; each agent controls its own TL.

    Parameters
    ----------
    sumoBinary : str
        'sumo' or 'sumo-gui'
    max_steps : int
        Episode length in simulation steps.
    sumocfg : str
        Path to the .sumocfg file.
    tl_config : dict, optional
        Explicit TL config dict (keyed by TL id). If omitted, pass net_file
        and the config will be built automatically via NetworkParser.
    net_file : str, optional
        Path to the .net.xml file. Used only when tl_config is None.
    """

    def __init__(self, sumoBinary, max_steps, sumocfg,
                 tl_config=None, net_file=None):
        if tl_config is None:
            if net_file is None:
                raise ValueError("Provide either tl_config or net_file")
            from NetworkParser import NetworkParser
            tl_config = NetworkParser(net_file).tl_config

        os.makedirs("results", exist_ok=True)
        self.sumoCmd = [
            sumoBinary, "-c", sumocfg,
            "--no-step-log", "true",
            "--waiting-time-memory", str(max_steps),
            "--log", "results/logfile.txt",
        ]
        self.SUMO_INT_LANE_LENGTH = 500
        self.num_states = 88  # 80 binary + 8 normalised scalar features
        self.max_steps = max_steps
        self.tl_ids = list(tl_config.keys())
        self.tl_config = tl_config

        # reward weights (same as single-agent SumoEnv)
        self.w1 = 0.4      # delay
        self.w2 = 0.2      # stops
        self.w3 = 0.2      # queue
        self.w4 = 0.0001   # CO2 (mg/s → needs small weight)

        self._init()

    def _init(self):
        self.steps = 0
        self.curr_wait_time = {tl_id: 0.0 for tl_id in self.tl_ids}
        # last_metrics exposed so TLAgent can accumulate episode stats
        self.last_metrics = {
            tl_id: {"queue": 0, "delay": 0, "stops": 0, "co2": 0,
                    "speed": 0, "vehicle_count": 0}
            for tl_id in self.tl_ids
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        try:
            traci.start(self.sumoCmd)
            return {tl_id: self._encode_state(tl_id) for tl_id in self.tl_ids}
        except Exception as e:
            print(f"SUMO start FAILED: {type(e).__name__}: {e}")
            raise

    def reset(self):
        try:
            traci.close()
            traci.start(self.sumoCmd)
            self._init()
            return {tl_id: self._encode_state(tl_id) for tl_id in self.tl_ids}
        except Exception as e:
            print(f"SUMO reset FAILED: {type(e).__name__}: {e}")
            raise

    # ------------------------------------------------------------------
    # Simulation step  –  advances time and returns per-TL results
    # ------------------------------------------------------------------

    def step(self, num_steps=1):
        if self.steps + num_steps > self.max_steps:
            num_steps = self.max_steps - self.steps

        # Accumulators per TL
        acc = {
            tl_id: {"queue": 0, "delay": 0, "stops": 0,
                    "co2": 0, "speed": 0, "count": 0}
            for tl_id in self.tl_ids
        }

        for _ in range(num_steps):
            traci.simulationStep()
            for tl_id in self.tl_ids:
                roads = self.tl_config[tl_id]["incoming_roads"]
                acc[tl_id]["queue"] += self._get_queue(roads)

                new_wait = self._get_waiting_time(roads)
                acc[tl_id]["delay"] += new_wait - self.curr_wait_time[tl_id]
                self.curr_wait_time[tl_id] = new_wait

                acc[tl_id]["stops"] += self._get_queue(roads)   # halting = stops
                acc[tl_id]["co2"]   += self._get_co2(roads)
                acc[tl_id]["speed"] += self._get_avg_speed(roads)
                acc[tl_id]["count"] += self._get_vehicle_count(roads)

        self.steps += num_steps

        is_terminal = (
            self.steps >= self.max_steps
            or traci.simulation.getMinExpectedNumber() == 0
        )

        results = {}
        for tl_id in self.tl_ids:
            a = acc[tl_id]
            self.last_metrics[tl_id] = {
                "queue": a["queue"], "delay": a["delay"],
                "stops": a["stops"], "co2":   a["co2"],
                "speed": a["speed"], "vehicle_count": a["count"],
            }
            reward = -(
                self.w1 * a["delay"] +
                self.w2 * a["stops"] +
                self.w3 * a["queue"] +
                self.w4 * a["co2"]
            ) / max(1, num_steps)

            results[tl_id] = (reward, self._encode_state(tl_id))

        return results, is_terminal

    # ------------------------------------------------------------------
    # Per-TL state encoding  (80-bit binary + 8 normalised scalars)
    # ------------------------------------------------------------------

    def _encode_state(self, tl_id):
        state = np.zeros(80)
        lane_groups = self.tl_config[tl_id]["lane_groups"]
        incoming    = self.tl_config[tl_id]["incoming_roads"]

        for veh_id in traci.vehicle.getIDList():
            lane_id  = traci.vehicle.getLaneID(veh_id)
            edge_id, lane_str = lane_id.rsplit("_", 1)

            if edge_id not in lane_groups:
                continue

            lane_pos = self.SUMO_INT_LANE_LENGTH - traci.vehicle.getLanePosition(veh_id)

            if   lane_pos < 7:   lane_cell = 0
            elif lane_pos < 14:  lane_cell = 1
            elif lane_pos < 21:  lane_cell = 2
            elif lane_pos < 28:  lane_cell = 3
            elif lane_pos < 40:  lane_cell = 4
            elif lane_pos < 60:  lane_cell = 5
            elif lane_pos < 100: lane_cell = 6
            elif lane_pos < 200: lane_cell = 7
            elif lane_pos < 350: lane_cell = 8
            elif lane_pos <= 500: lane_cell = 9
            else: continue

            base_group = lane_groups[edge_id]
            lane_group = base_group if int(lane_str) <= 2 else base_group + 1
            state[lane_group * 10 + lane_cell] = 1

        extra = np.array([
            self._get_queue(incoming)         / 500,
            self._get_vehicle_count(incoming) / 500,
            self._get_avg_speed(incoming)     / 25,
            traci.trafficlight.getPhase(tl_id) / 7,
            self._get_queue(incoming)         / 500,   # stops ≡ queue (halting)
            self._get_co2(incoming)           / 100000,
            self.steps / self.max_steps,
            traci.simulation.getMinExpectedNumber() / 500,
        ])
        return np.concatenate([state, extra])

    # ------------------------------------------------------------------
    # TraCI helpers  (all parameterised by road list)
    # ------------------------------------------------------------------

    def _get_waiting_time(self, roads):
        total = 0.0
        road_set = set(roads)
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getRoadID(veh_id) in road_set:
                total += traci.vehicle.getAccumulatedWaitingTime(veh_id)
        return total

    def _get_queue(self, roads):
        return sum(traci.edge.getLastStepHaltingNumber(r) for r in roads)

    def _get_vehicle_count(self, roads):
        return sum(traci.edge.getLastStepVehicleNumber(r) for r in roads)

    def _get_co2(self, roads):
        return sum(traci.edge.getCO2Emission(r) for r in roads)

    def _get_avg_speed(self, roads):
        speeds = [traci.edge.getLastStepMeanSpeed(r) for r in roads]
        active = [s for s in speeds if s > 0]
        return sum(active) / len(active) if active else 0.0

    # ------------------------------------------------------------------

    def __del__(self):
        try:
            traci.close()
        except Exception:
            pass
