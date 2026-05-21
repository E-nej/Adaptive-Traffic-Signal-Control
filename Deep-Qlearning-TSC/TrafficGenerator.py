#MIT License

#Copyright (c) 2019 Andrea Vidali

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import math
import os
import subprocess
import sys

import numpy as np

# HANDLE THE GENERATION OF VEHICLES IN ONE EPISODE
class TrafficGenerator:
    def __init__(self, max_steps, num_cars_generated=500):
        self._n_cars_generated = num_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    # generation of routes of cars
    def generate_routefile(self, seed):     
        if seed >=0 :
            np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - min_old) + min_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/trips.trips.4L.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)


class TrafficGenerator2TL:
    """Route generator for the 2-intersection linear corridor (TL1 ↔ TL2)."""

    # Straight routes: vehicle passes through one or both intersections without turning
    STRAIGHT_ROUTES = [
        ("W_E",   "W2TL1 TL12TL2 TL22E"),   # west → east (through both)
        ("E_W",   "E2TL2 TL22TL1 TL12W"),   # east → west (through both)
        ("N1_S1", "N12TL1 TL12S1"),          # north1 → south1 (TL1 only)
        ("S1_N1", "S12TL1 TL12N1"),          # south1 → north1 (TL1 only)
        ("N2_S2", "N22TL2 TL22S2"),          # north2 → south2 (TL2 only)
        ("S2_N2", "S22TL2 TL22N2"),          # south2 → north2 (TL2 only)
    ]

    # Turn routes: vehicle turns at one intersection
    TURN_ROUTES = [
        ("W_S1",  "W2TL1 TL12S1"),           # right turn at TL1
        ("W_N1",  "W2TL1 TL12N1"),           # left turn at TL1
        ("N1_W",  "N12TL1 TL12W"),           # right turn at TL1
        ("S1_W",  "S12TL1 TL12W"),           # left turn at TL1
        ("E_N2",  "E2TL2 TL22N2"),           # right turn at TL2
        ("E_S2",  "E2TL2 TL22S2"),           # left turn at TL2
        ("N2_E",  "N22TL2 TL22E"),           # left turn at TL2
        ("S2_E",  "S22TL2 TL22E"),           # right turn at TL2
    ]

    def __init__(self, max_steps, num_cars_generated=700):
        self._n_cars_generated = num_cars_generated
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        if seed >= 0:
            np.random.seed(seed)

        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        car_gen_steps = np.rint(
            ((self._max_steps) / (max_old - min_old)) * (timings - min_old)
        )

        with open("intersection_2TL/trips.2TL.xml", "w") as routes:
            print('<routes>', file=routes)
            print('    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />', file=routes)
            for rid, edges in self.STRAIGHT_ROUTES + self.TURN_ROUTES:
                print(f'    <route id="{rid}" edges="{edges}"/>', file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                if np.random.uniform() < 0.75:
                    rid, _ = self.STRAIGHT_ROUTES[np.random.randint(len(self.STRAIGHT_ROUTES))]
                else:
                    rid, _ = self.TURN_ROUTES[np.random.randint(len(self.TURN_ROUTES))]
                print(f'    <vehicle id="{rid}_{car_counter}" type="standard_car" route="{rid}" depart="{step}" departLane="random" departSpeed="10" />', file=routes)

            print('</routes>', file=routes)


def _find_sumo_tools():
    candidates = [
        '/usr/share/sumo/tools',
        'C:\\Program Files (x86)\\Eclipse\\Sumo\\tools',
        'C:\\Program Files\\Sumo\\tools',
    ]
    s_h = os.environ.get('SUMO_HOME')
    if s_h:
        candidates.insert(0, os.path.join(s_h, 'tools'))
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError("SUMO tools directory not found. Set the SUMO_HOME environment variable.")


class AutoTrafficGenerator:
    """Generates random trips for any SUMO network using randomTrips.py.

    Works with any .net.xml — no hardcoded edge names.  Calls duarouter
    internally (via randomTrips.py -r) to produce a fully-routed file.
    """

    def __init__(self, net_file, route_file, num_vehicles=700, max_steps=900):
        self.net_file = net_file
        self.route_file = route_file
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        tools = _find_sumo_tools()
        self._rtrips = os.path.join(tools, 'randomTrips.py')
        if not os.path.exists(self._rtrips):
            raise RuntimeError(f"randomTrips.py not found at {self._rtrips}")

    def generate_routefile(self, seed=42):
        period = self.max_steps / self.num_vehicles
        cmd = [
            sys.executable, self._rtrips,
            '-n', self.net_file,
            '-r', self.route_file,
            '-e', str(self.max_steps),
            '-p', str(period),
            '--fringe-factor', '10',
            '--seed', str(int(seed)),
            '--trip-attributes', 'departLane="best" departSpeed="max"',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"randomTrips.py failed (seed={seed}):\n{result.stderr}"
            )
