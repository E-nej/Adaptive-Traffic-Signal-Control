import glob
import os
import numpy as np
from tqdm import tqdm

from NetworkParser import NetworkParser
from MultiSumoEnv import MultiSumoEnv
from TLAgentMA import TLAgentMA
from TrafficGenerator import AutoTrafficGenerator
from CommunicationModule import CommunicationModule
import utils

# ------------------------------------------------------------------
# Only this path needs to change when targeting a new network.
# Everything else (sumocfg, TL programs, traffic) is auto-generated.
# ------------------------------------------------------------------
NET_FILE = "intersection/test.net.xml"

if __name__ == "__main__":
    # --- SETTINGS ---
    mode             = "train"   # "train" | "evaluate" | "classical"
    gui              = False
    max_steps        = 900
    total_episodes   = 10
    num_experiments  = 1
    num_vehicles     = 20
    show_plots       = True
    save_plots       = True

    sumoBinary = 'sumo-gui' if gui else 'sumo'

    # --- derive all paths from NET_FILE ---
    net_dir  = os.path.dirname(NET_FILE)
    net_stem = os.path.basename(NET_FILE)
    for ext in ('.xml', '.net'):
        if net_stem.endswith(ext):
            net_stem = net_stem[:-len(ext)]
    SUMOCFG    = os.path.join(net_dir, f"{net_stem}.sumocfg")
    TL_ADD     = os.path.join(net_dir, "tl_programs.add.xml")
    ROUTE_FILE = os.path.join(net_dir, "trips.xml")

    # --- auto-configure from the network file ---
    parser = NetworkParser(NET_FILE)
    parser.generate_tl_programs(TL_ADD)
    parser.generate_sumocfg(SUMOCFG, route_file="trips.xml", tl_add_file="tl_programs.add.xml")

    tl_ids         = parser.tl_ids
    neighbours     = parser.neighbours
    max_neighbours = parser.max_neighbours
    tl_config      = parser.tl_config

    traffic_gen = AutoTrafficGenerator(NET_FILE, ROUTE_FILE,
                                       num_vehicles=num_vehicles,
                                       max_steps=max_steps)

    stats = {
        'rewards':            np.zeros((num_experiments, total_episodes)),
        'intersection_queue': np.zeros((num_experiments, total_episodes)),
        'delay':              np.zeros((num_experiments, total_episodes)),
        'stops':              np.zeros((num_experiments, total_episodes)),
        'co2':                np.zeros((num_experiments, total_episodes)),
    }

    for experiment in range(num_experiments):

        # --- locate saved models for evaluate mode ---
        model_files = {}
        if mode == "evaluate":
            for tl_id in tl_ids:
                files = glob.glob(f'results/qmodel_{tl_id}_{experiment}_*.keras')
                if not files:
                    raise FileNotFoundError(
                        f"No saved model for {tl_id} experiment {experiment}. "
                        "Run in 'train' mode first."
                    )
                model_files[tl_id] = max(
                    files,
                    key=lambda f: int(f.rsplit('_', 1)[-1].replace('.keras', ''))
                )
                print(f"{tl_id}: loading {model_files[tl_id]}")

        env = MultiSumoEnv(sumoBinary, max_steps, SUMOCFG, tl_config=tl_config)

        agents = {
            tl_id: TLAgentMA(
                tl_id=tl_id,
                num_local_states=88,
                num_neighbours=max_neighbours,
                total_episodes=total_episodes,
                qmodel_filename=model_files.get(tl_id),
                learn=(mode == "train"),
            )
            for tl_id in tl_ids
        }

        comm = CommunicationModule(neighbours, max_neighbours=max_neighbours)

        traffic_gen.generate_routefile(seed=experiment * total_episodes)
        raw_states = env.start()

        ep_bar = tqdm(range(total_episodes), desc=f"Exp {experiment} [{mode}]",
                      unit="ep", position=0, leave=True)

        for e in ep_bar:
            # Build initial states (not needed for classical)
            curr_states = {}
            if mode != "classical":
                for tl_id in tl_ids:
                    msgs = comm.get_neighbor_messages(tl_id)
                    curr_states[tl_id] = agents[tl_id].preprocess(
                        np.concatenate([raw_states[tl_id], msgs])
                    )

            if mode == "train" and e > 0 and e % agents[tl_ids[0]].tau == 0:
                for agent in agents.values():
                    agent.sync_target()

            old_actions    = {tl_id: None for tl_id in tl_ids}
            classical_step = 0
            done           = False

            sum_queue  = {tl_id: 0.0 for tl_id in tl_ids}
            sum_reward = {tl_id: 0.0 for tl_id in tl_ids}
            sum_delay  = {tl_id: 0.0 for tl_id in tl_ids}
            sum_stops  = {tl_id: 0.0 for tl_id in tl_ids}
            sum_co2    = {tl_id: 0.0 for tl_id in tl_ids}

            step_bar = tqdm(total=max_steps, desc=f"  Ep {e:3d} steps",
                            unit="step", position=1, leave=False)

            while not done:
                # --- select actions ---
                if mode == "classical":
                    # fixed-timer: cycle all agents through actions 0→1→2→3
                    actions = {tl_id: classical_step % 4 for tl_id in tl_ids}
                    classical_step += 1
                else:
                    for tl_id, agent in agents.items():
                        comm.update_message(tl_id, agent.build_comm_message(env))
                    actions = {
                        tl_id: agents[tl_id].select_action(
                            e, curr_states[tl_id], learn=(mode == "train")
                        )
                        for tl_id in tl_ids
                    }

                # --- yellow phase for agents that changed action ---
                yellow_rewards = {tl_id: 0.0 for tl_id in tl_ids}
                needs_yellow = [
                    tl_id for tl_id in tl_ids
                    if old_actions[tl_id] is not None
                    and old_actions[tl_id] != actions[tl_id]
                ]
                if needs_yellow:
                    for tl_id in needs_yellow:
                        agents[tl_id].set_yellow_phase(old_actions[tl_id])
                    yellow_results, _ = env.step(agents[tl_ids[0]].yellow_duration)
                    step_bar.update(agents[tl_ids[0]].yellow_duration)
                    for tl_id in tl_ids:
                        yellow_rewards[tl_id] = yellow_results[tl_id][0]

                # --- green phase ---
                for tl_id, agent in agents.items():
                    agent.set_green_phase(actions[tl_id])
                green_results, done = env.step(agents[tl_ids[0]].green_duration)
                step_bar.update(agents[tl_ids[0]].green_duration)

                # --- update comm after step ---
                if mode != "classical":
                    for tl_id, agent in agents.items():
                        comm.update_message(tl_id, agent.build_comm_message(env))

                # --- per-agent bookkeeping ---
                for tl_id, agent in agents.items():
                    reward   = green_results[tl_id][0] + yellow_rewards[tl_id]
                    raw_next = green_results[tl_id][1]

                    if mode != "classical":
                        msgs = comm.get_neighbor_messages(tl_id)
                        next_state = agent.preprocess(np.concatenate([raw_next, msgs]))
                        if mode == "train":
                            agent.store(curr_states[tl_id], actions[tl_id],
                                        reward, next_state, done)
                            agent.replay()
                        curr_states[tl_id] = next_state

                    old_actions[tl_id] = actions[tl_id]

                    m = env.last_metrics[tl_id]
                    sum_queue[tl_id]  += m['queue']
                    sum_delay[tl_id]  += m['delay']
                    sum_stops[tl_id]  += m['stops']
                    sum_co2[tl_id]    += m['co2']
                    if reward < 0:
                        sum_reward[tl_id] += reward

            step_bar.close()
            steps_used = max(1, env.steps)

            avg_reward = float(np.mean([sum_reward[t] / steps_used for t in tl_ids]))
            avg_queue  = float(np.mean([sum_queue[t]  / steps_used for t in tl_ids]))
            avg_delay  = float(np.mean([sum_delay[t]  / steps_used for t in tl_ids]))
            avg_stops  = float(np.mean([sum_stops[t]  / steps_used for t in tl_ids]))
            avg_co2    = float(np.mean([sum_co2[t]    / steps_used for t in tl_ids]))

            stats['rewards'][experiment, e]            = avg_reward
            stats['intersection_queue'][experiment, e] = avg_queue
            stats['delay'][experiment, e]              = avg_delay
            stats['stops'][experiment, e]              = avg_stops
            stats['co2'][experiment, e]                = avg_co2

            ep_bar.set_postfix({
                'reward': f'{avg_reward:.4f}',
                'queue':  f'{avg_queue:.2f}',
                'delay':  f'{avg_delay:.2f}',
            })

            # --- checkpointing (train only) ---
            if mode == "train":
                for tl_id, agent in agents.items():
                    agent_key = f'{tl_id}_{experiment}'
                    utils.save_qmodel(agent.QModel, agent_key, e)
                    if e > 0:
                        utils.remove_qmodel(agent_key, e - 1)

                np.save(f'results/stats_2TL_{experiment}_{e}.npy', stats)
                if e > 0:
                    try:
                        os.remove(f'results/stats_2TL_{experiment}_{e - 1}.npy')
                    except FileNotFoundError:
                        pass

            if e + 1 < total_episodes:
                traffic_gen.generate_routefile(
                    seed=experiment * total_episodes + e + 1
                )
            raw_states = env.reset()

        del env
        del agents

        print(f'Experiment {experiment} [{mode}] complete')
        utils.plot_rewards(stats['rewards'][:experiment + 1], show=show_plots, save=save_plots)
        utils.plot_intersection_queue_size(stats['intersection_queue'][:experiment + 1], show=show_plots, save=save_plots)
        utils.plot_delay(stats['delay'][:experiment + 1], show=show_plots, save=save_plots)
        utils.plot_stops(stats['stops'][:experiment + 1], show=show_plots, save=save_plots)
        utils.plot_co2(stats['co2'][:experiment + 1], show=show_plots, save=save_plots)
