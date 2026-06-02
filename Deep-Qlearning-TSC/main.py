# -*- coding: utf-8 -*-

from  Model import Model
from SumoEnv import SumoEnv
import numpy as np
from TrafficGenerator import TrafficGenerator
import utils
import copy
import sys
import os

from TLAgent import TLAgent

if __name__ == "__main__":
    # --- TRAINING OPTIONS ---
    learn = True
    gui = False

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = 'sumo'
    else:
        sumoBinary = 'sumo-gui'

    max_steps = 100
    total_episodes = 10
    num_experiments = 1
    num_cars_generated = 10
    show_plots = True
    save_plots = True

    # --- NETWORK CONFIG ---
    # Set to a .net.xml path to run independent agents on any multi-intersection network.
    # Set to None to use the original single-intersection setup.
    NET_FILE = "intersection/test.net.xml"

    # =========================================================================
    # MULTI-INTERSECTION PATH
    # One independent TLAgent per intersection, no communication between them.
    # Agent behaviour (DQN, reward, state) is identical to the original authors.
    # =========================================================================
    if NET_FILE:
        from NetworkParser import NetworkParser
        from TrafficGenerator import AutoTrafficGenerator
        from MultiSumoEnv import MultiSumoEnv
        import helpers
        from tqdm import tqdm

        net_dir  = os.path.dirname(NET_FILE)
        net_stem = os.path.splitext(os.path.basename(NET_FILE))[0]
        SUMOCFG    = os.path.join(net_dir, f"{net_stem}.sumocfg")
        TL_ADD     = os.path.join(net_dir, "tl_programs.add.xml")
        ROUTE_FILE = os.path.join(net_dir, "trips.xml")

        parser = NetworkParser(NET_FILE)
        parser.generate_tl_programs(TL_ADD)
        parser.generate_sumocfg(SUMOCFG, route_file="trips.xml",
                                tl_add_file="tl_programs.add.xml")

        tl_ids      = parser.tl_ids
        tl_config   = parser.tl_config
        traffic_gen = AutoTrafficGenerator(NET_FILE, ROUTE_FILE,
                                           num_vehicles=num_cars_generated,
                                           max_steps=max_steps)

        stats = {
            'rewards':            np.zeros((num_experiments, total_episodes)),
            'intersection_queue': np.zeros((num_experiments, total_episodes)),
            'delay':              np.zeros((num_experiments, total_episodes)),
            'stops':              np.zeros((num_experiments, total_episodes)),
            'co2':                np.zeros((num_experiments, total_episodes)),
            'spawned':            np.zeros((num_experiments, total_episodes)),
            'arrived':            np.zeros((num_experiments, total_episodes)),
            'emergency_stops':    np.zeros((num_experiments, total_episodes)),
            'collisions':         np.zeros((num_experiments, total_episodes)),
        }

        for experiment in range(num_experiments):
            env = MultiSumoEnv(sumoBinary, max_steps, SUMOCFG,
                               tl_config=tl_config, num_vehicles=num_cars_generated)

            # One independent TLAgent per intersection — no communication
            agents = {
                tl_id: TLAgent(
                    env=None, traffic_gen=traffic_gen,
                    max_steps=max_steps, num_experients=num_experiments,
                    total_episodes=total_episodes,
                    qmodel_filename=None, stats_filename=None,
                    stats=stats, init_epoch=0, learn=learn,
                    tl_id=tl_id,
                )
                for tl_id in tl_ids
            }

            traffic_gen.generate_routefile(seed=experiment * total_episodes)
            raw_states = env.start()

            ep_bar = tqdm(range(total_episodes),
                          desc=f"Exp {experiment} [singleAgent]",
                          unit="ep", position=0, leave=True)
            best_reward = -np.inf
            best_epoch  = None

            for e in ep_bar:
                curr_states = {
                    t: agents[t]._preprocess_input(raw_states[t]) for t in tl_ids
                }
                old_actions = {t: None for t in tl_ids}
                done = False

                sum_queue  = {t: 0.0 for t in tl_ids}
                sum_delay  = {t: 0.0 for t in tl_ids}
                sum_stops  = {t: 0.0 for t in tl_ids}
                sum_co2    = {t: 0.0 for t in tl_ids}
                sum_reward = {t: 0.0 for t in tl_ids}
                sum_spawned   = 0
                sum_arrived   = 0
                sum_emergency = 0
                sum_collisions = 0

                if e > 0 and e % agents[tl_ids[0]].tau == 0:
                    for agent in agents.values():
                        agent._sync_target_model()

                step_bar = tqdm(total=max_steps, desc=f"  Ep {e:3d} steps",
                                unit="step", position=1, leave=False)

                while not done:
                    # Each agent picks its action independently (no messages)
                    actions = {
                        t: agents[t]._agent_policy(e, curr_states[t]) for t in tl_ids
                    }

                    # Yellow phase for agents that changed action
                    yellow_rewards = {t: 0.0 for t in tl_ids}
                    needs_yellow = [
                        t for t in tl_ids
                        if old_actions[t] is not None and old_actions[t] != actions[t]
                    ]
                    if needs_yellow:
                        for t in needs_yellow:
                            agents[t]._set_yellow_phase(old_actions[t])
                        yellow_results, _ = env.step(agents[tl_ids[0]].yellow_duration)
                        step_bar.update(agents[tl_ids[0]].yellow_duration)
                        for t in tl_ids:
                            yellow_rewards[t] = yellow_results[t][0]
                        sm = env.last_sim_metrics
                        sum_spawned    += sm['spawned']
                        sum_arrived    += sm['arrived']
                        sum_emergency  += sm['emergency_stops']
                        sum_collisions += sm['collisions']

                    # Green phase
                    for t in tl_ids:
                        agents[t]._set_green_phase(actions[t])
                    green_results, done = env.step(agents[tl_ids[0]].green_duration)
                    step_bar.update(agents[tl_ids[0]].green_duration)
                    sm = env.last_sim_metrics
                    sum_spawned    += sm['spawned']
                    sum_arrived    += sm['arrived']
                    sum_emergency  += sm['emergency_stops']
                    sum_collisions += sm['collisions']

                    # Per-agent bookkeeping — independent, no communication
                    for t in tl_ids:
                        reward     = green_results[t][0] + yellow_rewards[t]
                        next_state = agents[t]._preprocess_input(green_results[t][1])

                        if learn:
                            agents[t]._add_to_replay_buffer(
                                curr_states[t], actions[t], reward, next_state, done
                            )
                            agents[t]._replay()

                        curr_states[t] = next_state
                        old_actions[t] = actions[t]

                        m = env.last_metrics[t]
                        sum_queue[t]  += m['queue']
                        sum_delay[t]  += m['delay']
                        sum_stops[t]  += m['stops']
                        sum_co2[t]    += m['co2']
                        if reward < 0:
                            sum_reward[t] += reward

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
                stats['spawned'][experiment, e]            = sum_spawned
                stats['arrived'][experiment, e]            = sum_arrived
                stats['emergency_stops'][experiment, e]    = sum_emergency / max(1, steps_used)
                stats['collisions'][experiment, e]         = sum_collisions

                ep_bar.set_postfix({
                    'reward': f'{avg_reward:.4f}',
                    'queue':  f'{avg_queue:.2f}',
                })

                # Save best models (one per TL, prefixed sa_ for singleAgent)
                if learn and avg_reward > best_reward:
                    best_reward = avg_reward
                    for t in tl_ids:
                        utils.save_qmodel(agents[t].QModel, f'sa_{t}_{experiment}', e)
                        if best_epoch is not None:
                            utils.remove_qmodel(f'sa_{t}_{experiment}', best_epoch)
                    best_epoch = e

                # Rolling stats checkpoint
                np.save(f'results/stats_sa_{experiment}_{e}.npy', stats)
                if e > 0:
                    try:
                        os.remove(f'results/stats_sa_{experiment}_{e - 1}.npy')
                    except FileNotFoundError:
                        pass

                if e + 1 < total_episodes:
                    traffic_gen.generate_routefile(
                        seed=experiment * total_episodes + e + 1
                    )
                    raw_states = env.reset()

            del env
            del agents
            print(f'Experiment {experiment} [singleAgent] complete.')

        utils.plot_rewards(stats['rewards'], run_tag='sa', show=show_plots, save=save_plots)
        utils.plot_intersection_queue_size(stats['intersection_queue'], run_tag='sa', show=show_plots, save=save_plots)
        utils.plot_delay(stats['delay'], run_tag='sa', show=show_plots, save=save_plots)
        utils.plot_stops(stats['stops'], run_tag='sa', show=show_plots, save=save_plots)
        utils.plot_co2(stats['co2'], run_tag='sa', show=show_plots, save=save_plots)
        utils.plot_throughput(stats['spawned'], stats['arrived'], run_tag='sa', show=show_plots, save=save_plots)
        utils.plot_safety(stats['emergency_stops'], stats['collisions'], run_tag='sa', show=show_plots, save=save_plots)

        # Overlay comparison if multiagent results exist
        ma_stats = utils.load_latest_stats('results/stats_2TL_*.npy')
        if ma_stats:
            utils.plot_comparison(stats, ma_stats, run_tag='sa_vs_ma', show=show_plots, save=save_plots)

    # =========================================================================
    # ORIGINAL SINGLE-INTERSECTION PATH (original authors' code, unchanged)
    # =========================================================================
    else:
        # ce je ime modela recimo: qmodel_0_9
        # 0 predstavlja experiment number, 9 predstavlja epoch number (stetje je od 0)
        # ce das total_episodes na vec kot 10, potem se nadaljuje training tega modela, ce das na manj kot 10, potem se nadaljuje training naslednjega modela (npr. qmodel_1_0)
        #
        traffic_gen = TrafficGenerator(max_steps, num_cars_generated=num_cars_generated)
        qmodel_filename, stats_filename = utils.get_file_names()
        init_experiment, init_epoch = utils.get_init_epoch( stats_filename, total_episodes)
        if not learn:
            init_experiment, init_epoch = 0, 0
        print('init_experiment={} init_epoch={}'.format(init_experiment,init_epoch ))
        stats = utils.get_stats(stats_filename, num_experiments, total_episodes)

        # Safeguard: prevent data loss from config mismatches
        if stats_filename and learn:
            try:
                saved_stats = np.load(stats_filename, allow_pickle=True)[()]
                saved_shape = saved_stats['rewards'].shape
                expected_shape = (num_experiments, total_episodes)

                if saved_shape == expected_shape and init_experiment >= num_experiments:
                    print(f"Config matches saved stats {saved_shape}. All experiments already complete. Nothing to do.")
                    sys.exit(0)
                elif saved_shape[0] > num_experiments:
                    print(f"ERROR: num_experiments reduced from {saved_shape[0]} to {num_experiments}. This would lose data from experiments {num_experiments}-{saved_shape[0]-1}.")
                    print(f"To proceed, either:")
                    print(f"  1. Increase num_experiments to {saved_shape[0]} or higher")
                    print(f"  2. Remove old result files in results/ folder")
                    sys.exit(1)
            except Exception as e:
                print(f"Warning: Could not check saved stats shape: {e}")


        for experiment in range(init_experiment, num_experiments):
            env = SumoEnv(sumoBinary,max_steps )
            tl = TLAgent( env, traffic_gen, max_steps, num_experiments, total_episodes, qmodel_filename, stats_filename, stats,init_epoch, learn )
            init_epoch = 0 # reset init_epoch after first experiment
            if learn:
                tl.train(experiment)
            else:
                seeds = np.load('seed.npy')
                tl.evaluate_model( experiment, seeds)

            stats = copy.deepcopy(tl.stats)
            print(stats['rewards'][0:experiment+1, :])
            print(stats['intersection_queue'][0:experiment+1, :])
            utils.plot_rewards(stats['rewards'][0:experiment+1, :])
            utils.plot_intersection_queue_size(stats['intersection_queue'][0:experiment+1, :])
            utils.plot_delay(stats['delay'][0:experiment+1, :])
            utils.plot_stops(stats['stops'][0:experiment+1, :])
            utils.plot_co2(stats['co2'][0:experiment+1, :])
            del env
            del tl
            print('Experiment {} complete.........'.format(experiment))
