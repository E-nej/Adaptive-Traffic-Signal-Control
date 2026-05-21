# -*- coding: utf-8 -*-

from  Model import Model
from SumoEnv import SumoEnv
import numpy as np
from TrafficGenerator import TrafficGenerator
import utils 
import copy
import sys

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

    # initializations
    # max_steps = 5400  # seconds = 1 h 30 min each episode
    # total_episodes = 100
    # ce je ime modela recimo: qmodel_0_9 
    # 0 predstavlja experiment number, 9 predstavlja epoch number (stetje je od 0)
    # ce das total_episodes na vec kot 10, potem se nadaljuje training tega modela, ce das na manj kot 10, potem se nadaljuje training naslednjega modela (npr. qmodel_1_0)
    #
    max_steps = 900
    total_episodes = 30
    num_experiments = 1
    
    traffic_gen = TrafficGenerator(max_steps, num_cars_generated=500)
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