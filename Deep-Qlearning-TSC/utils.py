# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:55:20 2020

@author: XZ01M2
"""

import os

import matplotlib.pyplot as plt # type: ignore
import numpy as np 
import glob
import seaborn as sns # type: ignore

FOLDER = 'results/'

def get_file_names():
    qmodel_file_name = glob.glob(f'{FOLDER}qmodel*')
    stats_file_name = glob.glob(f'{FOLDER}stats*')
    
    if not qmodel_file_name:
        qmodel_file_name = ''
    else:
        qmodel_file_name = qmodel_file_name[0]

    if not stats_file_name:
        stats_file_name = ''
    else:
        stats_file_name = stats_file_name[0]

    return qmodel_file_name, stats_file_name

def get_init_epoch(filename, total_episodes):
    if filename:
        index = filename.find('_')
        exp_start = index + 1 
        exp_end  = int(filename.find('_', exp_start))
        exp = int(filename[exp_start:exp_end])
        epoch_start= exp_end + 1
        epoch_end = int(filename.find('.', epoch_start))
        epoch = int(filename[epoch_start:epoch_end])
        if epoch < total_episodes -1:
            epoch +=1
        else:
            epoch = 0
            exp +=1
    else:
        exp=0
        epoch = 0
    return exp , epoch

def _ensure_stats_keys(stats, shape):
    for key in ('rewards', 'intersection_queue', 'delay', 'stops', 'co2'):
        if key not in stats or stats[key].shape != shape:
            existing = stats.get(key)
            store = np.zeros(shape)
            if existing is not None and existing.ndim == 2:
                n_exp = min(shape[0], existing.shape[0])
                n_ep  = min(shape[1], existing.shape[1])
                store[:n_exp, :n_ep] = existing[:n_exp, :n_ep]
            stats[key] = store
    return stats

def get_stats(stats_filename, num_experiments, total_episodes, learn = True):
    expected_shape = (num_experiments, total_episodes)
    if stats_filename and learn:
        stats = np.load(stats_filename, allow_pickle = True)[()]
        stats = _ensure_stats_keys(stats, expected_shape)
    else:
        reward_store = np.zeros((num_experiments,total_episodes))
        intersection_queue_store = np.zeros((num_experiments,total_episodes))
        delay_store = np.zeros((num_experiments,total_episodes))
        stops_store = np.zeros((num_experiments,total_episodes))
        co2_store = np.zeros((num_experiments,total_episodes))
        stats = {
            'rewards': reward_store,
            'intersection_queue': intersection_queue_store,
            'delay': delay_store,
            'stops': stops_store,
            'co2': co2_store,
        }

    return stats
    
def plot_sample(sample, title, xlabel, legend_label, show=True):
    ax = sns.kdeplot(sample, label=legend_label)
    ax.set(xlabel=xlabel, title=title)
    ax.legend()
    if show:
        plt.show()
    
def _plot_path(name, run_tag):
    suffix = f'_{run_tag}' if run_tag else ''
    return f'{FOLDER}plot_{name}{suffix}.png'

def plot_rewards(reward_store, save=True, show=False, run_tag=''):
    x = np.mean(reward_store, axis=0)
    plt.figure()
    plt.plot(x, label="Average reward per step")
    plt.xlabel('Episodes')
    plt.ylabel('Average reward per step')
    plt.title('Average reward across episodes')
    plt.legend()
    if save:
        plt.savefig(_plot_path('rewards', run_tag), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_intersection_queue_size(intersection_queue_store, save=True, show=False, run_tag=''):
    x = np.mean(intersection_queue_store, axis=0)
    plt.figure()
    plt.plot(x, label="Average intersection queue size per step", color='m')
    plt.xlabel('Episodes')
    plt.ylabel('Average intersection queue size per step')
    plt.title('Average intersection queue size across episodes')
    plt.legend()
    if save:
        plt.savefig(_plot_path('queue', run_tag), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_delay(delay_store, save=True, show=False, run_tag=''):
    x = np.mean(delay_store, axis=0)
    plt.figure()
    plt.plot(x, label="Average delay per step", color='steelblue')
    plt.xlabel('Episodes')
    plt.ylabel('Average delay per step (s)')
    plt.title('Average vehicle delay across episodes')
    plt.legend()
    if save:
        plt.savefig(_plot_path('delay', run_tag), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_stops(stops_store, save=True, show=False, run_tag=''):
    x = np.mean(stops_store, axis=0)
    plt.figure()
    plt.plot(x, label="Average stops per step", color='darkorange')
    plt.xlabel('Episodes')
    plt.ylabel('Average stops per step')
    plt.title('Average vehicle stops across episodes')
    plt.legend()
    if save:
        plt.savefig(_plot_path('stops', run_tag), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_co2(co2_store, save=True, show=False, run_tag=''):
    x = np.mean(co2_store, axis=0)
    plt.figure()
    plt.plot(x, label="Average CO2 per step", color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Average CO2 per step (mg)')
    plt.title('Average CO2 emissions across episodes')
    plt.legend()
    if save:
        plt.savefig(_plot_path('co2', run_tag), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_throughput(spawned_store, arrived_store, save=True, show=False, run_tag=''):
    plt.figure()
    plt.plot(np.mean(spawned_store, axis=0), label="Spawned", color='steelblue')
    plt.plot(np.mean(arrived_store, axis=0), label="Arrived", color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Vehicles per episode')
    plt.title('Vehicle throughput across episodes')
    plt.legend()
    if save:
        plt.savefig(_plot_path('throughput', run_tag), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_safety(emergency_store, collision_store, save=True, show=False, run_tag=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(np.mean(emergency_store, axis=0), color='red', label="Emergency stops per step")
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Emergency stops per step')
    ax1.set_title('Emergency braking events')
    ax1.legend()

    ax2.plot(np.mean(collision_store, axis=0), color='darkred', label="Collisions per episode")
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Collisions per episode')
    ax2.set_title('Collisions across episodes')
    ax2.legend()

    plt.tight_layout()
    if save:
        plt.savefig(_plot_path('safety', run_tag), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def remove_qmodel(experiment, e):
    os.remove('{}qmodel_{}_{}.keras'.format(FOLDER, experiment, e))

def remove_stats(experiment, e):
    os.remove('{}stats_{}_{}.npy'.format(FOLDER, experiment, e))

def save_qmodel(qmodel, experiment, e):
    qmodel.save('{}qmodel_{}_{}.keras'.format(FOLDER, experiment, e))
    
def save_stats(stats, experiment, e):
    np.save('{}stats_{}_{}.npy'.format(FOLDER, experiment, e), stats)