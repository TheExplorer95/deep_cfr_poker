from utils import get_info_state, activate_memory_growth; activate_memory_growth(cpu=False)
import numpy as np
from poker_agent import TensorflowAgent, Bet_Fct
from jupyter_nb_utils import plot_results_0
import gym
import clubs_gym
from datetime import datetime
import pickle
import os
from tqdm import trange

"""
Experimental code used for evaluating entire dirs, not intended for basic use.
Will be integrated into jupyter notebook after finish up. Automates some stuff.
"""

n_games = 40_000
save_plot = True
model_output_types = ['action', 'action_2', 'bet']
model_type_index = 1
model_folder = 'training_batch_size_10000/mathis_model'
random_model_fn = 'random_value_model'

# allways [p0, p1]

# -1 = random, -2 strategy
players = [[0,  1],
           [0, -1],
           [1, -1]] * 3

CFR_iterations = [[1, 1],
                  [1, 1],
                  [1, 1]] * 3

eval_strategy_net = [[False, False],
                     [False, False],
                     [False, False]] * 3

# Model Stuff ---------------

# model type
model_type = model_output_types[model_type_index]
bet_fct = Bet_Fct(model_type)

# Agent used for sampling
agent_fct = TensorflowAgent

# model paths
strategy = ['advantage-model', 'strategy-model']
trained_model_dir = f'trained_models/{model_type}_models/{model_folder}/'

# folders for models
fns_p0 = []
for p, iteration in zip(np.array(players)[:, 0], np.array(CFR_iterations)[:, 0]):
    if p == -1:
        fns_p0.append(os.path.join(trained_model_dir, f'{random_model_fn}'))
    elif p == -2:
        fns_p0.append(os.path.join(trained_model_dir, f'strategy-network_player-{p}'))
    else:
        fns_p0.append(os.path.join(trained_model_dir, f'advantage-network_player-{p}_CRF-iteration-{iteration}'))
fns_p1 = []
for p, iteration in zip(np.array(players)[:, 1], np.array(CFR_iterations)[:, 1]):
    if p == -1:
        fns_p1.append(os.path.join(trained_model_dir, f'{random_model_fn}'))
    elif p == -2:
        fns_p1.append(os.path.join(trained_model_dir, f'strategy-network_player-{p}'))
    else:
        fns_p1.append(os.path.join(trained_model_dir, f'advantage-network_player-{p}_CRF-iteration-{iteration}'))

sub_titles = []
for p0, p1 in players:
    sub_titles.append(f'p0 CFR_it {p0} - p1 CFR_it{p1}')

player_labels_p0 = []
for p, iteration in zip(np.array(players)[:, 0], np.array(CFR_iterations)[:, 0]):
    if p == -1:
        player_labels_p0.append(f'{random_model_fn}')
    elif p == -2:
        player_labels_p0.append(f'strategy-network_player-{p}')
    else:
        player_labels_p0.append(f'advantage-network_player-{p}_CRF-iteration-{iteration}')
player_labels_p1 = []
for p, iteration in zip(np.array(players)[:, 1], np.array(CFR_iterations)[:, 1]):
    if p == -1:
        player_labels_p1.append(f'{random_model_fn}')
    elif p == -2:
        player_labels_p1.append(f'strategy-network_player-{p}')
    else:
        player_labels_p1.append(f'advantage-network_player-{p}_CRF-iteration-{iteration}')

# Env stuff -------------------------

# Set game parameters
env_str = 'LDRL-Poker-v0'
num_players = 2
num_streets = 2
num_raises = 3
num_actions = 4
num_cards = [2, 3]
num_suits = 4
num_ranks = 13

# automatic setting of some params
n_community_cards = [0] + num_cards[1:]
n_cards_for_hand = min(5, sum(num_cards))
max_bet_number = num_streets * (num_raises + ((num_players-1) * 2))

# environment params dict
config_dict = {'num_players': num_players,
               'num_streets': num_streets,
               'blinds': [1, 2],
               'antes': 0,
               'raise_sizes': 'pot',
               'num_raises': num_raises,
               'num_suits': num_suits,
               'num_ranks': num_ranks,
               'num_hole_cards': num_cards[0],
               'mandatory_num_hole_cards': 0,
               'num_community_cards': n_community_cards,
               'start_stack': 10_000_000,
               'num_cards_for_hand': n_cards_for_hand}

clubs_gym.envs.register({env_str: config_dict})

# iterate over possible combinations
iterations = len(sub_titles)
count = 0
for path_i in range(iterations):
    count += 1
    print(f'[INFO] - Started model path {count}.')

    # create result path
    exp_path = f'numberGames-{n_games}_modelType-{model_type}_model_dir-{model_folder}_{sub_titles[path_i]}'
    t_start = datetime.now()
    datetime_str = t_start.strftime('%Y%m%d-%H%M%S')
    results_dir = os.path.join('results/eval', datetime_str + exp_path)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # create new agents
    agents = []
    if eval_strategy_net[path_i][0]:
        agents.append(agent_fct(fns_p0[path_i], regr_matching=False))
    else:
        agents.append(agent_fct(fns_p0[path_i]))

    if eval_strategy_net[path_i][1]:
        agents.append(agent_fct(fns_p1[path_i], regr_matching=False))
    else:
        agents.append(agent_fct(fns_p1[path_i]))

    # initialize env
    env = gym.make(env_str)
    env.register_agents(agents)

    # data logs
    reward_history = None
    action_history_p0 = []
    action_history_p1 = []
    preflop_history_p0 = []
    preflop_history_p1 = []
    flop_history_p0 = []
    flop_history_p1 = []

    max_action = num_actions

    for i in trange(n_games):
        obs = env.reset()
        counter = 1

        history = []
        while True:
            counter += 1

            # non terminal-state
            if all(obs['active']) and not obs['action'] == -1:
                # 1.
                # agent chooses action based on info_state
                info_state = get_info_state(
                    obs, history, max_bet_number, env.dealer.num_streets, config_dict)
                action_idx = env.act(info_state)

                # save bets for plots
                if obs['action'] == 0:
                    action_history_p0.append(action_idx)
                    if not obs['community_cards']:
                        preflop_history_p0.append(action_idx)
                    else:
                        flop_history_p0.append(action_idx)
                else:
                    action_history_p1.append(action_idx)
                    if not obs['community_cards']:
                        preflop_history_p1.append(action_idx)
                    else:
                        flop_history_p1.append(action_idx)

                # 2.
                # take action within environment
                bet = bet_fct(action_idx, obs)
                obs, rewards, done, _ = env.step(bet)
                history.append(bet)  # for info states

            # terminal state
            else:
                # save results
                if reward_history is None:
                    reward_history = np.array([rewards])
                else:
                    reward_history = np.append(reward_history, [rewards], axis=0)

                if i == n_games-1:
                    print('[INFO] - Done.')

                break

    # save eval data ---------------------

    exp_data = {'reward_history': reward_history,
                'action_history_p0': action_history_p0,
                'action_history_p1': action_history_p1,
                'preflop_history_p0': preflop_history_p0,
                'preflop_history_p1': preflop_history_p1,
                'flop_history_p0': flop_history_p0,
                'flop_history_p1': flop_history_p1}

    fn_exp_data = os.path.join(results_dir, 'expData.pkl')

    with open(fn_exp_data, 'wb') as f:
        pickle.dump(exp_data, f)

    # make plots -----------------------------

    if save_plot:
        fn = 'plot_results.svg'
    save_path = os.path.join(results_dir, fn)

    # create plot
    plot_results_0(reward_history, num_actions, action_history_p0,
                   action_history_p1, preflop_history_p0, preflop_history_p1, save_path,
                   sub_title=sub_titles[path_i],
                   player_labels=[player_labels_p0[path_i], player_labels_p1[path_i]],
                   flop_history_p0=flop_history_p0, flop_history_p1=flop_history_p1)
