import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import psutil
import ray
from utils import activate_memory_growth; activate_memory_growth(cpu=False)
from deep_CFR_algorithm import DeepCFR_Coordinator
from deep_CFR_model import get_DeepCFR_model
from poker_agent import TensorflowAgent, Bet_Fct
from memory_utils import flatten_data_for_memory
from datetime import datetime

"""
Information about the parameters can be found in the specific class or
function it is used for, but are in general designed to be self explanatory.
The poker environment is initialized as a clubs_gym, whichs documentation can
be found on GitHub.
"""

# ---------- Define parameters --------------------------------------

# Experiment params
cpu_counts_for_work = 1  # number of cpus left unused by the script
experiment_str = 'increased_batch_size_to_10000'

# CFR params
num_traversals = 10_000
CFR_start_itartion = 1
CFR_iterations = 20
number_batches = 4_000
batch_size = 10_000
reservoir_size = 40_000_000

# Agent params
agent_fct = TensorflowAgent
custom_model_save_paths = None  # set agent models, None if trained from scratch
model_types = ['action', 'action_2', 'bet']  # choose one below (model_idx)
model_idx = 1
bet_fct = Bet_Fct(model_types[model_idx])
output_dim = 256  # latent dim for card embeddings

# Game params
env_str = 'LDRL-Poker-v0'
num_players = 2
num_actions = 4
start_stack = 1_000_000
num_suits = 4
num_ranks = 13
num_streets = 2
num_cards = [2, 3]
num_raises = 3
raise_size = 'pot'
blinds = [1, 2]
antes = 0
mandatory_num_hole_cards = 0

# automatic initialization of the CFR algorithm ---------------------------

# get env params
n_community_cards = [0] + num_cards[1:]
n_cards_for_hand = min(5, sum(num_cards))
max_bet_number = num_streets * (num_raises + ((num_players-1)*2))

# create environment params dict
config_dict = {'num_players': num_players,
               'num_streets': num_streets,
               'blinds': blinds,
               'antes': antes,
               'raise_sizes': raise_size,
               'num_raises': num_raises,
               'num_suits': num_suits,
               'num_ranks': num_ranks,
               'num_hole_cards': num_cards[0],
               'mandatory_num_hole_cards': mandatory_num_hole_cards,
               'num_community_cards': n_community_cards,
               'start_stack': start_stack,
               'num_cards_for_hand': n_cards_for_hand}

# get number of cpus for amount of runner
num_cpus = psutil.cpu_count(logical=True) - cpu_counts_for_work
ray.init(logging_level=logging.INFO)

# check if num_cpus good to go
if not num_traversals > num_cpus:
    # need less runners
    num_cpus = num_traversals

# create save folders
t_start = datetime.now()
datetime_str = t_start.strftime('%Y%m%d-%H%M%S')
exp_folder = f'exp_data/{datetime_str}-{experiment_str}/'
models_dir = os.path.join(exp_folder, f'train_{model_types[model_idx]}-Model/')
mem_dir = os.path.join(exp_folder, f'memories_{model_types[model_idx]}-Model/')

# initialize training networks
if custom_model_save_paths is not None:
    model_save_paths = custom_model_save_paths
else:
    model_save_paths = [os.path.join(models_dir, f'random_value_model_p_{i}') for i in range(num_players)]
    cfr_models = [get_DeepCFR_model(output_dim, num_cards, max_bet_number,
                  num_actions, zero_outputs=True) for _ in range(num_players)]

    saves = [model.save(fn) for fn, model in zip(model_save_paths, cfr_models)]
    agents = [agent_fct(p) for p in model_save_paths]

# prepare dict for env creation within runner
runner_kwargs = {'model_save_paths': model_save_paths,
                 'agent_fct': agent_fct,
                 'config_dict': config_dict,
                 'max_bet_number': max_bet_number,
                 'bet_fct': bet_fct}

# start deep CFR
trainer = DeepCFR_Coordinator(memory_buffer_size=500,
                              reservoir_size=reservoir_size,
                              batch_size=batch_size,
                              vector_length=sum(num_cards) + max_bet_number + num_actions + 1,
                              num_actions=num_actions,
                              num_batches=number_batches,
                              output_dim=output_dim,
                              n_cards=num_cards,
                              flatten_func=flatten_data_for_memory,
                              memory_dir=mem_dir,
                              result_dir=models_dir)

trainer.deep_CFR(env_str, config_dict, CFR_start_itartion, CFR_iterations,
                 num_traversals, num_players, runner_kwargs, num_runners=num_cpus)

# trainer.train_strat_model(max_bet_number)
