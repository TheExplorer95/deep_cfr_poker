import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import psutil
import ray
from utils_ray import activate_memory_growth; activate_memory_growth(cpu=False)
from deep_CFR_traversal_ray import Coordinator
from Tensorflow_Model import get_DeepCFR_model
from PokerAgent import TensorflowAgent, Bet_Fct
from memory_utils import flatten_data_for_memory

"""
Information of the here set parameters can be found in the specific class or
function it is used for. The poker environment is initialized as a clubs_gym,
whichs documentation can be found on GitHub.

Maybe check if there is a readme file for the parameters, its definetly
to make one...at least at some point.
"""

# how many cpus do you wanna leave free from work during CRF sampling?
cpu_counts_for_work = 1

# -------------------- The Algorithm -------------------------------------
# 1.
# Set algorithm parameters
num_traversals = 10_000
CFR_start_itartion = 1
CFR_iterations = 20
number_batches = 4_000
batch_size = 10_000
reservoir_size = 40_000_000
output_dim = 256  # model for card embeddings

# Set agent
custom_model_save_paths = None  # set agents strategy networks, None if trained from scratch
model_output_types = ['action', 'action_2', 'bet']  # choose one below (index)
model_type = model_output_types[1]
agent_fct = TensorflowAgent
bet_fct = Bet_Fct(model_type)

# Set game parameters
env_str = 'LDRL-Poker-v0'
num_players = 2
num_streets = 2
num_raises = 3
num_actions = 4
num_cards = [2, 3]

n_community_cards = [0] + num_cards[1:]
n_cards_for_hand = min(5, sum(num_cards))
max_bet_number = num_streets * (num_raises + ((num_players-1)*2))

# environment params dict
config_dict = {'num_players': num_players,
               'num_streets': num_streets,
               'blinds': [1, 2],
               'antes': 0,
               'raise_sizes': 'pot',
               'num_raises': num_raises,
               'num_suits': 4,
               'num_ranks': 13,
               'num_hole_cards': num_cards[0],
               'mandatory_num_hole_cards': 0,
               'num_community_cards': n_community_cards,
               'start_stack': 1_000_000,
               'num_cards_for_hand': n_cards_for_hand}

# get number of cpus for amount of runner
num_cpus = psutil.cpu_count(logical=True) - cpu_counts_for_work
ray.init(logging_level=logging.INFO)

# check if num_cpus good to go
if not num_traversals > num_cpus:
    # need less runners
    num_cpus = num_traversals

# 2.
# initialize training networks
if custom_model_save_paths is not None:
    model_save_paths = custom_model_save_paths
else:
    model_save_paths = [f'random_value_model_p_{i}' for i in range(num_players)]
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

# 3.
# execution loop
trainer = Coordinator(memory_buffer_size=500,
                      reservoir_size=reservoir_size,
                      batch_size=batch_size,
                      vector_length=sum(num_cards) + max_bet_number + num_actions + 1,
                      num_actions=num_actions,
                      num_batches=number_batches,
                      output_dim=output_dim,
                      n_cards=num_cards,
                      flatten_func=flatten_data_for_memory,
                      memory_dir=f'memories_{model_type}-Model/',
                      result_dir=f'results_train_{model_type}-Model/')

trainer.deep_CFR(env_str, config_dict, CFR_start_itartion, CFR_iterations,
                 num_traversals, num_players, runner_kwargs, num_runners=num_cpus)
