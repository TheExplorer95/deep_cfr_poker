import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import psutil
import ray
from utils_ray import activate_memory_growth
from deep_CFR_traversal_ray import Coordinator
from Tensorflow_Model import get_DeepCFR_model
from PokerAgent import TensorflowAgent
from memory_utils import flatten_data_for_memory


# ------------------- initialization stuff -------------------------------
# for ray backend
num_cpus = psutil.cpu_count(logical=False)
ray.init(logging_level=logging.INFO)

# might be impoortant for model reinitialization during traversal
activate_memory_growth(cpu=False)


# -------------------- The Algorithm -------------------------------------
# 1.
# Set algorithm parameters
num_traversals = 16#_000
CFR_iterations = 2#8

if not num_traversals > num_cpus:
    # need less runners
    num_cpus = num_traversals

# Set agent
agent_fct = TensorflowAgent

# Set game parameters
env_str = 'LDRL-Poker-v0'
num_players = 2
num_streets = 1
num_raises = 3
num_actions = 3
num_cards = [2]

n_community_cards = [0] #+ num_cards[1:]
n_cards_for_hand = min(5, sum(num_cards))
max_bet_number = num_players * num_streets * num_raises

output_dim = 256  # model for card embeddings

# environment params dict
config_dict = {'num_players': num_players,
               'num_streets': num_streets,
               'blinds': [1, 2],
               'antes': 0,
               'raise_sizes': "pot",
               'num_raises': num_raises,
               'num_suits': 4,
               'num_ranks': 13,
               'num_hole_cards': num_cards[0],
               'mandatory_num_hole_cards': 0,
               'num_community_cards': n_community_cards,
               'start_stack': 1_000_000,
               'num_cards_for_hand': n_cards_for_hand}

# 2.
# create environment
# clubs_gym.envs.register({env_str: config_dict})
# env = gym.make(env_str)

# initialize value_networks
model_save_paths = [f'value_model_p_{i}' for i in range(num_players)]

cfr_models = [get_DeepCFR_model(output_dim, num_cards, max_bet_number, num_actions)
              for _ in range(num_players)]

saves = [model.save(fn) for fn, model in zip(model_save_paths, cfr_models)]

agents = [agent_fct(p) for p in model_save_paths]

runner_kwargs = {'model_save_paths': model_save_paths,
                 'agent_fct': agent_fct,
                 'config_dict': config_dict,
                 'max_bet_number': max_bet_number}


vector_length = sum(num_cards) + max_bet_number + num_actions + 1

# 3.
# execution loop
trainer = Coordinator(memory_buffer_size=500,
                      reservoir_size=1_000_000,
                      batch_size = 10_000,
                      vector_length=vector_length,
                      num_actions = num_actions,
                      num_batches = 3,#1000,
                      output_dim = 256,
                      n_cards = num_cards,
                      flatten_func=flatten_data_for_memory,
                      memory_dir='memories/')
trainer.deep_CFR(env_str, config_dict, CFR_iterations, num_traversals, num_players,
                 runner_kwargs, num_runners=num_cpus)
