import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import ray
from utils_ray import activate_memory_growth
from deep_CFR_traversal_ray import deep_CFR
from Tensorflow_Model import get_DeepCFR_model
from PokerAgent import TensorflowAgent


# ------------------- initialization stuff -------------------------------
# for ray backend
ray.init(logging_level=logging.INFO)

# might be impoortant for model reinitialization during traversal
activate_memory_growth(cpu = True)


# -------------------- The Algorithm -------------------------------------
# 1.
# Set algorithm parameters
num_runners = 10
num_traversals = 100
CFR_iterations = 1


# Set agent
agent_fct = TensorflowAgent

# Set game parameters
env_str = 'limit_easyHoldem-v0'
num_players = 2
num_streets = 2
num_raises = 3
n_actions = 5
n_cards = [2, 3]

n_community_cards = [0] + n_cards[1:]
n_cards_for_hand = min(5, sum(n_cards))
max_bet_number = num_players * num_streets * num_raises

output_dim = 256  # model for card embeddings

# environment params dict
config_dict = {'num_players': num_players,
               'num_streets': num_streets,
               'blinds': [1, 2],
               'antes': 0,
               'raise_sizes': [2, 4],
               'num_raises': num_raises,
               'num_suits': 4,
               'num_ranks': 13,
               'num_hole_cards': n_cards[0],
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

cfr_models = [get_DeepCFR_model(output_dim, n_cards, max_bet_number, n_actions)
              for _ in range(num_players)]
              
saves = [model.save(fn) for fn, model in zip(model_save_paths, cfr_models)]

agents = [agent_fct(p) for p in model_save_paths]
# Pass agents with internal policy/strategy to the env (dealer object)
# env.register_agents([TensorflowAgent(f'test_model_{i}') for i in range(num_players)])

runner_kwargs = {'model_save_paths': model_save_paths,
                 'agent_fct': agent_fct,
                 'config_dict': config_dict,
                 'max_bet_number': max_bet_number}


# 3.
# execution loop
deep_CFR(env_str, config_dict, CFR_iterations, num_traversals, num_players, num_runners,
         runner_kwargs)
