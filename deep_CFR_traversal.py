from copy import deepcopy, copy
from random import shuffle
from Tensorflow_Model import get_DeepCFR_model
from PokerAgent import TensorflowAgent
import clubs_gym
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from memory_utils import MemoryWriter, flatten_data_for_memory ###### <<<<<---------- NEW


def get_env_cpy(env):
    env_cpy = gym.make(env.unwrapped.spec.id)  # gets the gym environment name, e.g. "noLimit_TH-v0"
    env_cpy.reset()

    # 1. set previous obs
    env_cpy.prev_obs = deepcopy(env.prev_obs)

    # 2. make copy of the dealer and shuffle its deck (new dealer produces different community cards)
    env_cpy.dealer = deepcopy(env.dealer)
    shuffle(env_cpy.dealer.deck.cards)

    # 3. create reference to the original agents (new env uses old models for sampling)
    # creating new agents also possible, depends on computation overhead by ray:
    #                                     [TensorflowAgent(f'test_model_{i}') for i in range(2)]
    env_cpy.agents = copy(env.agents)

    return env_cpy

def get_history_cpy(orig_history):
    """copies a list"""

    return deepcopy(orig_history)

def convert_cards_to_id(cards):
    """
    Computes the unique card id for a clubs.cards type card. Assumes a card
    deck of size 52 with ranks clubs (♣), diamonds (♦), hearts (♥) and
    spades (♠), and suits 2, 3, 4, 5, 6, 7, 8, 9, 10, B, D, K, A.
    0 = duce of clubs; 1 = duce of diamonds ...
                                ... 50 = ace of hearts; 51 ace of spades
    Parameters
    ----------
    cards: list(clubs.card)
        List of clubs.card cards to convert.
    Returns
    -------
    converted_cards: list(card_ids)
    """

    # length of each card encoding (see class clubs.Card for reference)
    # prime = 8, bit_rank = 4
    oneHot_suit = 4
    oneHot_rank = 16
    oneHot_rank_offset = 3
    encoding_length = 32

    converted_cards = []
    for card in cards:
        # get card binary string (deals with shortening of string when
        # 32Bit int not 32Bit long; python truncates higher_order bits in string
        # representation when not flipped)
        card_binary = '0'*(encoding_length-len(card._bin_str)) + card._bin_str

        # extract rank and suit
        card_suit = card_binary[oneHot_rank:oneHot_rank+oneHot_suit].find('1')
        card_rank = card_binary[oneHot_rank_offset:oneHot_rank][::-1].find('1')

        # compute card id
        card_id = card_rank * 4 + card_suit

        converted_cards.append([card_id])

    return converted_cards

def get_info_state(obs, history, max_bet_number, mode):
    """ Transforms the observation dictionary from clubs env and the history
        list to an info state (input to the ANN model)"""

    bet_history = get_history_cpy(history)
    h_cards = obs["hole_cards"]
    c_cards = obs["community_cards"]

    # convert hole cards to indices.
    hole_cards = [convert_cards_to_id(h_cards)]

    # convert community cards to indices and split into flop, turn, river
    c_cards_len = len(c_cards)
    flop_cards = []
    turn = []
    river = []

    if c_cards_len:
        c_cards = convert_cards_to_id(c_cards)
        if c_cards_len >= 3:
            flop_cards = c_cards[:3]

        if c_cards_len >= 4:
            turn = c_cards[3]

        if c_cards_len ==5:
            river = c_cards[4]

    # padding of bet history
    while len(bet_history) < max_bet_number:
        bet_history.append(-1)

    bet_history = tf.constant([bet_history], dtype = tf.float32)
    hole_cards = tf.constant(hole_cards, dtype = tf.float32)


    # padding for not yet given cards
    # only hole cards
    if mode == 1:
        output = [[hole_cards], bet_history]

    # flop only
    elif mode == 2:
        # if no flop card is given, use no-card index (-1)
        if not len(flop_cards):
            flop_cards = tf.constant([
            [[-1], [-1], [-1]]
            ], dtype= tf.float32)

#         [[-1] for i in range(len(flop_cards))]
        else:
            flop_cards = tf.constant([flop_cards], dtype = tf.float32)

        output = [[hole_cards, flop_cards], bet_history]



    # three streets (hole, flop, turn)
    elif mode == 3:

        if not len(flop_cards):
            flop_cards = tf.constant([
            [[-1], [-1],[-1]]
            ], dtype= tf.float32)
        else:
            flop_cards = tf.constant([flop_cards], dtype = tf.float32)

        if not len(turn):
            turn = tf.constant([
            [[-1]]
            ], dtype = tf.float32)
        else:
            turn = tf.constant([turn], dtype = tf.float32)

        output = [[hole_cards, flop_cards, turn], bet_history]

    # 4 streets (full poker)
    elif mode == 4:
        if not len(flop_cards):
            flop_cards = tf.constant([
            [[-1], [-1],[-1]]
            ], dtype= tf.float32)

        else:
            flop_cards = tf.constant([flop_cards], dtype = tf.float32)

        if not len(turn):
            turn = tf.constant([
            [[-1]]
            ], dtype = tf.float32)
        else:
            turn = tf.constant([turn], dtype = tf.float32)

        if not len(river):
            river = tf.constant([
            [[-1]]
            ], dtype = tf.float32)
        else:
            river = tf.constant(river, dtype = tf.float32)

        output = [[hole_cards, flop_cards, turn, river], bet_history]

    return output

def deep_CFR(CFR_iterations, num_traversals, num_players):
    global strategy_memory, advantage_memory
    """
    Parameters
    ----------
    env : gym.env instance
          The game to optimize a strategy for.


    CFR_iterations : int
                     Number of times deepCFR is applied.

    Returns
    -------
    strat_net : tf.keras.model class
                The trained strategy network."""

    # initialize ANNs and memories
    num_players = 2
    num_streets = 1
    num_raises = 3

    output_dim = 256 # model

    n_cards = [2]
    n_community_cards = [0] #+ n_cards[1:]

    n_cards_for_hand = min(5, sum(n_cards))

    max_bet_number = n_bets = num_players * num_streets * num_raises
    n_actions = 5

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use cpu instead of gpu
    cfr_models = [get_DeepCFR_model(output_dim, n_cards, n_bets, n_actions) for i in range(num_players)] # 2 agents

    [model.save(f'test_model_{i}') for i, model in enumerate(cfr_models)]
    # reset environment, obtain obs,
    #Create HU LHE (1, 2) environment
    config_dict = {'num_players': num_players,
                   'num_streets': num_streets,
                   'blinds': [1, 2],
                   'antes': 0,
                   'raise_sizes': [2],
                   'num_raises': num_raises,
                   'num_suits': 4,
                   'num_ranks': 13,
                   'num_hole_cards': n_cards[0],
                   'mandatory_num_hole_cards': 0,
                   'num_community_cards': n_community_cards,
                   'start_stack': 1_000_000,
                   'num_cards_for_hand': n_cards_for_hand}

    clubs_gym.envs.register({"limit_easyHoldem-v0": config_dict})


    # Pass agents with internal policy/strategy to the env (dealer object)

    #env.register_agents([TensorflowAgent(f'test_model_{i}') for i in range(2)]) # 2 because two players

    StrategyWriter = MemoryWriter(max_size = 100_000, vector_length = 14, flatten_func = flatten_data_for_memory, file_name = "strategy_memory.h5")
    AdvantageWriter = MemoryWriter(max_size = 100_000, vector_length = 14, flatten_func = flatten_data_for_memory, file_name ="advantage_memory.h5")

    for t in range(CFR_iterations):

        for p in range(num_players):
            # reload trained agent models
            env = gym.make("limit_easyHoldem-v0")
            env.register_agents([TensorflowAgent(f'test_model_{i}') for i in range(2)]) # 2 because two players
            advantage_memory = []   # for RAY this should be done inside each runner as well !!!
            strategy_memory = []

            for k in range(num_traversals):
                # collect data from env via external sampling
                if not k%1000:
                    print(f"{k} completed")
                obs = env.reset()
                history = []

                ### Multiple runner run traverse in parallel and each should send back a filled advantage_memory and strategy_memory list
                traverse(env, obs, history, p, t)

                # concatenate all advantage_memory lists and all strategy_memory lists respectively


                # write to disk every 30_000 info_states
                if len(advantage_memory) > 10000:
                    AdvantageWriter.save_to_memory(advantage_memory)
                    advantage_memory = []

                if len(strategy_memory) > 10000:
                    StrategyWriter.save_to_memory(strategy_memory)
                    strategy_memory = []

            print("players switch")

            # initialize new value network (if not first iteration) and train with val_mem_p
            # (only used for prediction of the next regret values)
            #train_val_net()

        # train the strat_net with strat_mem
        #train_strat_net()

    return 0




def traverse(env, obs, history, traverser, CFR_iteration):
    """

    Following the pseudocode from [DeepCFR]


    # input(history, traverser, val_model_0, val_model_1, val_mem_trav, strat_mem, t)



    Parameters
    ----------
    env : clubs_gym env
        Has 2 agents in its dealer.
        Each agent has a value network (val_net_p1, val_net_p2)

    history : list
        Betting history [10, 10, 0, 0, 23, -1,-1,-1,-1, ...]
        Each value indicated the amount of money played in one turn.
        0 indicates a check or a fold. 1 - inf indicates a check or a call
        (can be deduced from previous entry).

    traverser : int()
        Heads up -> 0 or 1

    val_net_p1 :

    val_net_p2 :

    val_mem_traverser :

    strat_mem :

    CFR_iteration :


    Returns
    -------
    None

    """


    global counter, strat_counter

    global advantage_memory, strategy_memory
    mode = env.dealer.num_streets
    max_bet_number = 6 # env.dealer.num_raises * env.dealer.num_players * mode

    state_terminal = not all(obs['active']) or obs["action"] == -1
    if state_terminal:
        # game state: end
        # calculate traversers payoff
        traverser_payoff = env.dealer._payouts()[traverser]    # - obs['street_commitments'][traverser] # gain or loss
        #print("payoff traverser", traverser_payoff)
        return traverser_payoff

    # chance node
    elif False:
        # game state: the dealer has to hand out cards or turn the river
        # does not count into traversal depth
        # !!chance nodes are automatically handled by environment when taking
        # an action!!
        pass

    elif obs['action'] == traverser:

        # game state: traverser has to take an action

        # 1.
        # compute strategy (next action) from the Infoset of the traverser and
        # his val_net via regret matching (used for weighting when calculating
        # the advantages)
        # call model on observation, no softmax
        info_state = get_info_state(obs, history, max_bet_number, mode)
        strategy = env.agents[traverser].act(info_state, strategy=True)

        # 2.
        # iterate over all actions and do traversals starting from each actions
        # subsequent history
        values = []
        for a in range(len(strategy.numpy()[0])):
            # cpy environment
            # take selected action within copied environment
            history_cpy = get_history_cpy(history) # copy bet size history
            env_cpy = get_env_cpy(env)
            obs, reward, done, _ = env_cpy.step(a)

            history_cpy.append(a) # add bet to bet history

            traverser_payoff = traverse( env_cpy, obs, history_cpy, traverser, CFR_iteration)
            values.append(traverser_payoff)

            #return traverser_payoff

        # 3.
        # use returned payoff for advantage/regret computation
        advantages = []

        for a in range(len(strategy.numpy()[0])):
            # compute advantages of each action
            advantages.append(values[a] - np.sum(strategy.numpy()[0] * np.array(values)))

        # 4.
        # append Infoset, action_advantages and CFR_iteration t to advantage_mem_traverser

        cards = []
        for tensor in info_state[0]:
            cards.append(tensor.numpy())
        bet_hist = info_state[1].numpy()


        advantage_memory.append(([cards, bet_hist], CFR_iteration, advantages))

        expected_infostate_value = np.sum(strategy.numpy()[0] * np.array(values))

        return expected_infostate_value
    else:
        # game state: traversers opponent has to take an action

        # 1.
        # compute strategy (next action) from the Infoset of the opponent and his
        # val_net via regret matching
        # call model on observation, no softmax
        info_state = get_info_state(obs, history, max_bet_number, mode)

        non_traverser = 1 - traverser

        strategy = env.agents[non_traverser].act(info_state, strategy=True) # env.act(orig_obs, strategy = True) probably is what works

        # 2.
        # append Infoset, action_probabilities and CFR_iteration t to strat_mem
        cards = []
        for tensor in info_state[0]:
            cards.append(tensor.numpy())
        bet_hist = info_state[1].numpy()

        strategy_memory.append(([cards, bet_hist], CFR_iteration, strategy.numpy()[0]))
        # 3.
        # copy env and take action according to action_probabilities
        dist = tfp.distributions.Categorical(probs = strategy.numpy())

        sampled_action = dist.sample().numpy()
        action = env.act(info_state)
        obs, reward, done, _ = env.step(action)
        # update history
        history.append(action)
        return traverse(env, obs, history, traverser, CFR_iteration)


## test traverse
# create environment

CFR_iterations = 1
num_traversals = 10_000
num_players = 2

strategy_memory = []
advantage_memory = []

deep_CFR(CFR_iterations, num_traversals, num_players)

"""
num_players = 2
num_streets = 1
num_raises = 3


output_dim = 256 # model

n_cards = [2]
n_community_cards = [0] #+ n_cards[1:]

n_cards_for_hand = min(5, sum(n_cards))

max_bet_number = n_bets = num_players * num_streets * num_raises
n_actions = 5


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use cpu instead of gpu
cfr_models = [get_DeepCFR_model(output_dim, n_cards, n_bets, n_actions) for i in range(num_players)] # 2 agents

[model.save(f'test_model_{i}') for i, model in enumerate(cfr_models)]
# reset environment, obtain obs,
#Create HU LHE (1, 2) environment
config_dict = {'num_players': num_players,
               'num_streets': num_streets,
               'blinds': [1, 2],
               'antes': 0,
               'raise_sizes': [2],
               'num_raises': num_raises,
               'num_suits': 4,
               'num_ranks': 13,
               'num_hole_cards': n_cards[0],
               'mandatory_num_hole_cards': 0,
               'num_community_cards': n_community_cards,
               'start_stack': 1_000,
               'num_cards_for_hand': n_cards_for_hand}

clubs_gym.envs.register({"limit_easyHoldem-v0": config_dict})
env = gym.make("limit_easyHoldem-v0")

# Pass agents with internal policy/strategy to the env (dealer object)

env.register_agents([TensorflowAgent(f'test_model_{i}') for i in range(2)]) # 2 because two players
import time
times = []
traversals = 100
#counter = 0
for i in range(traversals):

    counter = 0
    strat_counter = 0
    t1 = time.time()
    obs = env.reset()

    history = []
    traverser = 0
    CFR_iteration = 1

    traverse(env, obs, history, traverser, CFR_iteration)
    dt = time.time() - t1
    times.append(dt)
    print(f"{i+1}/100 done in {dt:.6f} seconds! Number of values appended: {counter}. Strategy memory filled: {strat_counter} times")
print(f"{np.mean(times)} seconds on average for {traversals} traversals.")
"""
