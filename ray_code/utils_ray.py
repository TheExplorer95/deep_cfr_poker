from copy import deepcopy, copy
from random import shuffle
import tensorflow as tf
import gym
import os


def print_obs(obs, num_suits):
    """
    Only for heads up poker, assumes 2 player.
    """
    # hole_cards = convert_cards_to_id(obs['hole_cards'], num_suits)
    # community_cards = convert_cards_to_id(obs['community_cards'], num_suits)
    print('--------- game Stats ---------')
    print(f'community_cards: {obs["community_cards"]}')
    print(f'Dealer/Button: {obs["button"]%2}')
    print(f'Pot: {obs["pot"]}')
    print(f'Commit: player_0 {obs["street_commits"][0]}, player_1 {obs["street_commits"][1]}')

    if not obs['action'] == -1:
        print(f"\n--------- player {obs['action']}'s turn ---------")
        print(f'hole_cards: {obs["hole_cards"]}')
        print(f'min_raise: {obs["min_raise"]}, max_raise: {obs["max_raise"]}')
        print(f'call: {obs["call"]}', end='\n\n')
    else:
        print('\n[INFO] - End of the Game.')


def activate_memory_growth(cpu: bool):
    """
    Sets the desired device for Tensorflow computations
    """
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('[INFO] - CPU only computations activated.')
    # allows for GPU memory growth

    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print('[INFO] - Models trained on GPU, with memory growth activated.')
        except Exception:
            print('[INFO] - Cannot activate memory growth, now this program uses all the available GPU memory.')


def get_env_cpy(env, initialize_new_model=False):
    """
    Creates a copy of the given clubs_gym envyronment. The dealers cards are
    shuffled to ensure randomness at chance nodes (where the dealer hands out
    the flop, river and street) and the reference for the model is creted
    to the same model from before.
    """
    env_cpy = gym.make(env.unwrapped.spec.id)
    env_cpy.reset()

    # 1. set previous obs
    env_cpy.prev_obs = deepcopy(env.prev_obs)

    # 2. make copy of the dealer and shuffle its deck
    # (new dealer produces different community cards)
    env_cpy.dealer = deepcopy(env.dealer)
    shuffle(env_cpy.dealer.deck.cards)

    # 3. create reference to the original agents (new env uses old models for sampling)
    # creating new agents also possible, depends on computation overhead by ray:
    #                                     [TensorflowAgent(f'test_model_{i}') for i in range(2)]
    env_cpy.register_agents(copy(env.agents))

    return env_cpy


def convert_cards_to_id(cards, num_suits):
    """
    Computes the unique card id for a clubs.cards type card. Assumes a card
    deck of size 52 with ranks spades (♠), hearts (♥), diamonds (♦), clubs (♣)
    and suits A, K, D, B, 10, 9, 8, 7, 6, 5, 4, 3, 2.
    0 = ace of spades; 1 = King of spades ...
                                ... 50 = three of ; 51 = duce of clubs
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
    oneHot_rank = 13
    oneHot_suit = 4
    oneHot_rank_offset = 3
    begin_card_suit = oneHot_rank + oneHot_rank_offset
    encoding_length = 32

    converted_cards = []
    for card in cards:
        # get card binary string (deals with shortening of string when
        # 32Bit int not 32Bit long; python truncates higher_order bits in string
        # representation when not flipped)
        card_binary = '0'*(encoding_length-len(card._bin_str)) + card._bin_str

        # extract rank and suit then extract
        card_suit = card_binary[begin_card_suit:begin_card_suit+oneHot_suit][::-1].find('1')
        card_rank = card_binary[oneHot_rank_offset:oneHot_rank_offset+oneHot_rank].find('1')

        # compute card id
        card_id = card_rank * num_suits + card_suit
        converted_cards.append([card_id])

    return converted_cards


def get_info_state(obs, history, max_bet_number, mode, env_config):
    """ Transforms the observation dictionary from clubs env and the history
        list to an info state (input to the ANN model)"""

    bet_history = deepcopy(history)
    h_cards = obs["hole_cards"]
    c_cards = obs["community_cards"]

    # convert hole cards to indices.
    hole_cards = [convert_cards_to_id(h_cards,
                                      num_suits=env_config['num_suits'])]

    # convert community cards to indices and split into flop, turn, river
    c_cards_len = len(c_cards)
    flop_cards = []
    turn = []
    river = []

    if c_cards_len:
        c_cards = convert_cards_to_id(c_cards,
                                      num_suits=env_config['num_suits'])
        if c_cards_len >= 3:
            flop_cards = c_cards[:3]

        if c_cards_len >= 4:
            turn = c_cards[3]

        if c_cards_len ==5:
            river = c_cards[4]

    # padding of bet history
    while len(bet_history) < max_bet_number:
        bet_history.append(-1)

    bet_history = tf.constant([bet_history], dtype=tf.float32)
    hole_cards = tf.constant(hole_cards, dtype=tf.float32)

    # padding for not yet given cards
    # only hole cards
    if mode == 1:
        output = [[hole_cards], bet_history]

    # flop only
    elif mode == 2:
        # if no flop card is given, use no-card index (-1)
        if not len(flop_cards):
            flop_cards = tf.constant([[[-1], [-1], [-1]]], dtype=tf.float32)
        else:
            flop_cards = tf.constant([flop_cards], dtype=tf.float32)

        output = [[hole_cards, flop_cards], bet_history]

    # three streets (hole, flop, turn)
    elif mode == 3:
        if not len(flop_cards):
            flop_cards = tf.constant([[[-1], [-1], [-1]]], dtype=tf.float32)
        else:
            flop_cards = tf.constant([flop_cards], dtype=tf.float32)

        if not len(turn):
            turn = tf.constant([[[-1]]], dtype=tf.float32)
        else:
            turn = tf.constant([turn], dtype=tf.float32)

        output = [[hole_cards, flop_cards, turn], bet_history]

    # 4 streets (full poker)
    elif mode == 4:
        if not len(flop_cards):
            flop_cards = tf.constant([[[-1], [-1], [-1]]], dtype=tf.float32)
        else:
            flop_cards = tf.constant([flop_cards], dtype=tf.float32)

        if not len(turn):
            turn = tf.constant([[[-1]]], dtype=tf.float32)
        else:
            turn = tf.constant([turn], dtype=tf.float32)

        if not len(river):
            river = tf.constant([[[-1]]], dtype = tf.float32)
        else:
            river = tf.constant(river, dtype=tf.float32)

        output = [[hole_cards, flop_cards, turn, river], bet_history]

    return output
