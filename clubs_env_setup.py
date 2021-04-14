import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import clubs_gym
from clubs_gym.agent.base import BaseAgent
import numpy as np
from Tensorflow_Model import get_DeepCFR_model
import tensorflow as tf


class Random_Agent(BaseAgent):
    """
    Agent that acts randomly and prints its Name and action.
    """
    def __init__(self, id):
        self.id = id
        self.model = get_DeepCFR_model(256, [2, 3, 1, 1], 4, 3)

    def act(self, obs, strategy):
        # action = np.random.randint(low=-1, high=5)
        hole_cards = tf.constant([[[1], [10]]], dtype=tf.float32)
        flop = tf.constant([[[2], [9], [8]]], dtype=tf.float32)

        cards_inp = [hole_cards, flop]
        bets = tf.constant([[1, 2, 2, 4]], dtype=tf.float32)

        action = softmax(self.model([cards_inp, bets]).numpy()).argmax()

        breakpoint()
        print(f'agent-{self.id} action: {action}')
        return action


def convert_cards_to_id(cards):
    """
    Computes the unique card IDs for a list of cards with type clubs.card.
    Assumes a card deck of size 52 with ranks clubs (♣), diamonds (♦),
    hearts (♥) and spades (♠), and suits 2, 3, 4, 5, 6, 7, 8, 9, 10, B, D, K, A.

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

        converted_cards.append(card_id)

    return converted_cards


# Create HU LHE (1, 2) environment
config_dict = {'num_players': 2,
               'num_streets': 4,
               'blinds': [1, 2],
               'antes': 0,
               'raise_sizes': [2, 2, 4, 4],
               'num_raises': 4,
               'num_suits': 4,
               'num_ranks': 13,
               'num_hole_cards': 2,
               'mandatory_num_hole_cards': 0,
               'num_community_cards': [0, 3, 1, 1],
               'start_stack': 200,
               'num_cards_for_hand': 5}

clubs_gym.envs.register({"noLimit_TH-v0": config_dict})
env = gym.make("noLimit_TH-v0")

# Pass agents with internal policy/strategy to the env (dealer object)
env.register_agents([Random_Agent(i) for i in range(2)])

# iterate over games
for i in range(100):
    obs = env.reset()
    env.render()

    counter = 0
    while True:
        counter += 1
        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)
        env.render()

        card = convert_cards_to_id(obs['hole_cards'])
        print(card, obs["hole_cards"])

        if all(done):
            break
