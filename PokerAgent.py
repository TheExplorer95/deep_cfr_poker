import random
import tensorflow as tf
from clubs_gym.agent.base import BaseAgent


class TensorflowAgent(base.BaseAgent):
    def __init__(self, model_path, agent_parameters):
        super().__init__()

        # instantiate tensorflow model (functional api if possible)
        self.model = tf.keras.models.load_model(
        model_path, custom_objects=None,
        compile=True, options=None
        )

        # define possible actions (implicit in model structure)


    def act(self, obs):
        # betsizes, cards = obs[.....]
        # tf_model(input, output)
        # actions = tf_model((betsizes, cards)) -> output: integer
        # sampling_arg (softmax(actions)) ---> 0 for fold, 1, 2, 3, 4 (bet/raise size meaning?)
        # a ~ [0, 0.1, 0.9] a.shape -> [0,0,1], [0,1,0]
        # 0,0 in bet size history means (check, check -> next round). 3,3 means (bets 3, calls 3 -> next round).
        # 3,6,6 means (bets 3, raises to 6, calls 6)

        ### assuming the observation is a tuple : cards = (52, 15, -1, -1 , ..); bet_history = (10, 1, 0, 0, 23, 0,0,0)

        cards, bet_history = obs

        # GET MODEL INPUTS
        # state preflop: cards = [[1,2],[-1,-1,-1],[-1],[-1]]
        # state flop: ...

        # state preflop: bets = [0,0, ...... (max number of future bets)]
        #                       [0,0, -1,-1,-1,-1]

        cards = tf.constant(cards, dtype= tf.float32)
        bet_history = tf.constant(bet_history, dtype = tf.float32)

        action_advantages = self.model(cards, bet_history)

        ### Softmax on action action_advantages

        # sample from resulting probability distribution

        action = 0 # (sampled result from action distribution)

        return action
