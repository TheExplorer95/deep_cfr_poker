import tensorflow as tf
from clubs_gym.agent.base import BaseAgent
import tensorflow_probability as tfp

class TensorflowAgent(BaseAgent):
    def __init__(self, model_path):
        super().__init__()

        # instantiate tensorflow model (functional api if possible)
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # define possible actions (implicit in model structure)

    def act(self, info_state, strategy = False):
        # betsizes, cards = obs[.....]
        # tf_model(input, output)
        # actions = tf_model((betsizes, cards)) -> output: integer
        # sampling_arg (softmax(actions)) ---> 0 for fold, 1, 2, 3, 4 (bet/raise size meaning?)
        # a ~ [0, 0.1, 0.9] a.shape -> [0,0,1], [0,1,0]
        # 0,0 in bet size history means (check, check -> next round). 3,3 means (bets 3, calls 3 -> next round).
        # 3,6,6 means (bets 3, raises to 6, calls 6)

        ### assuming the observation is a tuple : cards = [[51, 15], [-1, -1 ,-1], ..]; bet_history = [10, 10, 0, 0, 23, -1,-1,-1,-1,...]

        # GET MODEL INPUTS
        # state preflop: cards = [[1,2],[-1,-1,-1],[-1],[-1]]
        # state flop: ...

        # state preflop: bets = [0,0, ...... (max number of future bets)]
        #                       [0,0, -1,-1,-1,-1]

        action_advantages = self.model(info_state)

        ### Softmax on action action_advantages
        action_probabilities = tf.nn.softmax(action_advantages)

        if strategy:
            return action_probabilities

        dist = tfp.distributions.Categorical(probs=action_probabilities.numpy()[0])
        sampled_action = dist.sample().numpy()
        #print(f'action{sampled_action}')
        # sample from resulting probability distribution

        #action = 0 # (sampled result from action distribution)

        return sampled_action
