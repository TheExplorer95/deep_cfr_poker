import tensorflow as tf
from clubs_gym.agent.base import BaseAgent
import tensorflow_probability as tfp
from Tensorflow_Model import regret_matching


class TensorflowAgent(BaseAgent):
    def __init__(self, model_path):
        super().__init__()

        # instantiate tensorflow model (functional api if possible)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.bet_history = []

    def act(self, info_state, strategy=False, regr_matching=True):

        action_advantages = self.model(info_state)

        if regr_matching:
            ### Softmax on action action_advantages
            action_probabilities = regret_matching(action_advantages)

        if strategy:
            return action_probabilities

        dist = tfp.distributions.Categorical(probs=action_probabilities.numpy()[0])
        sampled_action = dist.sample().numpy()

        self.bet_history.append(int(sampled_action))
        return sampled_action

    def append_to_bet(self, action):
        self.bet_history.append(action)
