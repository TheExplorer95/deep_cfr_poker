import tensorflow as tf
from clubs_gym.agent.base import BaseAgent
import tensorflow_probability as tfp

class TensorflowAgent(BaseAgent):
    def __init__(self, model_path):
        super().__init__()

        # instantiate tensorflow model (functional api if possible)
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def act(self, info_state, strategy = False):

        action_advantages = self.model(info_state)

        ### Softmax on action action_advantages
        action_probabilities = tf.nn.softmax(action_advantages)

        if strategy:
            return action_probabilities

        dist = tfp.distributions.Categorical(probs=action_probabilities.numpy()[0])
        sampled_action = dist.sample().numpy()

        return sampled_action
