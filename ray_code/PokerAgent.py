import random
import tensorflow as tf
from clubs_gym.agent.base import BaseAgent
import tensorflow_probability as tfp
from Tensorflow_Model import regret_matching


class action_fct:
    """
    Function that converts the action taken by the bot to the actual bet
    made.

    Params:
    action_idx - int (output of the model)
    obs - dict (current observation)

    Returns:
    action - int (bet made by the agent)
    """

    def __init__(self, input_type):
        # input_type: what kind of input to await from the model
        self.input_type = input_type

    def __call__(self, action, obs):
        if self.input_type == 'action':
            # Mapping:
            # 0 = Fold, 1 = Call, 2 = min_raise, 3 = max_raise

            if action == 0:
                bet = 0
            elif action == 1:
                bet = obs['call']
            elif action == 2:
                bet = max(int(obs['min_raise']), int(obs['call']))
            elif action == 3:
                bet = max(int(obs['min_raise'])*2, int(obs['call']))
            else:
                print(f'[ERROR] - Your Network output ({action}) is not designed for the environment, change either your num_output node or the action_func!')
                raise ValueError

        elif self.type == 'bet':
            # Mapping
            # 0 = Fold, [1,2,...] = bet

            # raises the bet made to the min call size
            if action > 0 and action < int(obs['call']):
                action = min(int(obs['call']), self.max_bet_agent)

            # raises the bet made to the min raise size if not called
            elif int(obs['call']) == 0 and int(obs['call']) < action and int(obs['min_raise']) > action:
                action = min(int(obs['min_raise']), self.max_bet_agent)

        else:
            print('[ERROR] - {self.type} is no valid action function input.')
            raise

        return int(bet)


class CustomAgent(BaseAgent):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def act(self, obs):
        return self.action


class MinRaiseAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def act(self, obs):
        return max(int(obs['min_raise']), int(obs['call']))


class TensorflowAgent(BaseAgent):
    def __init__(self, model_path, regr_matching=True):
        super().__init__()

        # instantiate tensorflow model (functional api if possible)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.bet_history = []
        self.regr_matching = regr_matching

    def act(self, info_state, strategy=False):
        try:
            action_advantages = self.model(info_state)
        except Exception as e:
            print(str(e))
            breakpoint()

        # initialized as strategy or advantage network (strat net has already regr matching)
        if self.regr_matching:
            action_probabilities = regret_matching(action_advantages)

        if strategy:
            return action_probabilities
        else:
            dist = tfp.distributions.Categorical(probs=action_probabilities.numpy()[0])
            sampled_action = dist.sample().numpy()
            self.bet_history.append(int(sampled_action))

            return sampled_action

    def append_to_bet(self, action):
        self.bet_history.append(action)
