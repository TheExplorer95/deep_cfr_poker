import random
import time
import clubs_gym
import gym
import numpy as np
import ray
from random import shuffle
# import tensorflow_probability as tfp
from copy import deepcopy, copy
from utils_ray import get_info_state


def save_to_memory(type, player, info_state, iteration, values):
    """This function saves stuff to memory"""
    pass

    # TODO
    # bring info state and values and iteration into a useful data structure
    # save this data structure, e.g. a dict (for instance each iteration gets its own file?)


def deep_CFR(env_str, config_dict, CFR_iterations, num_traversals, num_players, num_runners,
             runner_kwargs):
    """
    Parameters
    ----------
    env : gym.env instance
          The game to optimize a strategy for.

    val_net : tf.keras.model class
              The advantage value network used to approximate the regret value for action
              taken and actions possible.

    strat_net : tf.keras.model class
                The strategy network used to approximate the average strategy at the end of
                each iteration of deepCFR

    CFR_iterations : int
                     Number of times deepCFR is applied.

    Returns
    -------
    strat_net : tf.keras.model class
                The trained strategy network."""
    # initialize memories
    saves = []
    times = []
    runners = [Traversal_Runner.remote(env_str, **runner_kwargs) for i in range(num_runners)]

    for t in range(CFR_iterations):
        for p in range(num_players):
            for k in range(num_traversals):
                # for k in range(num_traversals):
                t1 = time.time()

                # collect data from env via MonteCarlo style external sampling

                futures = [runner.traverse.remote(history=[],
                                                  traverser=p,
                                                  CFR_iteration=t,
                                                  action='first') for runner in runners]
                while futures:
                    ids, futures = ray.wait(futures)
                    # do additional stuff, like saving while other still sample

                futures = [runner.get_counter.remote() for runner in runners]
                while futures:
                    ids, futures = ray.wait(futures)

                counter = [ray.get(counter) for counter in ids]

                # fancy output and stats stuff
                dt = time.time() - t1
                times.append(dt)
                saves.extend(counter)

                print(f"traversals so far: - {k * num_runners} - d_t: {np.sum(times[-2:]):.4f}s, total_t: {np.sum(times):.4f}s, n_saved: {np.sum(saves[-2:])}, total_saved: {np.sum(saves)}")

            # initialize new value network (if not first iteration) and train with val_mem_p
            # (only used for prediction of the next regret values)
            # train_val_net()

        # train the strat_net with strat_mem
        # train_strat_net()


@ray.remote
class Traversal_Runner:
    """
    Used for the distributed sampling of CFR_data
    """
    def __init__(self, env_str, config_dict, max_bet_number,
                 model_save_paths=None, agent_fct=None):
        self.env_str = env_str
        self.config_dict = config_dict
        self.create_env(model_save_paths, agent_fct)
        self.counter = 0
        self.max_bet_number = max_bet_number

    def create_env(self, model_save_paths, agent_fct):
        """
        Creates an environment with new beginning state and new agent instances
        with models passed as filepaths.
        """

        clubs_gym.envs.register({self.env_str: self.config_dict})
        self.env = gym.make(self.env_str)

        # create new agents
        self.env.register_agents([agent_fct(model_save_path) for model_save_path in model_save_paths])

        # random dealer position
        self.env.dealer.button += 1 if random.randint(0, 1) == 0 else 0

    def create_env_cpy(self, orig_env):
        """
        Creates a copy of the given clubs_gym envyronment. The dealers cards are
        shuffled to ensure randomness at chance nodes (where the dealer hands out
        the flop, river and street) and the reference for the model is creted
        to the same model from before.
        """

        env_cpy = gym.make(self.env_str)

        # 1. set previous obs
        env_cpy.prev_obs = deepcopy(orig_env.prev_obs)

        # 2. make copy of the dealer and shuffle its deck
        # (new dealer produces different community cards)
        env_cpy.dealer = deepcopy(orig_env.dealer)
        shuffle(env_cpy.dealer.deck.cards)

        # 3. create reference to the original agents
        env_cpy.register_agents(copy(orig_env.agents))

        return env_cpy

    def traverse(self, history, traverser, CFR_iteration, action=None):
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

        # 0.
        # set and get parameters
        if action == 'first':
            self.counter = 0
            obs = self.env.reset()

            # more randomness for beginnings
            if random.randint(0, 1) == 0:
                obs = self.env.reset()

        elif isinstance(action, (int, np.int32)):
            # take given action within the environment
            obs, reward, done, _ = self.env.step(action)

        else:
            print(f'[ERROR] - {action} is not a valid action')


        mode = self.env.dealer.num_streets

        # ---------- start MonteCarlo sampling ---------------------
        # 1.
        # Terminal Node
        state_terminal = not all(obs['active']) or obs["action"] == -1
        if state_terminal:
            # game state: end
            # calculate traversers payoff
            traverser_payoff = self.env.dealer._payouts()[traverser]
            return traverser_payoff

        # 2.
        # Chance Node
        elif False:
            # game state: the dealer has to hand out cards or turn the river
            # does not count into traversal depth
            # !!chance nodes are automatically handled by environment when taking
            # an action!!
            pass

        # 3.
        # Traverser Node
        elif obs['action'] == traverser:
            # game state: traverser has to take an action

            # 3.1
            # compute strategy (next action) from the Infoset of the traverser and
            # his val_net via regret matching (used for weighting when calculating
            # the advantages)
            # call model on observation, no softmax
            info_state = get_info_state(obs, history, self.max_bet_number, mode)
            strategy = self.env.agents[traverser].act(info_state, strategy=True)

            # 3.2
            # iterate over all actions and do traversals starting from each actions
            # subsequent history
            values = []
            orig_env = self.env
            for a in range(len(strategy.numpy()[0])):
                # send out new ray worker to sample the traverser_payoff
                self.env = self.create_env_cpy(orig_env)
                history_cpy = deepcopy(history)  # copy bet size history
                history_cpy.append(a)  # add bet to bet history

                traverser_payoff = self.traverse(history_cpy, traverser, CFR_iteration, action=a)
                values.append(traverser_payoff)

            self.env = orig_env

            # 3.3
            # use returned payoff for advantage/regret computation
            advantages = []
            for a in range(len(strategy)):
                # compute advantages of each action
                advantages.append(values[a] - np.sum(strategy.numpy()[0] * np.array(values)))

            # 3.4
            # append Infoset, action_advantages and CFR_iteration t to advantage_mem_traverser
            save_to_memory(type="value",
                           player=traverser,
                           info_state=info_state,
                           iteration=CFR_iteration,
                           values=advantages)
            self.counter += 1

            expected_infostate_value = np.sum(strategy.numpy()[0] * np.array(values))

            return expected_infostate_value

        # 4.
        # Non-Traverser Node
        else:
            # game state: traversers opponent has to take an action

            # 1.
            # compute strategy (next action) from the Infoset of the opponent and his
            # val_net via regret matching
            # call model on observation, no softmax
            info_state = get_info_state(obs, history, self.max_bet_number, mode)
            non_traverser = 1 - traverser
            strategy = self.env.agents[non_traverser].act(info_state, strategy=True)

            # 2.
            # append Infoset, action_probabilities and CFR_iteration t to strat_mem
            save_to_memory(type="strategy",
                           player=non_traverser,
                           info_state=info_state,
                           iteration=CFR_iteration,
                           values=strategy.numpy())
            self.counter += 1

            # 3.
            # get action (softmax) according to strategy
            # dist = tfp.distributions.Categorical(probs=strategy.numpy())
            # sampled_action = dist.sample().numpy()
            action = self.env.act(info_state)
            history.append(action)
            return self.traverse(history, traverser, CFR_iteration, action)

    def get_counter(self):
        return self.counter
