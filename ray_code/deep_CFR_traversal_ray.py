import random
import time
import clubs_gym
import gym
import os
import numpy as np
import ray
from random import shuffle
# import tensorflow_probability as tfp
from copy import deepcopy, copy
from utils_ray import get_info_state
from memory_utils import MemoryWriter


def save_to_memory(type, player, info_state, iteration, values):
    """This function saves stuff to memory"""
    pass

    # TODO
    # bring info state and values and iteration into a useful data structure
    # save this data structure, e.g. a dict (for instance each iteration gets its own file?)


class Coordinator:
    """
    Coordinates acquestion, saving and training for deep CFR.
    """

    def __init__(self, memory_buffer_size, reservoir_size, vector_length, flatten_func, memory_dir):
        self.advantage_memory_0 = []
        self.advantage_memory_1 = []
        self.strategy_memory = []

        self.memory_buffer_size = memory_buffer_size

        self.initialize_memory_writers(reservoir_size, vector_length,
                                       flatten_func, memory_dir)

    def initialize_memory_writers(self, reservoir_size, vector_length,
                                  flatten_func, memory_dir):
        if not os.path.isdir(memory_dir):
            os.mkdir(memory_dir)

        mem_fn = os.path.join(memory_dir, 'advantage_memory_0.h5')
        self.advantage_writer_0 = MemoryWriter(reservoir_size,
                                               vector_length,
                                               flatten_func,
                                               mem_fn)

        mem_fn = os.path.join(memory_dir, 'advantage_memory_1.h5')
        self.advantage_writer_1 = MemoryWriter(reservoir_size,
                                               vector_length,
                                               flatten_func,
                                               mem_fn)

        mem_fn = os.path.join(memory_dir, 'strategy_memory.h5')
        self.strategy_writer = MemoryWriter(reservoir_size,
                                            vector_length,
                                            flatten_func,
                                            mem_fn)

    def extract_data_from_runner(self, runner, player):
        adv, strat = ray.get(runner.get_memories.remote())
        if player == 0:
            self.advantage_memory_0.extend(adv)
        elif player == 1:
            self.advantage_memory_1.extend(adv)
        else:
            print('[ERROR] - Player {player}, is not implemented.')

        self.strategy_memory.extend(strat)

    def save_memory_to_files(self, player, force_save=False):
        if player == 0:
            if len(self.advantage_memory_0) >= self.memory_buffer_size or force_save:
                self.advantage_writer_0.save_to_memory(self.advantage_memory_0)
                del self.advantage_memory_0
                self.advantage_memory_0 = []

                self.strategy_writer.save_to_memory(self.strategy_memory)
                del self.strategy_memory
                self.strategy_memory = []

        elif player == 1:
            if len(self.advantage_memory_1) >= self.memory_buffer_size or force_save:
                self.advantage_writer_1.save_to_memory(self.advantage_memory_1)
                del self.advantage_memory_1
                self.advantage_memory_1 = []

                self.strategy_writer.save_to_memory(self.strategy_memory)
                del self.strategy_memory
                self.strategy_memory = []

        else:
            print('[ERROR] - Player {player}, is not implemented.')

    def deep_CFR(self, env_str, config_dict, CFR_iterations, num_traversals, num_players,
                 runner_kwargs, num_runners):
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
        num_saves = []
        times = []
        runners = [Traversal_Runner.remote(i, env_str, **runner_kwargs) for i in range(num_runners)]

        for t in range(CFR_iterations):
            for p in range(num_players):

                t1 = time.time()

                # collect data from env via MonteCarlo style external sampling

                futures = [runner.traverse.remote(history=[],
                                                  traverser=p,
                                                  CFR_iteration=t,
                                                  action='first') for runner in runners]

                traversal_counter = num_runners  # have been already started
                while futures:
                    ids, futures = ray.wait(futures)

                    if traversal_counter < num_traversals:
                        runner_ID = ray.get(ids[0])[1]
                        saves = ray.get(runners[runner_ID].get_counter.remote())
                        self.extract_data_from_runner(runners[runner_ID], p)
                        self.save_memory_to_files(p)
                        future = runners[runner_ID].traverse.remote(history=[],
                                                                    traverser=p,
                                                                    CFR_iteration=t,
                                                                    action='first')

                        traversal_counter += 1
                        futures.append(future)
                        num_saves.append(saves)

                # final cleanup of the runners and memories
                for runner in runners:
                    self.extract_data_from_runner(runner, p)
                self.save_memory_to_files(p, force_save=True)

                # fancy output and stats stuff
                dt = time.time() - t1
                times.append(dt)

                print(f'adv_0: {self.advantage_writer_0.counter[1]}, adv_1: {self.advantage_writer_1.counter[1]}, strat: {self.strategy_writer.counter[1]}')
                print(f"[CFR_iteration - {t}, player - {p}] d_t: {times[-1]:.4f}s, total_t: {np.sum(times):.4f}s, info_state_saves: {np.sum(num_saves)}")

                # initialize new value network (if not first iteration) and train with val_mem_p
                # (only used for prediction of the next regret values)
                # train_val_net()

            # train the strat_net with strat_mem


@ray.remote
class Traversal_Runner:
    """
    Used for the distributed sampling of CFR_data
    """
    def __init__(self, ID, env_str, config_dict, max_bet_number,
                 model_save_paths=None, agent_fct=None):
        self.ID = ID
        self.env_str = env_str
        self.config_dict = config_dict
        self.counter = 0
        self.max_bet_number = max_bet_number
        self.strategy_memory = []
        self.advantage_memory = []
        self.create_env(model_save_paths, agent_fct)

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

    def get_memories(self):
        advantage, strategy = self.advantage_memory, self.strategy_memory
        del self.advantage_memory, self.strategy_memory
        self.advantage_memory = []
        self.strategy_memory = []

        return advantage, strategy

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

            # randomize which player starts
            if random.randint(0, 1):
                obs = self.env.reset()

        elif isinstance(action, (int, np.int32)):
            # take given action within the environment
            obs, reward, done, _ = self.env.step(action)

        else:
            print(f'[ERROR] - {action} is not a valid action.')

        mode = self.env.dealer.num_streets

        # ---------- start MonteCarlo sampling ---------------------
        # 1.
        # Terminal Node
        state_terminal = not all(obs['active']) or obs["action"] == -1
        if state_terminal:
            # game state: end
            # calculate traversers payoff

            traverser_payoff = self.env.dealer._payouts()[traverser]
            return (traverser_payoff, self.ID)

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
                self.env = self.create_env_cpy(orig_env)
                history_cpy = deepcopy(history)  # copy bet size history
                history_cpy.append(a)  # add bet to bet history

                traverser_payoff = self.traverse(history_cpy, traverser, CFR_iteration, action=a)[0]
                values.append(traverser_payoff)

            self.env = orig_env

            # 3.3
            # use returned payoff for advantage/regret computation
            advantages = []
            for a in range(len(strategy.numpy()[0])):
                # compute advantages of each action
                advantages.append(values[a] - np.sum(strategy.numpy()[0] * np.array(values)))

            # 3.4
            # append Infoset, action_advantages and CFR_iteration t to advantage_mem_traverser
            cards = []
            for tensor in info_state[0]:
                cards.append(tensor.numpy())
            bet_hist = info_state[1].numpy()

            self.advantage_memory.append(([cards, bet_hist], CFR_iteration, advantages))
            self.counter += 1

            expected_infostate_value = np.sum(strategy.numpy()[0] * np.array(values))
            return (expected_infostate_value, self.ID)

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
            cards = []
            for tensor in info_state[0]:
                cards.append(tensor.numpy())
            bet_hist = info_state[1].numpy()
            self.strategy_memory.append(([cards, bet_hist], CFR_iteration, strategy.numpy()[0]))
            self.counter += 1

            # 3.
            # get action (softmax) according to strategy
            # dist = tfp.distributions.Categorical(probs=strategy.numpy())
            # sampled_action = dist.sample().numpy()
            action = self.env.act(info_state)
            history.append(action)
            return (self.traverse(history, traverser, CFR_iteration, action)[0], self.ID)

    def get_counter(self):
        return self.counter
