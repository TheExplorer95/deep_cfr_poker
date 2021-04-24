import random
import time
import clubs_gym
import gym
import os
import numpy as np
import ray
import pickle
import tensorflow as tf
from random import shuffle
# import tensorflow_probability as tfp
from copy import deepcopy, copy
from tqdm import tqdm
from utils_ray import get_info_state
from memory_utils import MemoryWriter
from Tensorflow_Model import get_DeepCFR_model
from training_utils import get_tf_dataset


class Coordinator:
    """
    Coordinates acquestion, saving and training for deep CFR.
    """

    def __init__(self, memory_buffer_size, reservoir_size, batch_size, vector_length, num_actions, num_batches, output_dim, n_cards, flatten_func, memory_dir, result_dir):
        self.advantage_memory_0 = []
        self.advantage_memory_1 = []
        self.strategy_memory = []
        self.bet_memory = []
        self.reservoir_size = reservoir_size
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_batches = num_batches
        self.output_dim = output_dim
        self.n_cards = n_cards
        self.memory_buffer_size = memory_buffer_size
        self.memory_dir = memory_dir
        self.result_dir = result_dir

        self.check_dirs()
        self.initialize_memory_writers(reservoir_size, vector_length,
                                       flatten_func)

    def check_dirs(self):
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.memory_dir):
            os.mkdir(self.memory_dir)

    def initialize_memory_writers(self, reservoir_size, vector_length,
                                  flatten_func):

        mem_fn = os.path.join(self.memory_dir, 'advantage_memory_0.h5')
        self.advantage_writer_0 = MemoryWriter(reservoir_size,
                                               vector_length,
                                               flatten_func,
                                               mem_fn)

        mem_fn = os.path.join(self.memory_dir, 'advantage_memory_1.h5')
        self.advantage_writer_1 = MemoryWriter(reservoir_size,
                                               vector_length,
                                               flatten_func,
                                               mem_fn)

        mem_fn = os.path.join(self.memory_dir, 'strategy_memory.h5')
        self.strategy_writer = MemoryWriter(reservoir_size,
                                            vector_length,
                                            flatten_func,
                                            mem_fn)

    def extract_data_from_runner(self, runner, player):
        adv, strat, bet = ray.get(runner.get_memories.remote())
        if player == 0:
            self.advantage_memory_0.extend(adv)
        elif player == 1:
            self.advantage_memory_1.extend(adv)
        else:
            print('[ERROR] - Player {player}, is not implemented.')

        self.strategy_memory.extend(strat)
        self.bet_memory.extend(bet)

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

    def deep_CFR(self, env_str, config_dict, CFR_start_itartion, CFR_iterations, num_traversals, num_players,
                 runner_kwargs, num_runners):
        """
        Parameters
        ----------
        env_str : gym.env ID string
              The game to optimize a strategy for.

        CFR_iterations : int
                         Number of times deepCFR is applied.

        Returns
        -------
        strat_net : tf.keras.model class
                    The trained strategy network."""
        # initialize memories
        times = []
        runners = [Traversal_Runner.remote(i, env_str, **runner_kwargs) for i in range(num_runners)]

        print(f'[INFO] - Starting with CFR-itaration {CFR_start_itartion}.')
        for t in range(CFR_start_itartion, CFR_iterations+1):
            print(f'\n------------------------- CFR-Iteration {t} -------------------------')
            for p in range(num_players):

                # if t == 3 and p == 0:
                #     continue
                # elif t == 3 and p == 1:
                #     num_traversals = 8409

                t1 = time.time()

                # collect data from env via MonteCarlo style external sampling
                print(f'[Payer - {p}] - Started sampling.')
                futures = [runner.traverse.remote(history=[],
                                                  traverser=p,
                                                  CFR_iteration=t,
                                                  action='first') for runner in runners]

                traversal_counter = num_runners  # have been already started
                with tqdm(total=int(num_traversals), initial=traversal_counter,
                          unit='traversals', desc='') as fancy_print:

                    while futures:
                        ids, futures = ray.wait(futures)

                        # continues sending out runner until no traversals are left
                        if traversal_counter < num_traversals:
                            # get finised runner
                            runner_ID = ray.get(ids[0])[1]

                            # get data from runner
                            self.extract_data_from_runner(runners[runner_ID], p)
                            self.save_memory_to_files(p)

                            # send out runner
                            future = runners[runner_ID].traverse.remote(history=[],
                                                                        traverser=p,
                                                                        CFR_iteration=t,
                                                                        action='first')
                            futures.append(future)

                            # prints and stats
                            traversal_counter += 1
                            fancy_print.set_description(f'Mem_count: adv_0 {self.advantage_writer_0.counter[1]} adv_1 {self.advantage_writer_1.counter[1]} strat {self.strategy_writer.counter[1]} - Travers_prog',
                                                        refresh=True)
                            fancy_print.update(1)

                            if not traversal_counter % (num_traversals//5):
                                self.print_bet_mem()

                print(f'[Payer - {p}] - Finished sampling - Mem_count: adv_0 {self.advantage_writer_0.counter[1]} adv_1 {self.advantage_writer_1.counter[1]} strat {self.strategy_writer.counter[1]}.')

                # final cleanup of the runners and memories
                for runner in runners:
                    self.extract_data_from_runner(runner, p)
                self.save_memory_to_files(p, force_save=True)

                # fancy output and stats stuff
                dt = time.time() - t1
                times.append(dt)

                # initialize new value network (if not first iteration) and train with val_mem_p
                # (only used for prediction of the next regret values)
                # train_val_net()

                file_name = os.path.join(self.memory_dir, f"advantage_memory_{p}.h5")

                if p == 0:
                    num_infostates = min(self.advantage_writer_0.counter[1], self.reservoir_size)
                elif p == 1:
                    num_infostates = min(self.advantage_writer_1.counter[1], self.reservoir_size)

                num_cards = sum(self.n_cards)
                num_bets = runner_kwargs.get("max_bet_number")
                num_actions = self.num_actions

                train_ds = get_tf_dataset(file_name, self.batch_size, num_infostates, num_cards, num_bets, num_actions)

                print(f'[Player - {p}] - Started training the Advantage Network.')
                model = self.train_model_from_scratch(p, train_ds, self.n_cards, num_bets, num_actions, strategy=False, CFR_iteration=t)
                # set model weights for player p
                ray.get([runner.set_weights.remote(p, model.get_weights()) for runner in runners])

        print('\n------------------------- Started training the Strategy Network -------------------------')
        file_name = os.path.join(self.memory_dir, "strategy_memory.h5")
        num_infostates = min(self.strategy_writer.counter[1], self.reservoir_size)

        train_ds = get_tf_dataset(file_name, self.batch_size, num_infostates, num_cards, num_bets, num_actions)
        model = self.train_model_from_scratch(0, train_ds, self.n_cards, num_bets, num_actions, strategy=True, CFR_iteration=t)

        return model

    def print_bet_mem(self):
        counts = float(len(self.bet_memory))
        if counts == 0:
            counts = 1

        action_0 = self.bet_memory.count(0) / counts * 100
        action_1 = self.bet_memory.count(1) / counts * 100
        action_2 = self.bet_memory.count(2) / counts * 100
        action_3 = self.bet_memory.count(3) / counts * 100

        print(f'action_probs:   fold/check {action_0:.2f}   check/call {action_1:.2f}   min_raise {action_2:.2f}   max_raise {action_3:.2f}', end='\n\n')

    def train_model_from_scratch(self, player, dataset, n_cards, num_bets, num_actions, strategy, CFR_iteration):
        # load model
        model = get_DeepCFR_model(self.output_dim, n_cards, num_bets, num_actions, strategy)
        # train model
        if not strategy:
            lr = 0.001 * 2 / CFR_iteration
            architecture = f'advantage-network_player-{player}'
        else:
            lr = 0.001
            architecture = 'strategy-network'

        opt = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=opt)

        history = model.fit(dataset.take(self.num_batches))
        model_name = os.path.join(self.memory_dir, f"trained_{architecture}_CRF-iteration-{CFR_iteration}")
        model.save(model_name)

        # save loss
        fn_loss = os.path.join(self.result_dir, f'{architecture}_CFR-iteration-{CFR_iteration}_lossData.pkl')
        with open(fn_loss, 'wb') as f:
            pickle.dump(history.history, f)

        print(f'[INFO] - Successfully saved the model.')

        return model


@ray.remote
class Traversal_Runner:
    """
    Used for the distributed sampling of CFR_data
    """
    def __init__(self, ID, env_str, config_dict, max_bet_number, max_bet_agent,
                 model_save_paths=None, agent_fct=None):
        self.ID = ID
        self.env_str = env_str
        self.config_dict = config_dict
        self.max_bet_number = max_bet_number
        self.max_bet_agent = max_bet_agent
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

        bet_memory = []
        for i in range(2):
            bet_memory.extend(self.env.agents[i].bet_history)
            self.env.agents[i].bet_history = []

        return advantage, strategy, bet_memory

    def action_func(self, action, obs):
        # 0 = Fold, 1 = Call, 2 = min_raise, 3 = max_raise
        if action == 0:
            pass
        elif action == 1:
            action = obs['call']
        elif action == 2:
            action = obs['min_raise']
        elif action == 3:
            action = obs['max_raise']
        else:
            print(f'[ERROR] - Your Network output ({action}) is not designed for the environment, change either your num_output node or the action_func!')
            raise ValueError

        # 0 = Fold, [1,2,...] = bet

        # if action > 0 and action < obs['call']:
        #     action = min(int(obs['call']), self.max_bet_agent)
        #
        # elif obs['call'] == 0 and obs['call'] < action and obs['min_raise'] > action:
        #     action = min(int(obs['min_raise']), self.max_bet_agent)

        return int(action)

    def traverse(self, history, traverser, CFR_iteration, action=None):
        """
        Following the pseudocode from [DeepCFR]
        # input(history, traverser, val_model_0, val_model_1, val_mem_trav, strat_mem, t)

        Parametershistory
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


        CFR_iteration :


        Returns
        -------
        None

        """

        # 0.
        # set and get parameters
        if action == 'first':
            obs = self.env.reset()

            # randomize which player starts
            if random.randint(0, 1):
                obs = self.env.reset()

        elif isinstance(action, (int, np.int32)):
            # take given action within the environment
            obs, reward, done, _ = self.env.step(action)

        else:
            print(f'[ERROR] - {action} type: {type(action)} is not a valid action.')
            raise ValueError

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
            info_state = get_info_state(obs, history, self.max_bet_number, mode, self.config_dict)
            strategy = self.env.agents[traverser].act(info_state, strategy=True)

            # 3.2
            # iterate over all actions and do traversals starting from each actions
            # subsequent history
            values = []
            orig_env = self.env
            for action_index in range(len(strategy.numpy()[0])):
                self.env = self.create_env_cpy(orig_env)
                history_cpy = deepcopy(history)  # copy bet size history
                bet = self.action_func(action_index, obs)
                history_cpy.append(bet)  # add bet to bet history

                # appending traaverser bets is redundant, as all actions would be appended
                # orig_env.agents[0].append_to_bet(a)

                traverser_payoff = self.traverse(history_cpy, traverser, CFR_iteration, bet)[0]
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
            info_state = get_info_state(obs, history, self.max_bet_number, mode, self.config_dict)
            non_traverser = 1 - traverser
            strategy = self.env.agents[non_traverser].act(info_state, strategy=True)

            # 2.
            # append Infoset, action_probabilities and CFR_iteration t to strat_mem
            cards = []
            for tensor in info_state[0]:
                cards.append(tensor.numpy())
            bet_hist = info_state[1].numpy()
            self.strategy_memory.append(([cards, bet_hist], CFR_iteration, strategy.numpy()[0]))

            # 3.
            # get action (softmax) according to strategy
            action = self.env.act(info_state)
            bet = self.action_func(action, obs)
            history.append(bet)

            return (self.traverse(history, traverser, CFR_iteration, bet)[0], self.ID)

    def set_weights(self, player, weights):
        self.env.agents[player].model.set_weights(weights)
