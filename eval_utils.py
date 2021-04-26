import ray
import clubs_gym


@ray.remote
class Traversal_Runner:
    """
    Used for the distributed sampling of CFR traversals with ray
    multiprocessing.

    Parameters
    ----------
    ID : int
        ID of the current Runner.

    env_str : gym.env ID string
        The game to optimize a strategy for.

    config_dict : clubs_gym.env config_dict; type - dict
        The config dict belonging to the env_str passed.

    max_bet_number : int
        Number of maximal possible bets per game.

    model_save_paths : list(str, str)
        Used for initialization of the Agent models.

    agent_fct : function
        Creates the Agent for the Poker environment.

    bet_fct : instance of bet_fct
        Used for adjusting the model output for the environment input.
    """

    def __init__(self, ID, env_str, config_dict, max_bet_number, bet_fct,
                 model_save_paths=None, agent_fct=None):
        self.ID = ID
        self.env_str = env_str
        self.config_dict = config_dict
        self.max_bet_number = max_bet_number
        self.strategy_memory = []
        self.advantage_memory = []
        self.bet_fct = bet_fct
        self.create_env(model_save_paths, agent_fct)

    def create_env(self, model_save_paths, agent_fct):
        # Creates an environment with new beginning state and new agent
        # instances with models passed as filepaths.

        clubs_gym.envs.register({self.env_str: self.config_dict})
        self.env = gym.make(self.env_str)

        # create new agents
        self.env.register_agents([agent_fct(model_save_path) for model_save_path in model_save_paths])

        # random dealer position
        self.env.dealer.button += 1 if random.randint(0, 1) == 0 else 0

        self.env.reset()

    def create_env_cpy(self, orig_env):
        """
        Creates a copy of the given clubs_gym envyronment. The dealers cards
        are shuffled to ensure randomness at chance nodes (where the dealer
        hands out the flop, river and street) and the reference for the model
        is creted to the same model from before.
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
        # returns the runners memory and deletes it
        advantage, strategy = self.advantage_memory, self.strategy_memory
        del self.advantage_memory, self.strategy_memory
        self.advantage_memory = []
        self.strategy_memory = []

        bet_memory = []
        for i in range(2):
            bet_memory.extend(self.env.agents[i].bet_history)
            self.env.agents[i].bet_history = []

        return advantage, strategy, bet_memory

    def set_weights(self, player, weights):
        # sets the weights of the given player/agent
        self.env.agents[player].model.set_weights(weights)

    def traverse(self, history, traverser, CFR_iteration, action=None):
        """
        Implementation of deep CFR traversals with external sampling, according
        to the pseudocode from the deep CFR paper (Algorithm 2).

        Parameters
        ----------
        history : list
            Betting history [10, 10, 0, 0, 23, -1,-1,-1,-1, ...]
            Each value indicated the amount of money played in one turn.
            0 indicates a check or a fold. 1 - inf indicates a check or a call
            (can be deduced from previous entry).

        traverser : int()
            The current traverser, for Heads up -> 0 or 1

        CFR_iteration :
            The current CFR iteration (Saved to memory for model training;
            liniar CFR).

        action: int
            Action to take within the runners environment.

        Returns:
        -------
            payoff : int
                The payoff for the current traversal (rekursively defined).

            self.ID : int
                The runner ID.

        """

        # 0.
        # set and get env parameters

        # if new game
        if action == 'first':
            obs = self.env.reset()

        else:
            # take given action within the environment
            obs, reward, done, _ = self.env.step(int(action))

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
            info_state = get_info_state(obs, history, self.max_bet_number, self.env.dealer.num_streets, self.config_dict)
            strategy = self.env.agents[traverser].act(info_state, strategy=True)

            # 3.2
            # iterate over all actions and do traversals starting from each actions
            # subsequent history
            values = []
            orig_env = self.env
            for action_index in range(len(strategy.numpy()[0])):
                # create env copy for traversal
                self.env = self.create_env_cpy(orig_env)
                history_cpy = deepcopy(history)
                bet = self.bet_fct(action_index, obs)
                history_cpy.append(bet)

                # get payoff for given action/bet
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

            # scaling
            std = np.std(advantages)
            if not std == 0:
                advantages = np.array(advantages) / std
                self.advantage_memory.append(([cards, bet_hist], CFR_iteration, advantages))
            elif np.count_nonzero(advantages) == 0:
                advantages = np.array(advantages)
                self.advantage_memory.append(([cards, bet_hist], CFR_iteration, advantages))
            else:
                print(f'[INFO] - Advantages {advantages} were not appended')

            expected_infostate_value = np.sum(strategy.numpy()[0] * np.array(values))
            return (expected_infostate_value, self.ID)

        # 4.
        # Non-Traverser Node
        else:
            # game state: traversers opponent has to take an action

            # 1.
            # compute strategy (next action) from the Infoset of the opponent and his
            # val_net via regret matching
            info_state = get_info_state(obs, history, self.max_bet_number, self.env.dealer.num_streets, self.config_dict)
            strategy = self.env.agents[1 - traverser].act(info_state, strategy=True)

            # 2.
            # append Infoset, action_probabilities and CFR_iteration t to strat_mem
            cards = []
            for tensor in info_state[0]:
                cards.append(tensor.numpy())
            bet_hist = info_state[1].numpy()
            self.strategy_memory.append(([cards, bet_hist], CFR_iteration, strategy.numpy()[0]))

            # 3.
            # get action according to strategy

            action = self.env.act(info_state)
            bet = self.bet_fct(action, obs)
            history.append(bet)

            return (self.traverse(history, traverser, CFR_iteration, bet)[0], self.ID)
