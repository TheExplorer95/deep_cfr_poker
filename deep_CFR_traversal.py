import copy

def get_env_cpy(orig_env):
    env_cpy = create_env()
    env_cpy._dealer = copy(orig_env._dealer)

    return env_cpy


def get_history_cpy(orig_history):
    """copies a list"""

    return copy.deepcopy(orig_history)




def get_info_state(obs, history):
    """ Transforms the observation dictionary from clubs env and the history list to an info state (input to the ANN model)"""

    ##### TODO #####
    pass

def save_to_memory(type, player, info_state, iteration, values):

    """This function saves stuff to memory"""

    ##### TODO #####

    ### bring info state and values and iteration into a useful data structure

    ### save this data structure, e.g. a dict (for instance each iteration gets its own file?)

    pass


def deep_CFR(env, val_net, strat_net, CFR_iterations, num_traversals, num_players):
    """"
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
                The trained strategy network.""""

    # initialize ANNs and memories

    for t in range(CFR_iterations):
        for p in range(num_players):
            for k in range(num_traversals):
                # collect data from env via external sampling
                env = create.env()
                obs = env.reset()
                traverse(env, obs, history, p, t)

            # initialize new value network (if not first iteration) and train with val_mem_p
            # (only used for prediction of the next regret values)
            train_val_net()

        # train the strat_net with strat_mem
        train_strat_net()

    return 0

def traverse(env, obs, history, traverser, CFR_iteration):
    """
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
    state_terminal = not all(obs['active'])
    if state_terminal:
        # game state: end
        # calculate traversers payoff
        traverser_payoff = obs['payoff'][traverser] - obs['commitment'][traverser] # gain or loss

        return traverser_payoff

    elif chance_node:
        # game state: the dealer has to hand out cards or turn the river
        # does not count into traversal depth
        # !!chance nodes are automatically handled by environment when taking
        # an action!!
        pass

    elif obs['action'] == traverser:
        # game state: traverser has to take an action

        # 1.
        # compute strategy (next action) from the Infoset of the traverser and
        # his val_net via regret matching (used for weighting when calculating
        # the advantages)
        # call model on observation, no softmax
        info_state = get_info_state(obs, history)
        strategy = env.dealer.agents[traverser].act(info_state, strategy=True)

        # 2.
        # iterate over all actions and do traversals starting from each actions
        # subsequent history
        values = []
        for a in range(len(strategy)):
            # cpy environment
            # take selected action within copied environment
            history_cpy = get_hist_cpy(history) # copy bet size history
            env_cpy = get_env_cpy(env)
            obs = env_cpy.step(a)

            history_cpy.append(a) # add bet to bet history


            traverser_payoff = traverse(env_cpy, obs, history_cpy, traverser, CFR_iteration)
            values.append(traverser_payoff)

            #return traverser_payoff

        # 3.
        # use returned payoff for advantage/regret computation
        advantages = []
        for a in range(len(strategy)):
            # compute advantages of each action
            advantages.append(values[a] - np.sum(strategy.numpy() * np.array(values)))

        # 4.
        # append Infoset, action_advantages and CFR_iteration t to advantage_mem_traverser

        save_to_memory(
        type = "value",
        player = traverser,
        info_state = info_state,
        iteration = CFR_iteration,
        values = advantages
        )
    else:
        # game state: traversers opponent has to take an action

        # 1.
        # compute strategy (next action) from the Infoset of the opponent and his
        # val_net via regret matching
        # call model on observation, no softmax
        info_state = get_info_state(obs, history)

        non_traverser = 3 - traverser
        strategy = env.dealer.agents[non_traverser].act(info_state, strategy=True) # env.act(orig_obs, strategy = True) probably is what works

        # 2.
        # append Infoset, action_probabilities and CFR_iteration t to strat_mem
        save_to_memory(
        type = "strategy",
        player = non_traverser,
        info_state = info_state,
        iteration = CFR_iteration,
        values = strategy.numpy()
        )
        # 3.
        # copy env and take action according to action_probabilities
        dist = tfp.distributions.Categorical(probs = strategy.numpy())

        sampled_action = dist.sample((1)).numpy()
        action = env.act(info_state)
        obs = env.step(action)
        # update history
        history.append(action)
        return traverse(env, obs, history, traverser, CFR_iteration)
