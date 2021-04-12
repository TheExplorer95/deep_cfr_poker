
def get_env_cpy(orig_env):
    env_cpy = create_env()
    env_cpy._dealer = copy(orig_env._dealer)

    return env_cpy


def trainings_loop():

    for CFR_iteration:

        traversal(env, obs)


def deep_CFR(env, val_net, strat_net, CFR_iterations, num_traversals, num_players):
    """

    Does some stuff.

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
                The trained strategy network.

    """

    # initialize ANNs and memories

    for t in range(CFR_iterations):
        for p in range(num_players):
            for k in range(num_traversals):
                # collect data from env via external sampling
                env = create.env()
                obs = env.reset()
                payoff = traverse(e)

            # initialize new value network (if not first iteration) and train with val_mem_p
            # (only used for prediction of the next regret values)
            train_val_net()

        # train the strat_net with strat_mem
        train_strat_net()

    return

def traverse(env, obs, history, traverser, advantage_mem_traverser, strat_mem, CFR_iteration):
    """
    Does some stuff.
    # input(histroy, traverser, val_model_0, val_model_1, val_mem_trav, strat_mem, t)

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
    state_terminal = all(obs['active'])
    if not state_terminal:
        # game state: end
        # calculate traversers payoff
        traverser_payoff = obs['payoff'][traverser] - obs['commitment'][traverser]

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
        obs = preprocess_obs(obs)
        strategy = env.dealer.agents[traverser].act(obs, strategy=True)

        # 2.
        # iterate over all actions and do traversals starting from each actions
        # subsequent history
        for a in range(len(strategy)):
            # cpy environment
            # take selected action within copied environment
            env_cpy = get_env_cpy(env)
            obs = env_cpy.step(a)
            traverser_payoff = traverse(env_cpy, obs, ...)

            return traverser_payoff

        # 3.
        # use returned payoff for advantage/regret computation
        for a in range(len(actions)):
            # compute advantages of each action

        # 4.
        # append Infoset, action_advantages and CFR_iteration t to advantage_mem_traverser

    else:
        # game state: traversers opponent has to take an action

        # 1.
        # compute strategy (next action) from the Infoset of the opponent and his
        # val_net via regret matching
        # call model on observation, no softmax
        obs = preprocess_obs(obs)
        strategy = env.dealer.agents[3-traverser].act(obs, strategy=True)

        # 2.
        # append Infoset, action_probabilities and CFR_iteration t to strat_mem

        # 3.
        # copy env and take action according to action_probabilities

        traverser_payoff = traverse()

        return traverser_payoff
