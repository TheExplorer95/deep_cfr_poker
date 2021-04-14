import copy
import copy.deepcopy as deepcopy
from random import shuffle

# def get_env_cpy(orig_env):
#     env_cpy = create_env()
#     env_cpy._dealer = copy(orig_env._dealer)
#
#     return env_cpy

def get_env_cpy(env):
    env_cpy = gym.make(env.unwrapped.spec.id)  # gets the gym environment name, e.g. "noLimit_TH-v0"
    env_cpy.reset()

    # 1. set previous obs
    env_cpy.prev_obs = copy.deepcopy(env.prev_obs)

    # 2. make copy of the dealer and shuffle its deck (new dealer produces different community cards)
    env_cpy.dealer = copy.deepcopy(env.dealer)
    shuffle(env_cpy.dealer.deck.cards)

    # 3. create reference to the original agents (new env uses old models for sampling)
    # creating new agents also possible, depends on computation overhead by ray:
    #                                     [TensorflowAgent(f'test_model_{i}') for i in range(2)]
    env_cpy.agents = copy.copy(env.agents)

    return env_cpy

def get_history_cpy(orig_history):
    """copies a list"""

    return copy.deepcopy(orig_history)

def convert_cards_to_id(cards):
    """
    Computes the unique card id for a clubs.cards type card. Assumes a card
    deck of size 52 with ranks clubs (♣), diamonds (♦), hearts (♥) and
    spades (♠), and suits 2, 3, 4, 5, 6, 7, 8, 9, 10, B, D, K, A.
    0 = duce of clubs; 1 = duce of diamonds ...
                                ... 50 = ace of hearts; 51 ace of spades
    Parameters
    ----------
    cards: list(clubs.card)
        List of clubs.card cards to convert.
    Returns
    -------
    converted_cards: list(card_ids)
    """

    # length of each card encoding (see class clubs.Card for reference)
    # prime = 8, bit_rank = 4
    oneHot_suit = 4
    oneHot_rank = 16
    oneHot_rank_offset = 3
    encoding_length = 32

    converted_cards = []
    for card in cards:
        # get card binary string (deals with shortening of string when
        # 32Bit int not 32Bit long; python truncates higher_order bits in string
        # representation when not flipped)
        card_binary = '0'*(encoding_length-len(card._bin_str)) + card._bin_str

        # extract rank and suit
        card_suit = card_binary[oneHot_rank:oneHot_rank+oneHot_suit].find('1')
        card_rank = card_binary[oneHot_rank_offset:oneHot_rank][::-1].find('1')

        # compute card id
        card_id = card_rank * 4 + card_suit

        converted_cards.append([card_id])

    return converted_cards


def get_info_state(obs, history, max_bet_number, mode="flop only"):
    """ Transforms the observation dictionary from clubs env and the history
        list to an info state (input to the ANN model)"""

    bet_history = get_history_cpy(history)
    h_cards = obs["hole_cards"]
    c_cards = obs["community_cards"]

    # convert hole cards to indices.
    hole_cards = [convert_cards_to_id(h_cards)]

    # convert community cards to indices and split into flop, turn, river
    c_cards_len = len(c_cards)
    flop_cards = []
    turn = []
    river = []

    if c_cards_len:
        c_cards = convert_cards_to_id(c_cards)
        if c_cards_len >= 3:
            flop_cards = c_cards[:3]

        if c_cards_len >= 4:
            turn = c_cards[3]

        if c_cards_len ==5:
            river = c_cards[4]

    # padding of bet history
    while len(bet_history) < max_bet_number:
        bet_history.append(-1)

    bet_history = tf.constant([bet_history], dtype = tf.float32)
    hole_cards = tf.constant(hole_cards, dtype = tf.float32)

    # padding for not yet given cards
    if mode == "flop only":
        # if no flop card is given, use no-card index (-1)
        if not len(flop_cards):
            flop_cards = tf.constant([
            [[-1], [-1], [-1]]
            ], dtype= tf.float32)

#         [[-1] for i in range(len(flop_cards))]
        else:
            flop_cards = tf.constant([flop_cards], dtype = tf.float32)

        output = [[hole_cards, flop_cards], bet_history]

    if mode == "hole cards only":
        output = [[hole_cards], bet_history]


    if mode == "flop + turn":

        if not len(flop_cards):
            flop_cards = tf.constant([
            [[-1], [-1],[-1]]
            ], dtype= tf.float32)
        else:
            flop_cards = tf.constant([flop_cards], dtype = tf.float32)

        if not len(turn):
            turn = tf.constant([
            [[-1]]
            ], dtype = tf.float32)
        else:
            turn = tf.constant([turn], dtype = tf.float32)

        output = [[hole_cards, flop_cards, turn], bet_history]


    if mode == "full poker":
        if not len(flop_cards):
            flop_cards = tf.constant([
            [[-1], [-1],[-1]]
            ], dtype= tf.float32)

        else:
            flop_cards = tf.constant([flop_cards], dtype = tf.float32)

        if not len(turn):
            turn = tf.constant([
            [[-1]]
            ], dtype = tf.float32)
        else:
            turn = tf.constant([turn], dtype = tf.float32)

        if not len(river):
            river = tf.constant([
            [[-1]]
            ], dtype = tf.float32)
        else:
            river = tf.constant(river, dtype = tf.float32)

        output = [[hole_cards, flop_cards, turn, river], bet_history]

    return output


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


## test traverse

# create environment

# reset environment, obtain obs,

history = []
