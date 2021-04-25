import logging
import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.WARNING)


class Normalize(tf.keras.layers.Layer):
    def __init__(self):
        super(Normalize, self).__init__()
        self.normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean = 0, variance = 1)

    def call(self, x):
        return self.normalize(x)


@tf.function
def regret_matching(x):
    """
    Outputs action probabilities proportional to positive regrets
    """
    zeros = tf.zeros_like(x)
    x = tf.math.maximum(zeros, x)

    if tf.math.count_nonzero(x) > 0:
        return x / tf.reduce_sum(x, axis=-1, keepdims=True)
    else:
        # if only negative or zero regrets, output uniform probability
        return tf.ones_like(x) / x.shape[-1]


class RegretMatching(tf.keras.layers.Layer):
    # small wrapper class for the regret matching function
    def __init__(self):
        super(RegretMatching, self).__init__()
        self.regr_func = regret_matching

    def call(self, x):
        return self.regr_func(x)


def get_embedding_model(output_dim, num_cards):
    # returns the embedding model for a hand of cards plus community cards.
    # Further information in deep CFR or our documentaiton.

    input_dim_rank = 13
    input_dim_suit = 4
    input_dim_card = 52

    cards_input = tf.keras.Input((num_cards,))

    rank_embedding = tf.keras.layers.Embedding(
        input_dim_rank, output_dim, embeddings_initializer='uniform',
        embeddings_regularizer=None, activity_regularizer=None,
        embeddings_constraint=None, mask_zero=False, input_length=None,
    )

    suit_embedding = tf.keras.layers.Embedding(
        input_dim_suit, output_dim, embeddings_initializer='uniform',
        embeddings_regularizer=None, activity_regularizer=None,
        embeddings_constraint=None, mask_zero=False, input_length=None,
    )

    card_embedding = tf.keras.layers.Embedding(
        input_dim_card, output_dim, embeddings_initializer='uniform',
        embeddings_regularizer=None, activity_regularizer=None,
        embeddings_constraint=None, mask_zero=False, input_length=None,
    )

    # cards is a list of card indices (2 for preflop, 3 for flop, 1 for turn, 1 for river)
    x = tf.keras.layers.Flatten()(cards_input)
    valid = tf.cast(x >= tf.constant(0.), tf.float32)
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1e6)
    embs = card_embedding(x) + rank_embedding(x // 4) + suit_embedding(x % 4)
    embs = embs * tf.expand_dims(valid, axis=-1)
    embs = tf.reduce_sum(embs, axis=1)  # sum over num_cards card embeddings

    model = tf.keras.Model(cards_input, embs)

    return model


# used within the custom model defined afterwards
# Info: we had problems with creating it as a new class variable, thats why its
# located outside of the class
loss_tracker = tf.keras.metrics.Mean(name="loss")


class CustomModel(tf.keras.Model):
    """
    Wrapper function for a Tensorflow Model, that allows use of the model.fit
    functionality of a tensorflow model. Not as handy as thought, needs some
    adjustments, as loss history is not saved.
    """

    def train_step(self, data):
        # uses tensorflow autograd

        if len(data) == 4:
            hole_cards, bets, iterations, targets = data
            network_input = [[hole_cards], bets]

        elif len(data) == 5:
            hole_cards, flop_cards, bets, iterations, targets = data
            network_input = [[hole_cards, flop_cards], bets]

        with tf.GradientTape() as tape:
            predictions = self(network_input)
            loss = tf.reduce_mean(iterations * tf.reduce_sum((targets - predictions)**2, axis = -1), axis=None)

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Let's update and return the training loss metric.
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # list of the metrics so reset_states() can be called automatically
        return [loss_tracker]


def get_DeepCFR_model(output_dim, n_cards, n_bets, n_actions, strategy=False,
                      zero_outputs=False):
    """
    Returns the deep CFR strategy or advantage/value Model depending on the
    arguments passed. More information of the architecture can be found in our
    documentation.

    Params
    ------
    output_dim : int
        Dimensionality used for the card embeddings (latent dim).

    n_cards : list of ints
        A list of card numbers for each phase of the game (e.g. 2 preflop, 3 flop).

    n_bets : int
        Amount of maximal possible bets in one poker game.

    n_actions : int
        Possible actions made by the Agent/Model.

    strategy : bool
        Defines whether to return a advantage/value model or strategy model

    zero_outputs : bool
        If set to True, the bias of the model output nodes is set to -5, which
        leads to random beahviour, due to the regret matching taking place.

    Returns
    -------
        The initialized tensorflow model.
    """

    # define inputs
    cards = [tf.keras.Input([n, ], name=f"cards{i}") for i, n in enumerate(n_cards)]
    bets = tf.keras.Input([n_bets], name="bets")

    # embedding layer for each card type (pre-flop, flop, turn, river)
    output_dims = [output_dim for _ in range(len(n_cards))]
    embedding_layers = [get_embedding_model(output_dim, num_cards) for num_cards,
                        num_output_dims in zip(n_cards, output_dims)]

    regr_matching = RegretMatching()
    card1 = tf.keras.layers.Dense(output_dim, activation="relu")
    card2 = tf.keras.layers.Dense(output_dim, activation="relu")
    card3 = tf.keras.layers.Dense(output_dim, activation="relu")

    bet1 = tf.keras.layers.Dense(output_dim)
    bet2 = tf.keras.layers.Dense(output_dim)

    comb1 = tf.keras.layers.Dense(output_dim)
    comb2 = tf.keras.layers.Dense(output_dim)
    comb3 = tf.keras.layers.Dense(output_dim)
    norm = Normalize()

    if zero_outputs:
        bias = -5
    else:
        bias = 0

    action_head = tf.keras.layers.Dense(n_actions,
                                        bias_initializer=tf.keras.initializers.Constant(value=bias))

    # card branch
    card_embs = []
    for embedding, card_group in zip(embedding_layers, cards):
        card_embs.append(embedding(card_group))

    card_embs = tf.concat(card_embs, axis=1)

    x = card1(card_embs)
    x = card2(x)
    x = card3(x)

    # bet branch
    bet_size = tf.clip_by_value(bets, tf.constant(0.), tf.constant(1e6))  # clip bet sizes
    bets_occured = tf.cast(bets >= tf.constant(0.), tf.float32)  # check if bet occured
    bet_features = tf.concat([bet_size, bets_occured], axis=-1)  # bet size and boolean bet
    y = bet1(bet_features)
    y = bet2(y)

    # combine bet history and card embedding branches
    z = tf.concat([x, y], axis=-1)
    z = tf.nn.relu(comb1(z))
    z = tf.nn.relu(comb2(z) + z)
    z = tf.nn.relu(comb3(z) + z)

    # normalize (needed because of bet sizes)
    z = norm(z)  # normalize(z) #(z - tf.math.reduce_mean(z, axis=-1)) / tf.math.reduce_std(z, axis=-1)

    output = action_head(z)

    if strategy:
        output = regr_matching(output)

    DeepCFR_model = CustomModel(inputs = [cards, bets], outputs = output)

    return DeepCFR_model
