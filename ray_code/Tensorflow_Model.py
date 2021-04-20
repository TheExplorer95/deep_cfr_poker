import logging
import tensorflow as tf; logging.getLogger("tensorflow").setLevel(logging.WARNING)


def get_embedding_model(output_dim, num_cards):
    input_dim_rank = 13
    input_dim_suit = 4
    input_dim_card = 52

    cards_input = tf.keras.Input((num_cards,))

    # EMBEDDING MODEL (used for each group of cards)

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

    x = tf.clip_by_value(x, clip_value_min = 0, clip_value_max = 1e6)

    embs = card_embedding(x) + rank_embedding(x // 4) + suit_embedding(x%4)

    embs = embs * tf.expand_dims(valid, axis=-1)

    embs = tf.reduce_sum(embs , axis=1) # sum over num_cards card embeddings

    model = tf.keras.Model(cards_input, embs)

    return model


loss_tracker = tf.keras.metrics.Mean(name="loss")
class CustomModel(tf.keras.Model):

    def train_step(self, data):
        if len(data)== 4:
            hole_cards, bets, iterations, targets = data
            network_input = [[hole_cards],bets]
        elif len(data) ==5:
            hole_cards, flop_cards, bets, iterations, targets = data
            network_input = [[hole_cards, flop_cards],bets]

        with tf.GradientTape() as tape:
            predictions = self(network_input)

            loss = tf.reduce_mean(iterations * tf.reduce_sum((targets - predictions)**2, axis = -1), axis=None)

        gradients = tape.gradient(loss, self.trainable_variables)


        gradients = [tf.clip_by_norm(g, 1.0)
             for g in gradients]


        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        # Let's update and return the training loss metric.
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [loss_tracker]

def get_DeepCFR_model(output_dim, n_cards, n_bets, n_actions, strategy = False):
    """
    output_dim: dimensionality of embedding
    n_cards: a list of card numbers for each phase of the game (e.g. 2 preflop, 3 flop)
    n_bets: maximal number of bets in a game
    n_actions: number of possible action categories
    """

    # define inputs
    cards = [tf.keras.Input([n,], name = f"cards{i}") for i,n in enumerate(n_cards)]
    bets = tf.keras.Input([n_bets], name = "bets")

    # embedding layer for each card type (pre-flop, flop, turn, river)
    output_dims = [output_dim for _ in range(len(n_cards))]

    embedding_layers = [get_embedding_model(output_dim, num_cards) for num_cards,
                        num_output_dims in zip(n_cards, output_dims)]

    card1 = tf.keras.layers.Dense(output_dim, activation = "relu")
    card2 = tf.keras.layers.Dense(output_dim, activation = "relu")
    card3 = tf.keras.layers.Dense(output_dim, activation = "relu")

    bet1 = tf.keras.layers.Dense(output_dim)
    bet2 = tf.keras.layers.Dense(output_dim)

    comb1 = tf.keras.layers.Dense(output_dim)
    comb2 = tf.keras.layers.Dense(output_dim)
    comb3 = tf.keras.layers.Dense(output_dim)

    if not strategy:
        action_head = tf.keras.layers.Dense(n_actions, bias_initializer = tf.keras.initializers.Constant(
        value=-5))
    else:
        action_head = tf.keras.layers.Dense(n_actions)


    # card branch
    card_embs = []
    for embedding, card_group in zip(embedding_layers, cards):
        card_embs.append(embedding(card_group))

    card_embs = tf.concat(card_embs, axis= 1)

    x = card1(card_embs)
    x = card2(x)
    x = card3(x)

    # bet branch
    bet_size = tf.clip_by_value(bets, tf.constant(0.), tf.constant(1e6)) # clip bet sizes
    bets_occured = tf.cast(bets >= tf.constant(0.), tf.float32) # check if bet occured
    bet_features = tf.concat([bet_size, bets_occured], axis = -1)   # bet size and boolean bet
    y = bet1(bet_features)
    y = bet2(y)

    # combine bet history and card embedding branches
    z = tf.concat([x,y],axis=-1)
    z = tf.nn.relu(comb1(z))
    z = tf.nn.relu(comb2(z) + z)
    z = tf.nn.relu(comb3(z) + z)

    # normalize (needed because of bet sizes)
    z = (z - tf.math.reduce_mean(z, axis=None)) / tf.math.reduce_std(z, axis=None)


    if not strategy:
        output = tf.nn.relu(action_head(z))

    else:
        output = tf.nn.softmax(action_head(z))

    DeepCFR_model = CustomModel(inputs = [cards, bets], outputs = output)

    return DeepCFR_model


# DeepCFR_model = get_DeepCFR_model(output_dim = 256, n_cards = [2,3], n_bets = 4, n_actions = 3)
