import tensorflow as tf
import numpy as np
import random
import h5py

def get_tf_dataset(file_name, batch_size, num_infostates, num_cards, num_bets, num_actions):
    """
    Creates a tensorflow dataset from a .h5 file

    sum(n_cards) == 2 -> "hole_cards only"
    sum(n_cards) == 5 -> "hole_cards + flop"
    """


    def get_input_targets(stored_vector, num_cards):

        if num_cards == 2:
            #indices = [2,8,9]
            indices = [2, 2+num_bets, 2+num_bets+1]
            hole_cards = stored_vector[:indices[0]]
            bets = stored_vector[indices[0]:indices[1]]
            iteration = stored_vector[indices[1]]
            values = stored_vector[indices[-1]:]

            return tf.constant(hole_cards), tf.constant(bets), tf.constant(iteration), tf.constant(values)

        elif num_cards == 5:
            indices = [2, 5, 5+num_bets, 5+num_bets+1]

            hole_cards = stored_vector[:indices[0]]
            flop_cards = stored_vector[indices[0]:indices[1]]
            bets = stored_vector[indices[1]:indices[2]]
            iteration = stored_vector[indices[2]]
            values = stored_vector[indices[-1]:]

            return tf.constant(hole_cards), tf.constant(flop_cards), tf.constant(bets), tf.constant(iteration), tf.constant(values)


        #return network_input, targets, iteration




    def memory_generator():
        with h5py.File(file_name,"r") as hf:

            while True:
                #np.array(hf.get("counter"))[1])
                idx = random.randint(1,num_infostates)
                stored_vector = np.array(hf.get("data")[idx])

                yield get_input_targets(stored_vector, num_cards)


    if num_cards == 2:
        out_signature = (tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                       tf.TensorSpec(shape=(num_bets,), dtype=tf.float32),
                                       tf.TensorSpec(shape=(), dtype=tf.float32),
                                       tf.TensorSpec(shape=(num_actions,), dtype=tf.float32))

    elif num_cards == 5:
        out_signature = (tf.TensorSpec(shape=(2,), dtype=tf.float32),
                         tf.TensorSpec(shape=(3,), dtype=tf.float32),
                                       tf.TensorSpec(shape=(num_bets,), dtype=tf.float32),
                                       tf.TensorSpec(shape=(), dtype=tf.float32),
                                       tf.TensorSpec(shape=(num_actions,), dtype=tf.float32))

    train_ds = tf.data.Dataset.from_generator(memory_generator,
                                   output_signature= out_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds
