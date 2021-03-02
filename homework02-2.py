import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


class Q_Net(tf.keras.Model):
    def __init__(self, output_units=2):
        super(Q_Net, self).__init__()

        self.all_layers = [tf.keras.layers.Dense(64, activation= "tanh"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation= "tanh"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64,  activation= "tanh"),
        tf.keras.layers.Dense(output_units, use_bias=False) ]

        self.gamma = tf.constant(0.99, dtype=tf.float32)

    @tf.function
    def call(self, x, training=False):

        output = {}
        for layer in self.all_layers:
            x = layer(x)

        output["q_values"] = x
        return output


if __name__ == "__main__":

    kwargs = {
        "model": Q_Net,
        "environment": "CartPole-v0",
        "num_parallel": 5,
        "total_steps": 100,
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 0.95,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 32
    saving_after = 5

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()

    loss_metric = tf.keras.metrics.Mean('loss')
    MSE = tf.keras.losses.MSE
    optimizer = tf.keras.optimizers.Adam(1e-3)

    model2= Q_Net()

    for e in range(epochs):

        # training core

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)

        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        #action, state, reward, next_action, terminal
        print("optimizing...")

        for state, action, reward, next_state, not_done in zip(*[ data_dict.get(key) for key in data_dict.keys() ]):


                reward = tf.cast(reward, tf.float32)

                # calculate desired q value predictions (y)
                maxq1 = tf.reduce_max(agent.model(next_state,training=True)["q_values"], axis=-1)

                maxq2 = tf.reduce_max(model2(next_state,training=True)["q_values"], axis=-1)
                maxq = tf.concat([tf.expand_dims(maxq1, axis=-1),tf.expand_dims(maxq2,axis=-1)],axis=-1)
                maxq = tf.math.reduce_min(maxq, axis=-1)

                y = tf.where(not_done==1, reward + agent.model.gamma * maxq, reward)

                with tf.GradientTape() as tape, tf.GradientTape() as tape2:
                    # make q value predictions for the actions taken during sampling
                    model_out = agent.model(state, training=True)
                    q_values = model_out["q_values"]
                    q_value_prediction = tf.gather(q_values, action, batch_dims=1)

                    loss = MSE(y, q_value_prediction)

                    model_out2 = model2(state, training=True)
                    q_values2 = model_out2["q_values"]
                    q_value_prediction2 = tf.gather(q_values2, action, batch_dims=1)

                    loss2 = MSE(y, q_value_prediction2)


                loss_metric.update_state(loss)
                # get gradients and update model parameters
                gradients = tape.gradient(loss, agent.model.trainable_variables)
                gradients2 = tape2.gradient(loss2, model2.trainable_variables)
                optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))
                optimizer.apply_gradients(zip(gradients2, model2.trainable_variables))



        new_weights = agent.get_weights()
        #
        # # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)

        # print progress
        print(
            f"epoch ::: {e}  loss ::: {loss_metric.result()}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        #manager.set_temperature(temperature=agent.temperature * 0.9)
        manager.set_epsilon(epsilon=0.8*agent.epsilon)

        # """if e % saving_after == 0:
        #     # you can save models
        #     manager.save_model(saving_path, e)"""
        # tf.print(loss_metric.result())
        # loss_metric.reset_states()
    # and load mmodels
    # manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
