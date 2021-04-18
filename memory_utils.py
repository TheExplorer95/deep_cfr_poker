import tensorflow as tf
import h5py
import numpy as np
import random

def flatten_data_for_memory(info_state, iteration, values):
    """
    Flattens data to store into a memory object.
    """
    #print(values)

    #print(values, info_state)


    """
    BIG PROBLEM: (TODO) values is sometimes a list and sometimes an array of shape (1,5) and sometimes it is a list of length


    """

    #### TODO fix that values is always the same shape and the same object type (currently isn't)
    try:

        values = np.reshape(values, (values.shape[1],))
    except:
        values = np.array(values)

    flattened_data = np.concatenate(
                    [
                        info_state[0][0].flatten(),
                        info_state[1].flatten(),
                        np.array([iteration]),
                        np.array(values)
                    ],  axis = 0)

    return flattened_data


class MemoryWriter(object):

    """
    Keeps track of how many items are already processed for this memory.
    It's main purpose is to store the generated data with the save_to_memory method
    """


    def __init__(self, max_size, vector_length, flatten_func, file_name):

        self.max_size = max_size

        self.vector_len = vector_length # flatten the input that we want to store and take len

        self.flatten_func = flatten_func

        self.counter = np.array([0,0]) # can't assign a single number to a dataset in hdf5 files

        self.file_name = file_name

        # load previous counter or initiate memory file
        try:
            with h5py.File(self.file_name, "r") as hf:
                self.counter = np.array(hf.get("counter"))
            print("previous counter loaded")

        except:
            # create new dataset file with a counter and an array
            with h5py.File(self.file_name, "w") as hf:

                hf.create_dataset("counter", data= self.counter)
                hf.create_dataset("data", (self.max_size, self.vector_len), dtype = np.float32)
            print(f"new counter set and dataset of size {(self.max_size, self.vector_len)} is initiated.")

    def save_to_memory(self, data):
        """
        Takes a list of tuples (info_state, iteration, values) and stores each to the memory hdf5 file.

        Uses Reservoir sampling for samples that exceed the specified max_size.
        """
        with h5py.File(self.file_name, 'r+') as hf:

            # store each tuple in data
            for info_state, iteration, values in data:

                self.counter[1] += 1

                # save counter only every 100 steps
                if not self.counter[1]%100:
                    hf.get("counter")[1] = self.counter[1]

                # if reservoir is not yet full, simply add new sample
                if self.counter[1] < self.max_size:

                    flattened_data = self.flatten_func(info_state, iteration, values)

                    #print(flattened_data.shape)
                    #print(flattened_data)
                    hf.get("data")[self.counter[1]] = flattened_data # fill empty row with data

                # if reservoir is full already, randomly replace or not replace old data
                else:
                    idx = random.randint(0, self.counter[1]) # index to replace (or not)
                    if idx < self.max_size:

                        flattened_data = self.flatten_func(info_state, iteration, values)

                        hf.get("data")[idx] =  flattened_data # replace the old data at idx with the new sample

                    else:
                        pass # data is not stored in favor of old data
