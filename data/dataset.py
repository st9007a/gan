from tensorflow.examples.tutorials.mnist import input_data

class mnist():

    def __init__(self):
        self.dataset = input_data.read_data_sets("MNIST_data/", one_hot = True)

    def get_batch(self, batch_size):
        data = self.dataset.train.next_batch(batch_size)
        return data[0], data[1]
