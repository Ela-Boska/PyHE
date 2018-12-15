import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

class DataLoader():
    def __init__(self,batch_size=128):
        mnist = tf.keras.datasets.mnist.load_data()
        self.train_data = mnist[0][0].astype('float') # np.array [60000,28,28] of int32
        self.train_labels = mnist[0][1].astype('int32') # np.array [55000] of int32
        self.eval_data = mnist[1][0].astype('float') # np.array [10000,28,28]
        self.eval_labels = mnist[1][1].astype('int32') # np.array [10000] of int32
        self.batch_size = batch_size

    def load_data(self,dataset):
        length = len(dataset)
        num_batch = length//self.batch_size
        index = np.random.shuffle(np.arange(0,length))
        index = index[0:num_batch*self.batch_size].reshape(num_batch,-1)
        for i in range(num_batch):
            indices = index[i]
            yield tf.constant(dataset[indices])

class Polynomial(tf.keras.layers.Layer):
    def __init__(self, coe):
        super(Polynomial, self).__init__()
        self.coe = coe
        self.degree = len(coe)
    
    def build(self, input_shape):
        pass
    
    def call(self, input):
        ans = sum([self.coe[i]*input**i for i in range(self.degree)])
        return ans

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32
        )