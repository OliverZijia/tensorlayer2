import tensorlayer as tl
import tensorflow as tf


class FeedForwardLayer(tl.layers.Layer):
    """
    Feed forward layer

    Parameters
    ----------
    hidden_size:
        hidden size of both input and output
    ff_size:
        hidden size used in the middle layer of this feed forward layer
    keep_prob:
        keep probability of dropout layer

    """

    def __init__(self, hidden_size, ff_size, keep_prob):
        super(FeedForwardLayer, self).__init__()
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.keep_prob = keep_prob

    def build(self, inputs_shape):
        self.dense1 = tl.layers.Dense(self.ff_size)
        self.dense2 = tl.layers.Dense(self.hidden_size)
        self.dropout = tl.layers.Dropout(self.keep_prob)

    def forward(self, inputs):
        return self.dense2(self.dropout(tf.nn.relu(self.dense1(inputs))))

    def __repr__(self):
        pass
