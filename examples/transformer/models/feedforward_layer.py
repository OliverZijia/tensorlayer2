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

        self._nodes_fixed = True
        if not self._built:
            self.build(tuple())
            self._built = True

    def build(self, inputs_shape):
        # self.dense1 = tl.layers.Dense(self.ff_size)
        # self.dense2 = tl.layers.Dense(self.hidden_size)
        # self.dropout = tl.layers.Dropout(self.keep_prob)
        self.W1 = self._get_weights('W1', (self.hidden_size, self.ff_size))
        self.W2 = self._get_weights('W2', (self.ff_size, self.hidden_size))

    def forward(self, inputs):
        # print(inputs.shape)
        # return self.dense2(self.dropout(tf.nn.relu(self.dense1(inputs))))
        out = tf.tensordot(inputs, self.W1, axes=[[2], [0]])
        out = tf.tensordot(out, self.W2, axes=[[2], [0]])
        return out

    def __repr__(self):
        return "feedforward layer"
