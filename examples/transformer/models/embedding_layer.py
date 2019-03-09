import tensorlayer as tl
import tensorflow as tf


class EmbeddingLayer(tl.layers.Layer):
    """
    Embedding layer

    Parameters:
        vocab_size: vocabulary size
        hidden_size: embedding size, the output size of each word after embedding
    """

    def __init__(self, vocab_size, hidden_size, init=None):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.init = init

    def __repr__(self):
        pass

    def build(self, inputs_shape):
        if self.init is not None:
            self.W = self._get_weights('weights', shape=(self.vocab_size, self.hidden_size), init=self.init)
        else:
            self.W = self._get_weights('weights', shape=(self.vocab_size, self.hidden_size))

    def forward(self, inputs):
        # inputs is of size (batch_size, length)
        # create mask for inputs, 0 is <pad> in dictionary
        mask = tf.to_float(tf.not_equal(inputs, 0))

        embeddings = tf.gather(self.W, inputs)
        embeddings *= mask

        return embeddings
