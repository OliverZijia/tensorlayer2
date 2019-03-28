import tensorlayer as tl
import tensorflow as tf


class EmbeddingLayer(tl.layers.Layer):
    """
    Embedding layer

    Parameters:
        vocab_size: vocabulary size
        hidden_size: embedding size, the output size of each word after embedding
    """

    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.build(tuple())
        self._built = True

    def __repr__(self):
        return "embedding"

    def build(self, inputs_shape):
        self.W = self._get_weights('weights', shape=(self.vocab_size, self.hidden_size))

    def forward(self, inputs):
        # inputs is of size (batch_size, length)
        # create mask for inputs, 0 is <pad> in dictionary
        mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.float32)

        embeddings = tf.gather(self.W, inputs)
        embeddings *= tf.expand_dims(mask, 2)

        return embeddings
