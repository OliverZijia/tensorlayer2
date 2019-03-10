import tensorflow as tf
import tensorlayer as tl


class MultiHeadAttentionLayer(tl.layers.Layer):
    """
    Attention layer

    Parameters
    ----------
    params: a parameter object
        refer to ../transformer/utils/model_params for details

    """

    def __init__(self, num_heads, hidden_size, keep_prob):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.dk = hidden_size / num_heads
        self.keep_prob = keep_prob

    def build(self, inputs_shape):
        self.Wq = self._get_weights("W_q", shape=(self.hidden_size, self.hidden_size))
        self.Wk = self._get_weights("W_k", shape=(self.hidden_size, self.hidden_size))
        self.Wv = self._get_weights("W_v", shape=(self.hidden_size, self.hidden_size))
        self.Wout = self._get_weights("Wout", shape=(self.hidden_size, self.hidden_size))

    def forward(self, x, y):
        """
        Parameters
        ----------
        x:
            input to generate query & key, shape=(batch_size, length, hidden_size)
        y:
            input to generate value, shape=(batch_size, length, hidden_size)

        Return
        -------
            shape=(batch_size, length, hidden_size)
        """
        q = tf.matmul(x, self.Wq)
        k = tf.matmul(x, self.Wk)
        v = tf.matmul(y, self.Wv)

        # split heads
        batch_size, length, hidden_size = tf.shape(x)
        q, k, v = map(
            lambda _: tf.transpose(tf.reshape(_, (batch_size, length, self.num_heads, self.dk)), perm=(0, 2, 1, 3)),
            [q, k, v])

        q *= tf.rsqrt(self.dk)

        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits)
        attention_out = tf.matmul(weights, v)

        attention_out = tf.transpose(attention_out, perm=(0, 2, 1, 3))
        attention_out = tf.reshape(attention_out, shape=(batch_size, length, -1))

        output = tf.matmul(attention_out, self.Wout)

        return output

    def __repr__(self):
        pass


class SelfAttentionLayer(MultiHeadAttentionLayer):
    def forward(self, x):
        return super(SelfAttentionLayer, self).forward(x, x)
