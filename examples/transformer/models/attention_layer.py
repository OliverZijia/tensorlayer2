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
        self.dk = hidden_size // num_heads
        self.keep_prob = keep_prob

        self.build(tuple())
        self._built = True

    def build(self, inputs_shape):
        self.Wq = self._get_weights("W_q", shape=(self.hidden_size, self.hidden_size))
        self.Wk = self._get_weights("W_k", shape=(self.hidden_size, self.hidden_size))
        self.Wv = self._get_weights("W_v", shape=(self.hidden_size, self.hidden_size))
        self.Wout = self._get_weights("Wout", shape=(self.hidden_size, self.hidden_size))

    def forward(self, x, y, mask):
        """
        Parameters
        ----------
        x:
            input to generate query & key, shape=(batch_size, length, hidden_size)
        y:
            input to generate value, shape=(batch_size, length, hidden_size)
        mask:
            mask to fill certain positions with 1e-9 (negative infinity)
            can occur when: (1) it is a padding
                            (2) decoding in Transformer

        Return
        -------
            shape=(batch_size, length, hidden_size)
        """
        q = tf.tensordot(x, self.Wq, axes=[[2], [0]])  # (batch_size, length, hidden_size)
        k = tf.tensordot(y, self.Wk, axes=[[2], [0]])  # (batch_size, length, hidden_size)
        v = tf.tensordot(y, self.Wv, axes=[[2], [0]])  # (batch_size, length, hidden_size)

        # split heads
        batch_size, length, hidden_size = tf.shape(x)
        q, k, v = map(
            lambda _: tf.transpose(tf.reshape(_, (batch_size, length, self.num_heads, self.dk)), perm=(0, 2, 1, 3)),
            [q, k, v])  # (batch_size, num_heads, length, dk)

        q *= tf.math.rsqrt(tf.cast(self.dk, dtype=tf.float32))

        logits = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, length, length)
        logits += mask
        weights = tf.nn.softmax(logits)
        attention_out = tf.matmul(weights, v)  # (batch_size, num_heads, length, dk)

        attention_out = tf.transpose(attention_out, perm=(0, 2, 1, 3))  # (batch_size, length, num_heads, dk)
        attention_out = tf.reshape(attention_out, shape=(batch_size, length, -1))  # (batch_size, length, hidden_size)

        output = tf.tensordot(attention_out, self.Wout, axes=[[2], [0]])  # (batch_size, length, hidden_size)

        return output

    def __repr__(self):
        return "attention layer"


class SelfAttentionLayer(MultiHeadAttentionLayer):
    def forward(self, x, mask):
        return super(SelfAttentionLayer, self).forward(x, x, mask)

    def __repr__(self):
        return "self attention layer"
